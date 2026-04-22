# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal plain Triton paged-MQA logits fixture for MI300X.

This fixture trims the original paged MQA logits family down to a single-stage
kernel that still exposes the dominant qk matmul path and paged-KV addressing.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

DIALECT = "plain_triton"
ATOL = 2e-2
RTOL = 2e-2
DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEFAULT_NUM_WARPS = 4

QUICK_CASES = [
    {"label": "mqa-logits-512", "num_seqs": 8, "query_heads": 8, "head_size": 128, "block_size": 64, "max_seq_len": 512},
    {"label": "mqa-logits-1024", "num_seqs": 16, "query_heads": 8, "head_size": 128, "block_size": 64, "max_seq_len": 1024},
]
FULL_CASES = [
    *QUICK_CASES,
    {"label": "mqa-logits-2048", "num_seqs": 16, "query_heads": 16, "head_size": 128, "block_size": 64, "max_seq_len": 2048},
]


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


@triton.jit
def paged_mqa_logits_kernel(
    out_ptr,
    q_ptr,
    k_cache_ptr,
    block_table_ptr,
    weights_ptr,
    seq_lens_ptr,
    stride_q_seq,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_hidden,
    stride_bt_seq,
    stride_w_seq,
    stride_out_seq,
    stride_out_head,
    stride_out_token,
    QUERY_HEADS: tl.constexpr,
    QUERY_HEADS_POW2: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_POW2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_POW2: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_offs = tl.arange(0, QUERY_HEADS_POW2)
    hidden_offs = tl.arange(0, HEAD_SIZE_POW2)
    token_offs = tl.arange(0, BLOCK_SIZE_POW2)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_SIZE)

    q_ptrs = q_ptr + seq_idx * stride_q_seq + head_offs[:, None] * stride_q_head + hidden_offs[None, :]
    q_mask = (head_offs[:, None] < QUERY_HEADS) & (hidden_offs[None, :] < HEAD_SIZE)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    weight_ptrs = weights_ptr + seq_idx * stride_w_seq + head_offs
    weights = tl.load(weight_ptrs, mask=head_offs < QUERY_HEADS, other=0.0).to(tl.float32)

    block_table_start = block_table_ptr + seq_idx * stride_bt_seq

    for block_idx in range(num_kv_blocks):
        physical_block = tl.load(block_table_start + block_idx)
        token_base = block_idx * BLOCK_SIZE
        token_mask = (token_offs < BLOCK_SIZE) & (token_base + token_offs < seq_len)

        k_ptrs = (
            k_cache_ptr
            + physical_block * stride_k_block
            + token_offs[:, None] * stride_k_token
            + hidden_offs[None, :] * stride_k_hidden
        )
        kv_mask = token_mask[:, None] & (hidden_offs[None, :] < HEAD_SIZE)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        logits = tl.dot(q, k.T, out_dtype=tl.float32)
        logits = tl.maximum(logits, 0.0) * weights[:, None]

        out_ptrs = (
            out_ptr
            + seq_idx * stride_out_seq
            + head_offs[:, None] * stride_out_head
            + (token_base + token_offs[None, :]) * stride_out_token
        )
        out_mask = (head_offs[:, None] < QUERY_HEADS) & token_mask[None, :]
        tl.store(out_ptrs, logits.to(out_ptr.type.element_ty), mask=out_mask)


def _build_block_table(seq_lens: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    max_blocks = math.ceil(int(seq_lens.max().item()) / block_size)
    block_table = torch.zeros((seq_lens.numel(), max_blocks), device=DEVICE, dtype=torch.int32)
    next_block = 0
    for seq_idx, seq_len in enumerate(seq_lens.tolist()):
        num_blocks = math.ceil(int(seq_len) / block_size)
        block_table[seq_idx, :num_blocks] = torch.arange(
            next_block, next_block + num_blocks, device=DEVICE, dtype=torch.int32
        )
        next_block += num_blocks
    return block_table, next_block


def make_inputs(case: dict[str, int]) -> dict[str, object]:
    _require_gfx942()
    torch.manual_seed(int(case["max_seq_len"]) + int(case["query_heads"]))
    num_seqs = int(case["num_seqs"])
    query_heads = int(case["query_heads"])
    head_size = int(case["head_size"])
    block_size = int(case["block_size"])
    max_seq_len = int(case["max_seq_len"])

    seq_lens = torch.randint(
        low=max_seq_len // 2,
        high=max_seq_len + 1,
        size=(num_seqs,),
        device=DEVICE,
        dtype=torch.int32,
    )
    block_table, total_blocks = _build_block_table(seq_lens, block_size)
    q = torch.randn(num_seqs, query_heads, head_size, device=DEVICE, dtype=torch.float16)
    k_cache = torch.randn(total_blocks, block_size, head_size, device=DEVICE, dtype=torch.float16)
    weights = torch.rand(num_seqs, query_heads, device=DEVICE, dtype=torch.float16)
    return {
        "q": q,
        "k_cache": k_cache,
        "block_table": block_table,
        "weights": weights,
        "seq_lens": seq_lens,
        "block_size": block_size,
        "query_heads": query_heads,
        "head_size": head_size,
    }


def _gather_tokens(cache: torch.Tensor, block_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    gathered = cache.index_select(0, block_ids.to(torch.long)).reshape(-1, cache.shape[-1])
    return gathered[:seq_len]


def reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    query_heads: int,
    head_size: int,
) -> torch.Tensor:
    output = torch.zeros(
        q.shape[0],
        q.shape[1],
        int(seq_lens.max().item()),
        device=q.device,
        dtype=q.dtype,
    )
    for seq_idx in range(q.shape[0]):
        seq_len = int(seq_lens[seq_idx].item())
        num_blocks = math.ceil(seq_len / block_size)
        block_ids = block_table[seq_idx, :num_blocks]
        k = _gather_tokens(k_cache, block_ids, seq_len).float()
        logits = q[seq_idx].float() @ k.T
        logits = torch.relu(logits) * weights[seq_idx].float()[:, None]
        output[seq_idx, :, :seq_len] = logits.to(output.dtype)
    return output


def run_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    query_heads: int,
    head_size: int,
) -> torch.Tensor:
    _require_gfx942()
    output = torch.zeros(
        q.shape[0],
        q.shape[1],
        int(seq_lens.max().item()),
        device=q.device,
        dtype=q.dtype,
    )
    grid = (q.shape[0],)
    paged_mqa_logits_kernel[grid](
        output,
        q,
        k_cache,
        block_table,
        weights,
        seq_lens,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        block_table.stride(0),
        weights.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        QUERY_HEADS=query_heads,
        QUERY_HEADS_POW2=triton.next_power_of_2(query_heads),
        HEAD_SIZE=head_size,
        HEAD_SIZE_POW2=triton.next_power_of_2(head_size),
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_POW2=triton.next_power_of_2(block_size),
        num_warps=DEFAULT_NUM_WARPS,
    )
    return output
