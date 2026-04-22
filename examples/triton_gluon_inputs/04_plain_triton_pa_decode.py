# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal plain Triton paged-attention decode fixture for MI300X.

This fixture is intentionally distilled from the plain Triton paged-attention
decode family in aiter. It keeps only the core decode loop so GEAK can later
rewrite it toward an `amd_gluon` candidate while still fitting into a small
benchmark harness.
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
    {"label": "pa-decode-512", "num_seqs": 8, "query_group_size": 4, "head_size": 128, "block_size": 64, "max_seq_len": 512},
    {"label": "pa-decode-1024", "num_seqs": 16, "query_group_size": 4, "head_size": 128, "block_size": 64, "max_seq_len": 1024},
]
FULL_CASES = [
    *QUICK_CASES,
    {"label": "pa-decode-2048", "num_seqs": 16, "query_group_size": 4, "head_size": 128, "block_size": 64, "max_seq_len": 2048},
]


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


@triton.jit
def paged_attention_decode_kernel(
    out_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    stride_q_seq,
    stride_q_group,
    stride_k_block,
    stride_k_token,
    stride_k_hidden,
    stride_v_block,
    stride_v_token,
    stride_v_hidden,
    stride_bt_seq,
    stride_out_seq,
    stride_out_group,
    QUERY_GROUP_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE_POW2: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_POW2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_POW2: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    q_group_offs = tl.arange(0, QUERY_GROUP_SIZE_POW2)
    head_offs = tl.arange(0, HEAD_SIZE_POW2)
    token_offs = tl.arange(0, BLOCK_SIZE_POW2)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_SIZE)

    q_ptrs = q_ptr + seq_idx * stride_q_seq + q_group_offs[:, None] * stride_q_group + head_offs[None, :]
    q_mask = (q_group_offs[:, None] < QUERY_GROUP_SIZE) & (head_offs[None, :] < HEAD_SIZE)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=tl.float32)
    max_logit = tl.zeros((QUERY_GROUP_SIZE_POW2,), dtype=tl.float32) + float("-inf")
    exp_sum = tl.zeros((QUERY_GROUP_SIZE_POW2,), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    block_table_start = block_table_ptr + seq_idx * stride_bt_seq

    for block_idx in range(num_kv_blocks):
        physical_block = tl.load(block_table_start + block_idx)
        token_base = block_idx * BLOCK_SIZE
        token_mask = (token_offs < BLOCK_SIZE) & (token_base + token_offs < seq_len)

        k_ptrs = (
            k_cache_ptr
            + physical_block * stride_k_block
            + token_offs[:, None] * stride_k_token
            + head_offs[None, :] * stride_k_hidden
        )
        v_ptrs = (
            v_cache_ptr
            + physical_block * stride_v_block
            + token_offs[:, None] * stride_v_token
            + head_offs[None, :] * stride_v_hidden
        )
        kv_mask = token_mask[:, None] & (head_offs[None, :] < HEAD_SIZE)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = tl.where(
            (q_group_offs[:, None] < QUERY_GROUP_SIZE) & token_mask[None, :],
            qk,
            float("-inf"),
        )

        max_logit_new = tl.maximum(tl.max(qk, axis=1), max_logit)
        probs = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit - max_logit_new) * log2e)
        acc *= alpha[:, None]
        acc += tl.dot(probs, v, out_dtype=tl.float32)
        exp_sum = exp_sum * alpha + tl.sum(probs, axis=1)
        max_logit = max_logit_new

    output = acc / exp_sum[:, None]
    out_ptrs = out_ptr + seq_idx * stride_out_seq + q_group_offs[:, None] * stride_out_group + head_offs[None, :]
    out_mask = (q_group_offs[:, None] < QUERY_GROUP_SIZE) & (head_offs[None, :] < HEAD_SIZE)
    tl.store(out_ptrs, output.to(out_ptr.type.element_ty), mask=out_mask)


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
    torch.manual_seed(int(case["max_seq_len"]) + int(case["num_seqs"]))
    num_seqs = int(case["num_seqs"])
    query_group_size = int(case["query_group_size"])
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
    q = torch.randn(num_seqs, query_group_size, head_size, device=DEVICE, dtype=torch.float16)
    k_cache = torch.randn(total_blocks, block_size, head_size, device=DEVICE, dtype=torch.float16)
    v_cache = torch.randn(total_blocks, block_size, head_size, device=DEVICE, dtype=torch.float16)
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "block_size": block_size,
        "query_group_size": query_group_size,
        "head_size": head_size,
    }


def _gather_tokens(cache: torch.Tensor, block_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    gathered = cache.index_select(0, block_ids.to(torch.long)).reshape(-1, cache.shape[-1])
    return gathered[:seq_len]


def reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    query_group_size: int,
    head_size: int,
) -> torch.Tensor:
    output = torch.empty_like(q)
    for seq_idx in range(q.shape[0]):
        seq_len = int(seq_lens[seq_idx].item())
        num_blocks = math.ceil(seq_len / block_size)
        block_ids = block_table[seq_idx, :num_blocks]
        k = _gather_tokens(k_cache, block_ids, seq_len).float()
        v = _gather_tokens(v_cache, block_ids, seq_len).float()
        logits = q[seq_idx].float() @ k.T
        probs = torch.softmax(logits, dim=-1)
        output[seq_idx] = (probs @ v).to(output.dtype)
    return output


def run_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    query_group_size: int,
    head_size: int,
) -> torch.Tensor:
    _require_gfx942()
    output = torch.empty_like(q)
    grid = (q.shape[0],)
    paged_attention_decode_kernel[grid](
        output,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        QUERY_GROUP_SIZE=query_group_size,
        QUERY_GROUP_SIZE_POW2=triton.next_power_of_2(query_group_size),
        HEAD_SIZE=head_size,
        HEAD_SIZE_POW2=triton.next_power_of_2(head_size),
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_POW2=triton.next_power_of_2(block_size),
        num_warps=DEFAULT_NUM_WARPS,
    )
    return output
