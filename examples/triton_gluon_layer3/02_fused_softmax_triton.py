"""
Frozen plain Triton input for the second Layer 3 sample.

This file keeps the bounded source semantics from `python/tutorials/02-fused-softmax.py`:
- `softmax_kernel`
- `softmax`
- the tutorial's correctness / benchmark intent

It intentionally leaves plotting and top-level demo execution to the Layer 3
harness and run manifest.
"""

import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEFAULT_MAX_PROGRAMS = 1024


def naive_softmax(x: torch.Tensor):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]


def _pick_num_warps(block_size: int):
    for num_warps in (8, 4, 2, 1):
        if block_size >= 64 * num_warps and block_size % (64 * num_warps) == 0:
            return num_warps
    return 1


def _pick_num_stages(block_size: int):
    return 4 if block_size <= 1024 else 2


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor, max_programs: int = DEFAULT_MAX_PROGRAMS):
    assert x.device == DEVICE
    assert x.ndim == 2
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    num_warps = _pick_num_warps(block_size)
    num_stages = _pick_num_stages(block_size)
    y = torch.empty_like(x)
    num_programs = min(n_rows, max_programs)
    softmax_kernel[(num_programs,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=block_size,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y


def benchmark(M: int, N: int, provider: str):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    elif provider == "naive_softmax":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    gbps = lambda ms_value: 2 * x.numel() * x.element_size() * 1e-9 / (ms_value * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
