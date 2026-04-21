"""
Frozen plain Triton input for the first Layer 3 sample.

This file keeps the bounded source semantics from `python/tutorials/01-vector-add.py`:
- `add_kernel`
- `add`
- the tutorial's benchmark intent

It intentionally excludes top-level demo prints and plot rendering because the
Layer 3 harness owns runtime validation.
"""

import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEFAULT_BLOCK_SIZE = 1024


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    assert x.is_contiguous() and y.is_contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output


def benchmark(size: int, provider: str):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    gbps = lambda ms_value: 3 * x.numel() * x.element_size() * 1e-9 / (ms_value * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
