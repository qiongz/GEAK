# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal amd_gluon input fixture for Triton-family feature runs."""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

DIALECT = "amd_gluon"
DEFAULT_BLOCK_SIZE = 256
DEFAULT_NUM_WARPS = 4
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


def make_layout(
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_warps: int = DEFAULT_NUM_WARPS,
):
    lanes_per_cta = 64 * num_warps
    if block_size % lanes_per_cta != 0:
        raise ValueError(
            f"block_size={block_size} must be divisible by wave64*num_warps={lanes_per_cta}"
        )
    size_per_thread = block_size // lanes_per_cta
    return ttgl.BlockedLayout(
        size_per_thread=[size_per_thread],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )


@gluon.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: ttgl.constexpr,
    layout: ttgl.constexpr,
):
    pid = ttgl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + ttgl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_elements

    x = ttgl.amd.cdna3.buffer_load(ptr=x_ptr, offsets=offsets, mask=mask, other=0.0, cache=".ca")
    y = ttgl.amd.cdna3.buffer_load(ptr=y_ptr, offsets=offsets, mask=mask, other=0.0, cache=".ca")
    ttgl.amd.cdna3.buffer_store(
        stored_value=x + y,
        ptr=output_ptr,
        offsets=offsets,
        mask=mask,
        cache=".cs",
    )


def make_inputs(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(size)
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)
    return x, y


def reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def run_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    _require_gfx942()
    output = torch.empty_like(x)
    n_elements = output.numel()
    layout = make_layout(block_size=block_size, num_warps=num_warps)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        layout=layout,
        num_warps=num_warps,
    )
    return output
