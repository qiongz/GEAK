# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal plain Triton input fixture for Gluon feature runs.

This file is intentionally small and acts only as an input fixture. GEAK may
keep the result in plain Triton or generate an `amd_gluon` candidate depending
on the active Triton-family feature policy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

DIALECT = "plain_triton"
DEFAULT_BLOCK_SIZE = 256
DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
    tl.store(output_ptr + offsets, x + y, mask=mask)


def make_inputs(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(size)
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)
    return x, y


def reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def run_kernel(x: torch.Tensor, y: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output
