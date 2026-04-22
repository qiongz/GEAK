# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal nv_gluon input fixture for Triton-family feature runs.

This fixture intentionally keeps common Gluon syntax while preserving a few
NVIDIA-oriented assumptions such as a wave32-style layout comment and marker.
GEAK should treat it as an input-only `nv_gluon` sample and, when the Gluon
feature is on, translate toward an `amd_gluon` candidate on AMD targets.
"""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

DIALECT = "nv_gluon"  # nv_gluon marker for dialect inference
DEFAULT_BLOCK_SIZE = 256
DEVICE = triton.runtime.driver.active.get_active_torch_device()


@gluon.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: gl.constexpr,
):
    pid = gl.program_id(0)
    # NVIDIA tutorial-style wave32 assumption; do not copy blindly onto AMD.
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_elements
    x = gl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = gl.load(y_ptr + offsets, mask=mask, other=0.0)
    gl.store(output_ptr + offsets, x + y, mask=mask)


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
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output
