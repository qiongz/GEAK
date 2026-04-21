"""
Frozen NV tutorial input for the third Layer 2 sample.

This file keeps only the allowed symbols from `python/tutorials/gluon/05-wgmma.py`:
- `small_mma_kernel`
- `small_mma`
- `test_small_mma`
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


@gluon.jit
def small_mma_kernel(a_desc, b_desc, c_desc, d_desc, LHS_IN_REG: gl.constexpr, INSTR_SHAPE_N: gl.constexpr,
                     num_warps: gl.constexpr):
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    m: gl.constexpr = 16
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )

    gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
    gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))

    if LHS_IN_REG:
        a = a_smem.load(a_reg_layout)
    else:
        a = a_smem

    c = c_smem.load(c_layout)
    d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))

    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(d)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG=False, num_warps=4):
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)

    small_mma_kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        LHS_IN_REG,
        INSTR_SHAPE_N,
        num_warps=num_warps,
    )


@pytest.mark.parametrize("M, N, K", [(64, 32, 32), (64, 256, 128)])
@pytest.mark.parametrize("LHS_IN_REG", [False, True])
@pytest.mark.parametrize("INSTR_SHAPE_N", [16, 64])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_small_mma(M, N, K, LHS_IN_REG, INSTR_SHAPE_N, num_warps):
    maxN = max(N // triton.cdiv(num_warps, triton.cdiv(M, 16)), 8)
    if INSTR_SHAPE_N > maxN:
        pytest.skip(f"INSTR_SHAPE_N={INSTR_SHAPE_N} is too large for M={M}, N={N}, num_warps={num_warps}")

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG, num_warps)
    torch.testing.assert_close(A @ B + C, D, atol=1e-3, rtol=1e-1)
