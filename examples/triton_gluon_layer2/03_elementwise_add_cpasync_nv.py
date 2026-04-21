"""
Frozen NV tutorial input for the second Layer 2 sample.

This file keeps only the allowed symbols from `python/tutorials/gluon/03-async-copy.py`:
- `elementwise_add_cpasync_kernel`
- `elementwise_add_cpasync`
- `test_elementwise_add_cpasync`
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


def is_ampere_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


@gluon.jit
def elementwise_add_cpasync_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_c,
    ystride_c,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    smem_layout: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        cp.async_copy_global_to_shared(a_smem, a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        cp.async_copy_global_to_shared(b_smem, b_ptrs + ystride_b * yoffs[None, :], mask=mask)
        cp.commit_group()
        cp.wait_group(0)

        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)
        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add_cpasync(A, B, C, smem_layout, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK),)
    return elementwise_add_cpasync_kernel[grid](
        A,
        B,
        C,
        xnumel,
        ynumel,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        XBLOCK,
        YBLOCK,
        smem_layout,
    )


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_elementwise_add_cpasync(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    elementwise_add_cpasync(a, b, c, smem_layout, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)
