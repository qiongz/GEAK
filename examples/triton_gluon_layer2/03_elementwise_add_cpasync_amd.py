"""
AMD translation of the second Layer 2 NV Gluon sample.

This sample starts from an explicit NVIDIA `cp.async` input but intentionally
degrades to a synchronous `gfx942` implementation. The goal is to preserve
elementwise-add semantics on CDNA3 without inventing a fake AMD `cp.async`
equivalent.
"""

import inspect

import triton
from triton._C.libtriton import ir
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


def _ensure_hip_gluon_runtime_compat():
    from triton.backends.amd.compiler import HIPOptions
    from triton.compiler.code_generator import ast_to_ttir
    from triton.compiler.compiler import make_backend
    from triton.experimental.gluon import _runtime as gluon_runtime

    if not hasattr(HIPOptions, "maxnreg"):
        HIPOptions.maxnreg = None

    make_ir = gluon_runtime.GluonASTSource.make_ir
    if getattr(make_ir, "_geak_hip_compat", False):
        return

    if len(inspect.signature(make_ir).parameters) != 5:
        return

    def make_ir_hip_compat(self, options, codegen_fns, module_map, context):
        builder = ir.builder(context)
        module = builder.create_module()

        target = triton.runtime.driver.active.get_current_target()
        backend = make_backend(target)
        target_name = backend.get_target_name(options)

        module.set_attr("ttg.target", builder.get_string_attr(target_name))
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
        module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(getattr(options, "warp_size", 32)))

        maxnreg = getattr(options, "maxnreg", None)
        if getattr(options, "backend_name", None) == "cuda" and maxnreg is not None:
            module.set_attr("ttg.maxnreg", builder.get_int32_attr(maxnreg))

        return ast_to_ttir(
            self.fn,
            self,
            context=context,
            options=options,
            codegen_fns=codegen_fns,
            module_map=module_map,
            module=module,
        )

    make_ir_hip_compat._geak_hip_compat = True
    gluon_runtime.GluonASTSource.make_ir = make_ir_hip_compat


_ensure_hip_gluon_runtime_compat()


DEFAULT_XBLOCK = 32
DEFAULT_YBLOCK = 64
DEFAULT_NUM_WARPS = 2


def make_gfx942_layout(num_warps=DEFAULT_NUM_WARPS):
    return ttgl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[1, 64],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )


def make_default_smem_layout():
    # Retained at the host-call boundary for source compatibility with the NV
    # cp.async launcher even though gfx942 uses a degraded synchronous path.
    return ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])


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
    XBLOCK: ttgl.constexpr,
    YBLOCK: ttgl.constexpr,
    layout: ttgl.constexpr,
):
    pid = ttgl.program_id(0)

    xoffs = pid * XBLOCK + ttgl.arange(0, XBLOCK, layout=ttgl.SliceLayout(dim=1, parent=layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + ttgl.arange(0, YBLOCK, layout=ttgl.SliceLayout(dim=0, parent=layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        # gfx942 does not provide a first-sample `cp.async` equivalent. Keep the
        # row/block traversal and elementwise semantics, but load synchronously.
        a_val = ttgl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = ttgl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)
        c_val = a_val + b_val

        ttgl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add_cpasync(
    A,
    B,
    C,
    smem_layout=None,
    XBLOCK=DEFAULT_XBLOCK,
    YBLOCK=DEFAULT_YBLOCK,
    num_warps=DEFAULT_NUM_WARPS,
    layout=None,
):
    assert A.shape == B.shape == C.shape
    if smem_layout is None:
        smem_layout = make_default_smem_layout()
    assert smem_layout is not None

    xnumel, ynumel = A.shape
    layout = make_gfx942_layout(num_warps) if layout is None else layout
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
        layout,
        num_warps=num_warps,
    )
