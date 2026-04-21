"""
AMD translation of the third Layer 2 NV Gluon sample.

This sample starts from a Hopper WGMMA/TMA-style small matmul input and rewrites
it into a `gfx942` CDNA3 MFMA path. The translation preserves matmul semantics
and the host/test interface, but intentionally drops:
- `TensorDescriptor`
- `tma`
- `mbarrier`
- `warpgroup_mma`

The AMD path uses direct buffer loads and MFMA accumulation instead.
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


DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_K = 32
SUPPORTED_NUM_WARPS = (4, 8)
SUPPORTED_INSTR_SHAPE_N = (16, 64)
CDNA3_VERSION = 3
CDNA3_INSTR_SHAPE = [32, 32, 8]
CDNA3_K_WIDTH = 4


def _pick_block_n(n):
    return 32 if n <= 32 else 64


def _make_blocked_layout(block_n, num_warps):
    return ttgl.BlockedLayout(
        size_per_thread=[4, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


def _make_mfma_layout(num_warps):
    return ttgl.amd.AMDMFMALayout(
        version=CDNA3_VERSION,
        instr_shape=CDNA3_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )


def _max_instr_shape_n(M, N, num_warps):
    m_reps = triton.cdiv(M, 16)
    n_reps = triton.cdiv(num_warps, m_reps)
    return max(N // n_reps, 8)


def _validate_compat_knobs(M, N, INSTR_SHAPE_N, LHS_IN_REG, num_warps):
    if num_warps not in SUPPORTED_NUM_WARPS:
        raise ValueError(f"gfx942 MFMA rewrite only supports num_warps in {SUPPORTED_NUM_WARPS}, got {num_warps}")
    if INSTR_SHAPE_N not in SUPPORTED_INSTR_SHAPE_N:
        raise ValueError(
            f"gfx942 MFMA rewrite keeps INSTR_SHAPE_N only as a compatibility knob; "
            f"supported values are {SUPPORTED_INSTR_SHAPE_N}, got {INSTR_SHAPE_N}"
        )
    max_instr_shape_n = _max_instr_shape_n(M, N, num_warps)
    if INSTR_SHAPE_N > max_instr_shape_n:
        raise ValueError(
            f"INSTR_SHAPE_N={INSTR_SHAPE_N} is too large for M={M}, N={N}, num_warps={num_warps}; "
            f"maximum compatible value is {max_instr_shape_n}"
        )
    if not isinstance(LHS_IN_REG, bool):
        raise TypeError(f"LHS_IN_REG must be bool, got {type(LHS_IN_REG).__name__}")


@gluon.jit
def small_mma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
    LHS_IN_REG: ttgl.constexpr,
    INSTR_SHAPE_N: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    blocked: ttgl.constexpr,
    k_width: ttgl.constexpr,
    mfma_layout: ttgl.constexpr,
):
    pid_n = ttgl.program_id(0)
    # Preserve the NV-facing kernel signature. On gfx942, CDNA3 MFMA exposes
    # neither a separate LHS-in-register toggle nor a variable instr_shape_n
    # knob, so the host launcher validates these parameters and lowers them to a
    # fixed MFMA instruction shape.

    dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
    dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

    offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked))
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked))
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    offs_c = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    offs_d = offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn

    c_tile = ttgl.amd.cdna3.buffer_load(ptr=c_ptr, offsets=offs_c, mask=mask_c, other=0.0)
    acc = ttgl.convert_layout(c_tile, layout=mfma_layout)

    for k_start in range(0, K, BLOCK_K):
        offs_ak = k_start + ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, blocked))
        offs_bk = k_start + ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(1, blocked))

        offs_a = offs_m[:, None] * stride_am + offs_ak[None, :] * stride_ak
        offs_b = offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn

        mask_a = (offs_m[:, None] < M) & (offs_ak[None, :] < K)
        mask_b = (offs_bk[:, None] < K) & (offs_n[None, :] < N)

        a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=mask_a, other=0.0)
        b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=mask_b, other=0.0)

        a_mfma = ttgl.convert_layout(a, layout=dot_a_layout)
        b_mfma = ttgl.convert_layout(b, layout=dot_b_layout)
        acc = ttgl.amd.cdna3.mfma(a_mfma, b_mfma, acc)

    result = ttgl.convert_layout(acc, layout=blocked)
    ttgl.amd.cdna3.buffer_store(stored_value=result, ptr=d_ptr, offsets=offs_d, mask=mask_c)


def small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG=False, num_warps=4):
    assert A.ndim == 2 and B.ndim == 2 and C.ndim == 2 and D.ndim == 2
    assert A.shape[0] == C.shape[0] == D.shape[0]
    assert B.shape[1] == C.shape[1] == D.shape[1]
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    _, N = B.shape
    _validate_compat_knobs(M, N, INSTR_SHAPE_N, LHS_IN_REG, num_warps)
    block_n = _pick_block_n(N)
    blocked = _make_blocked_layout(block_n, num_warps)
    mfma_layout = _make_mfma_layout(num_warps)
    grid = (triton.cdiv(N, block_n),)

    return small_mma_kernel[grid](
        A,
        B,
        C,
        D,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        D.stride(0),
        D.stride(1),
        LHS_IN_REG,
        INSTR_SHAPE_N,
        DEFAULT_BLOCK_M,
        block_n,
        DEFAULT_BLOCK_K,
        blocked,
        CDNA3_K_WIDTH,
        mfma_layout,
        num_warps=num_warps,
    )
