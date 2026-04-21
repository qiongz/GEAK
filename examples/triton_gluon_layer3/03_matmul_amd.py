"""
AMD Gluon direct lift of the third Layer 3 plain Triton sample.

Lift policy for this sample:
- keep the grouped launch order and 2D pointer arithmetic from the tutorial
- keep masked K-loop semantics
- make the `tl.dot` path explicit as CDNA3 MFMA on `gfx942`
- bound the first pass to the `activation=""` fp16 path and one stable config
"""

import inspect

import torch

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


DEFAULT_BLOCK_SIZE_M = 64
DEFAULT_BLOCK_SIZE_N = 64
DEFAULT_BLOCK_SIZE_K = 32
DEFAULT_GROUP_SIZE_M = 4
DEFAULT_NUM_WARPS = 4
CDNA3_VERSION = 3
CDNA3_INSTR_SHAPE = [32, 32, 8]
CDNA3_K_WIDTH = 4


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


def make_blocked_layout(num_warps: int = DEFAULT_NUM_WARPS):
    return ttgl.BlockedLayout(
        size_per_thread=[4, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


def make_mfma_layout(num_warps: int = DEFAULT_NUM_WARPS):
    return ttgl.amd.AMDMFMALayout(
        version=CDNA3_VERSION,
        instr_shape=CDNA3_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )


@gluon.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: ttgl.constexpr,
    BLOCK_SIZE_N: ttgl.constexpr,
    BLOCK_SIZE_K: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr,
    blocked: ttgl.constexpr,
    mfma_layout: ttgl.constexpr,
    k_width: ttgl.constexpr,
):
    pid = ttgl.program_id(0)
    num_pid_m = ttgl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    ttgl.assume(stride_am > 0)
    ttgl.assume(stride_ak > 0)
    ttgl.assume(stride_bk > 0)
    ttgl.assume(stride_bn > 0)
    ttgl.assume(stride_cm > 0)
    ttgl.assume(stride_cn > 0)

    dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
    dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

    offs_am = (pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked))) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked))) % N
    acc = ttgl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], ttgl.float32, mfma_layout)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_ak = k_start + ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
        offs_bk = k_start + ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(1, blocked))

        offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_a = offs_ak[None, :] < K
        mask_b = offs_bk[:, None] < K

        a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=mask_a, other=0.0, cache=".ca")
        b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=mask_b, other=0.0, cache=".ca")
        a_mfma = ttgl.convert_layout(a, layout=dot_a_layout)
        b_mfma = ttgl.convert_layout(b, layout=dot_b_layout)
        acc = ttgl.amd.cdna3.mfma(a_mfma, b_mfma, acc)

    result = ttgl.convert_layout(acc, layout=blocked).to(ttgl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked))
    offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked))
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.cdna3.buffer_store(stored_value=result, ptr=c_ptr, offsets=offs_c, mask=c_mask, cache=".cs")


def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = ""):
    _require_gfx942()
    assert activation == "", "The bounded Layer 3 AMD matmul path only supports activation=''"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    blocked = make_blocked_layout(DEFAULT_NUM_WARPS)
    mfma_layout = make_mfma_layout(DEFAULT_NUM_WARPS)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        DEFAULT_BLOCK_SIZE_M,
        DEFAULT_BLOCK_SIZE_N,
        DEFAULT_BLOCK_SIZE_K,
        DEFAULT_GROUP_SIZE_M,
        blocked,
        mfma_layout,
        CDNA3_K_WIDTH,
        num_warps=DEFAULT_NUM_WARPS,
    )
    return c


def benchmark(M: int, N: int, K: int, provider: str):
    _require_gfx942()
    a = torch.randn((M, K), device="cuda", dtype=torch.float16) - 0.5
    b = torch.randn((K, N), device="cuda", dtype=torch.float16) - 0.5
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == "amd_gluon":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    tflops = lambda ms_value: 2 * M * N * K * 1e-12 / (ms_value * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)
