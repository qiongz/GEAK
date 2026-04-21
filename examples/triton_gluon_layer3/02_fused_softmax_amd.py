"""
AMD Gluon direct lift of the second Layer 3 plain Triton sample.

Lift policy for this sample:
- keep row-wise softmax semantics and the numerical-stability shift
- make the row layout explicit for `gfx942` wave64 execution
- lower row memory traffic through CDNA3 buffer load/store
- keep irregular-column masking, but use a bounded one-row-per-program launcher
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


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


def _pick_num_warps(block_size: int):
    for num_warps in (8, 4, 2, 1):
        if block_size >= 64 * num_warps and block_size % (64 * num_warps) == 0:
            return num_warps
    return 1


def make_row_layout(block_size: int, num_warps: int):
    lanes_per_cta = 64 * num_warps
    if block_size % lanes_per_cta != 0:
        raise ValueError(
            f"block_size={block_size} must be divisible by wave64*num_warps={lanes_per_cta}"
        )
    return ttgl.BlockedLayout(
        size_per_thread=[block_size // lanes_per_cta],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )


@gluon.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: ttgl.constexpr,
    row_layout: ttgl.constexpr,
):
    row_idx = ttgl.program_id(0)
    col_offsets = ttgl.arange(0, BLOCK_SIZE, layout=row_layout)
    mask = col_offsets < n_cols

    input_offsets = row_idx * input_row_stride + col_offsets
    row = ttgl.amd.cdna3.buffer_load(
        ptr=input_ptr,
        offsets=input_offsets,
        mask=mask,
        other=-float("inf"),
        cache=".ca",
    )
    row_max = ttgl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = ttgl.exp(row_minus_max)
    denominator = ttgl.sum(numerator, axis=0)
    softmax_output = numerator * (1.0 / denominator)

    output_offsets = row_idx * output_row_stride + col_offsets
    ttgl.amd.cdna3.buffer_store(
        stored_value=softmax_output,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=mask,
        cache=".cs",
    )


def softmax(x: torch.Tensor):
    _require_gfx942()
    assert x.ndim == 2
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    num_warps = _pick_num_warps(block_size)
    row_layout = make_row_layout(block_size, num_warps)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        block_size,
        row_layout,
        num_warps=num_warps,
    )
    return y


def benchmark(M: int, N: int, provider: str):
    _require_gfx942()
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    elif provider == "amd_gluon":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    gbps = lambda ms_value: 2 * x.numel() * x.element_size() * 1e-9 / (ms_value * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
