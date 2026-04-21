"""
AMD Gluon direct lift of the first Layer 3 plain Triton sample.

Lift policy for this sample:
- keep the one-dimensional launcher and masking semantics from plain Triton
- make the layout explicit for `gfx942` wave64 execution
- lower memory traffic through CDNA3 buffer load/store
- keep benchmark intent, but leave runtime orchestration to the Layer 3 harness
"""

import inspect

import torch

import triton
from triton._C.libtriton import ir
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


def _ensure_hip_gluon_runtime_compat():
    # The validated MI3xx baseline still needs a small runtime patch for HIP
    # Gluon kernels when the older runtime shape is present.
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


DEFAULT_BLOCK_SIZE = 1024
DEFAULT_NUM_WARPS = 4


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


def make_gfx942_layout(block_size: int = DEFAULT_BLOCK_SIZE, num_warps: int = DEFAULT_NUM_WARPS):
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
    output = x + y
    ttgl.amd.cdna3.buffer_store(stored_value=output, ptr=output_ptr, offsets=offsets, mask=mask, cache=".cs")


def add(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_warps: int = DEFAULT_NUM_WARPS,
):
    _require_gfx942()
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()

    output = torch.empty_like(x)
    n_elements = output.numel()
    layout = make_gfx942_layout(block_size=block_size, num_warps=num_warps)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        block_size,
        layout,
        num_warps=num_warps,
    )
    return output


def benchmark(size: int, provider: str, block_size: int = DEFAULT_BLOCK_SIZE, num_warps: int = DEFAULT_NUM_WARPS):
    _require_gfx942()
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == "amd_gluon":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y, block_size=block_size, num_warps=num_warps),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    gbps = lambda ms_value: 3 * x.numel() * x.element_size() * 1e-9 / (ms_value * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
