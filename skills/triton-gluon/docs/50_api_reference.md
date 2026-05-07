# Triton-Gluon API Reference

Read this file when a worker needs concrete API surface, code skeletons, or
failure-driven debug order. It expands the concise rules in
`20_component_traits.md`.

Do not read this whole file by default. Use the routed section(s) below.

## Read Only These Sections

| If the task needs... | Read |
| --- | --- |
| import path, `@gluon.jit`, launcher, host-created layouts | `imports`, `jit_entry_and_host_launcher` |
| exact common Gluon API names (`full`, reductions, shape ops, atomics) | `core_language_and_layout_surface`, `common_language_api_surface` |
| plain Triton -> Gluon mechanical rewrite mapping | `common_rewrite_table` |
| AOT compile, signatures, target triples, scratch failures | `aot_compile_api_surface`, `version_and_compatibility_checklist` |
| shared memory, barriers, async phase ordering | `shared_memory_synchronization_cluster` |
| descriptor/TDM/tensor-memory concepts | `descriptor_and_tensor_memory_surface` |
| `[:, None]`, `[None, :]`, or broadcast layout errors | `slice_broadcast_recipe` |
| AMD MFMA/WMMA/scaled quick syntax | `amd_quick_patterns` |
| NVIDIA TMA/WGMMA/Blackwell recognition only | `nvidia_quick_patterns` |
| failure triage | `common_failures_and_fix_order`; read `common_pitfalls` only if stuck |

## Internal Index

- `imports`
- `jit_entry_and_host_launcher`
- `core_language_and_layout_surface`
- `common_language_api_surface`
- `common_rewrite_table`
- `aot_compile_api_surface`
- `slice_broadcast_recipe`
- `shared_memory_synchronization_cluster`
- `descriptor_and_tensor_memory_surface`
- `amd_quick_patterns`
- `nvidia_quick_patterns`
- `version_and_compatibility_checklist`
- `common_pitfalls`
- `common_failures_and_fix_order`

## imports

Supported Gluon import path:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

AMD-heavy code often aliases the language module:

```python
from triton.experimental.gluon import language as ttgl
```

Do not use `from triton import gluon` as an availability probe. Required AMD
Gluon tasks must fail from a real `triton.experimental.gluon`, `@gluon.jit`, or
`gl.*` patch before fallback is valid evidence.

## jit_entry_and_host_launcher

Gluon keeps Triton's host launcher model: `@gluon.jit`, `kernel[grid](...)`,
`triton.cdiv`, `program_id`, and `constexpr`. The difference is that layouts
may now be host-created and passed as `constexpr`.

Wiring rules before coding:

- Define the Gluon kernel/helper in the edited module before the host dispatch
  imports or calls it. Do not add an import for a guessed `_..._gluon` helper
  that is not present in the patch.
- A Gluon device helper must be `@gluon.jit` and must use Gluon language APIs.
  A plain `@triton.jit` helper is a separate kernel-language boundary, not a
  callable Gluon device helper.
- Keep host-only layout construction, arch checks, and Python dispatch outside
  `@gluon.jit`; pass layouts and constants as `constexpr`.

```python
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

def make_cdna_1d_layout(xblock, num_warps):
    lanes_per_cta = 64 * num_warps
    assert xblock % lanes_per_cta == 0
    return gl.BlockedLayout(
        size_per_thread=[xblock // lanes_per_cta],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )

@gluon.jit
def kernel(x_ptr, y_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel
    other = gl.full([XBLOCK], 0.0, x_ptr.dtype.element_ty, layout=layout)
    x = gl.amd.cdna3.buffer_load(ptr=x_ptr, offsets=offsets, mask=mask, other=other)
    gl.amd.cdna3.buffer_store(ptr=y_ptr, offsets=offsets, stored_value=x.to(y_ptr.dtype.element_ty), mask=mask)

def launch(x, y, XBLOCK=256, num_warps=4, num_ctas=1):
    layout = make_cdna_1d_layout(XBLOCK, num_warps)
    xnumel = x.numel()
    grid = (triton.cdiv(xnumel, XBLOCK),)
    kernel[grid](
        x,
        y,
        xnumel,
        XBLOCK=XBLOCK,
        layout=layout,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
```

Runtime facts:

- `GluonJITFunction` uses `GluonASTSource`.
- `GluonASTSource` sets the language to `Language.GLUON`.
- Launch-time attributes include `ttg.target`, `ttg.num-warps`,
  `ttg.num-ctas`, and `ttg.threads-per-warp`.
- Host launch attributes are part of the correctness contract. A task that
  rewrites only the kernel body while ignoring launcher shape, `num_warps`,
  `num_ctas`, target arch, or host-created layouts is underspecified.
- If a path intentionally stays vendor-neutral, keep the host layout pattern
  and use `gl.load` / `gl.store` instead of AMD buffer ops.

## core_language_and_layout_surface

| API | Purpose |
| --- | --- |
| `program_id` | Triton-like program id |
| `num_programs` | Query grid program count |
| `num_warps` / `num_ctas` | Query launch attributes inside Gluon |
| `warp_specialize` | Warp-specialization control, target-sensitive |
| `constexpr` | Compile-time argument marker |
| `BlockedLayout` | Base thread/warp/CTA distribution |
| `SliceLayout` | Select a sub-dimension from a parent layout |
| `DotOperandLayout` | Operand layout required by matrix instructions |
| `DistributedLinearLayout` | Explicit register and lane mapping |
| `SwizzledSharedLayout` | Shared-memory swizzle |
| `PaddedSharedLayout` | Shared-memory padding semantics |
| `PartitionedSharedLayout` | gfx1250-style partitioned shared layout |
| `AMDMFMALayout` | AMD MFMA result layout |
| `AMDWMMALayout` | AMD WMMA result layout |
| `TensorMemoryLayout` | Blackwell tensor-memory layout |
| `arange` | Layout-aware index construction |
| `full` / `full_like` | Layout-aware tensor creation |
| `load` / `store` | Generic Gluon memory access |
| `zeros(..., layout=...)` | Accumulator/register tensor construction |
| `convert_layout` | Move a tensor into another layout |
| `allocate_shared_memory` | Create shared-memory buffers |
| `barrier` | Synchronization primitive |
| `broadcast`, `expand_dims`, `reshape`, `permute`, `split`, `join`, `ravel` | Shape manipulation with layout implications |
| `sum`, `max`, `min`, `reduce`, `reduce_or` | Reduction APIs used in attention/softmax style kernels |
| `associative_scan`, `histogram` | Scan APIs; treat as specialized, source-first paths |
| `atomic_add`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_xchg`, etc. | Generic atomics; target-specific buffer atomics have stricter arch rules |
| `to_linear_layout` | Convert to a linearized layout view when needed |
| `set_auto_layout` | Allow automatic layout selection in constrained flows |

Recommended order:

1. Choose the target family and result/base layout.
2. Build indices with that layout.
3. Load values.
4. Convert layout when needed.
5. Run target-specific memory or matrix ops.

## common_language_api_surface

The official Gluon common language surface is broader than the small rewrite
subset most first candidates need. Use this section to route workers before
they invent API names.

Common creation and shape APIs:

- `arange`, `zeros`, `zeros_like`, `full`, `full_like`, `cast`, `to_tensor`;
- `broadcast`, `expand_dims`, `reshape`, `permute`, `split`, `join`, `ravel`,
  `map_elementwise`.

Common memory and indexing APIs:

- `load`, `store`;
- `gather`, `where`;
- generic atomics such as `atomic_add`, `atomic_and`, `atomic_cas`,
  `atomic_max`, `atomic_min`, `atomic_or`, `atomic_xchg`, `atomic_xor`.

Common scalar/math APIs used inside `@gluon.jit` paths:

- use `gl.cdiv`, `gl.minimum`, `gl.maximum`, `gl.exp`, `gl.exp2`, `gl.where`,
  `gl.floor`, `gl.ceil`, `gl.sqrt`, `gl.rsqrt`, and `gl.abs`;
- host-side Python/Triton launch math may still use `triton.cdiv`, but device
  scalar/math in the edited Gluon subpath should stay in the `gl.*` namespace.

Common reductions and scans:

- `sum`, `max`, `min`, `reduce`, `reduce_or`, `xor_sum`;
- `associative_scan`, `histogram`.

Planning implications:

- `full` / `zeros` / accumulator creation still need explicit layout.
- `reshape`, `permute`, `split`, `join`, and nested `SliceLayout` usage are
  correctness-sensitive in attention and preshuffle paths.
- `sum` / `max` reductions are common in softmax-style attention kernels and
  should route to this API reference before implementation.
- Atomics are late-stage features. Prefer simpler layout/memory candidates
  first, and use target-specific atomic docs before buffer atomics.

## common_rewrite_table

| Plain Triton pattern | First Gluon rewrite | Escalate when |
| --- | --- | --- |
| `tl.arange(0, XBLOCK)` | `gl.arange(0, XBLOCK, layout=layout)` | Always; Gluon needs explicit layout |
| `tl.load` / `tl.store` | `gl.load` / `gl.store` | Switch to AMD `buffer_load` / `buffer_store` when target family or existing AMD path requires it |
| `tl.zeros((M, N), dtype=...)` | `gl.zeros((M, N), dtype=..., layout=layout)` | Always; accumulators need explicit layout |
| `tl.full(...)` | `gl.full(..., layout=layout)` when supported by local Triton | Always for distributed tensors |
| `tl.cdiv` / `tl.minimum` / `tl.maximum` / `tl.exp` inside device code | `gl.cdiv` / `gl.minimum` / `gl.maximum` / `gl.exp` | Keep host launch math outside `@gluon.jit`; do not leave device scalar/math in `tl.*` |
| `tl.max` / `tl.sum` reductions | `gl.max` / `gl.sum` with matching layout assumptions | Softmax/reduction paths need API reference and correctness audit |
| `reshape` / `permute` / `split` / `join` | Gluon shape APIs with layout-aware tensors | Preserve transformation semantics; source-first for unshuffle paths |
| `tl.atomic_*` | generic `gl.atomic_*` or target-specific buffer atomics | Late-stage only; arch and dtype support must be checked |
| `tl.dot` / `tl.dot_scaled` | result layout + operand layouts + `convert_layout` + target-specific matrix op | Always; not a direct rename target |
| tensor descriptor helpers | target-specific descriptor family | Only when the target family actually supports descriptors or tensor memory |
| implicit shared-memory staging | `allocate_shared_memory` after the first correct candidate | Only after blocked-layout or matrix path is correct |

The table is per planned subpath, not per token. If L0 chooses one RoPE, mask,
load, or matrix subpath, every tensor in that subpath must be layout-aware
Gluon. Leaving one `tl.arange(0, BLOCK_R)` in a RoPE branch while the rest of the
kernel is `@gluon.jit` is still an invalid rewrite.

## slice_broadcast_recipe

Gluon broadcasting is parent-layout sensitive. Treat `[:, None]` and
`[None, :]` as layout operations, not plain NumPy syntax.

Bad pattern:

```python
# H index comes from [H, C], R index comes from [H, R].
# The expanded tensors do not share one parent layout.
head_hc = gl.arange(0, H, layout=gl.SliceLayout(1, blocked_hc))
offs_r = gl.arange(0, R, layout=gl.SliceLayout(0, blocked_hr))
expr = head_hc[:, None] * stride_h + offs_r[None, :]
```

Good pattern:

```python
# Both 1D tensors are slices of the same [H, R] parent layout.
slice_h_hr: gl.constexpr = gl.SliceLayout(1, blocked_hr)
slice_r_hr: gl.constexpr = gl.SliceLayout(0, blocked_hr)
head_hr = gl.arange(0, H, layout=slice_h_hr)
offs_r_hr = gl.arange(0, R, layout=slice_r_hr)
expr = head_hr[:, None] * stride_h + offs_r_hr[None, :]
```

Rules:

- Pick the logical parent layout for each 2D expression first.
- Derive every broadcasted 1D index from `SliceLayout(axis, parent)` of that
  exact parent.
- Use separate index tensors for separate parent contexts (`head_hc`,
  `head_hr`, etc.).
- Avoid `convert_layout` as a way to re-parent a 1D tensor before
  `[:, None]` / `[None, :]`; regenerate the index from the right parent layout.
- Mask tensors follow the same rule. Mask layout mismatches are correctness
  problems, not cleanup opportunities.

## aot_compile_api_surface

Downstream aiter code uses a real Gluon AOT compiler path. Do not treat AOT as
equivalent to JIT.

Important compile-time fields:

- `path`: Python source containing the Gluon kernel; the file is executed from
  its parent directory.
- `kernel_name`: function name; the function must be a real `@gluon.jit`
  kernel and expose `is_gluon()`.
- `signature`: comma-separated argument types and constexpr values. Pointer
  signatures may carry divisibility hints such as `*fp32:16`; constexpr values
  are removed from the generated C prototype.
- `grid`: launch grid for the generated entry point.
- `target`: `<backend>:<arch>:<warp-size>`, e.g. `hip:gfx942:64`.
- `num_warps`, `num_ctas`, `waves_per_eu`, `num_stages`.
- `matrix_instr_nonkdim`, `kpack`.
- `enable_fp_fusion`, `allow_flush_denorm`, `sanitize_overflow`,
  `launch_cooperative_grid`, `debug`.

AOT failure boundaries:

- AOT wrappers may reject kernels with `global_scratch_size` or
  `profile_scratch_size`.
- HIP AOT compile checks may prove that a HSACO is produced without proving the
  whole package/link/runtime path is valid.
- Mixed pipelines may compile a Gluon stage with `GluonASTSource` and a normal
  Triton helper stage with `ASTSource`; route and debug those stages separately.

## shared_memory_synchronization_cluster

Concrete shared-memory staging pattern:

```python
shared_a: gl.constexpr = gl.SwizzledSharedLayout(
    vec=16,
    per_phase=1,
    max_phase=16,
    order=[1, 0],
)
shared_b: gl.constexpr = gl.SwizzledSharedLayout(
    vec=16,
    per_phase=1,
    max_phase=16,
    order=[0, 1],
)

smem_a = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a)
smem_b = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b)

smem_a.store(a)
smem_b.store(b)
gl.barrier()

cur_a = smem_a.load(layout=dot_a_layout)
cur_b = smem_b.load(layout=dot_b_layout)
acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
```

Common concepts:

- `allocate_shared_memory`
- `barrier`
- `mbarrier`
- `cluster`
- `fence_async_shared`
- `warp_pipeline_stage`

If a target family adds an async path, preserve the high-level phase structure:

1. issue async transfer;
2. commit;
3. wait;
4. consume from shared memory or continue compute.

Do not copy literal async function names between NVIDIA, CDNA, and `gfx1250`
families. The phase structure is common; APIs are not.

## descriptor_and_tensor_memory_surface

NVIDIA-side concepts:

- host `TensorDescriptor`
- device-side `tma`
- `tensor_memory_descriptor`
- `TensorMemoryLayout`
- `tcgen05_*`

AMD-side concepts:

- gfx1250 host `TensorDescriptor`
- `tdm`
- `async_load`
- `async_wait`
- `prefetch`
- `async_scatter`
- `PartitionedSharedLayout`

These are not 1:1 renames. Treat descriptor setup as a correctness contract,
not as a late cosmetic optimization.

## amd_quick_patterns

Use AMD-specific memory or matrix patterns only after the layout plan is known.
For MFMA, the plan must name result layout, operand layouts, `convert_layout`,
target op, and epilogue/store layout. If no correctness-passing Gluon anchor
exists, keep the task at L0 layout/memory viability instead of trying MFMA over
the whole original kernel.

Buffer op dtype preconditions:

- `buffer_load(..., other=...)` should use a typed Gluon tensor with the same
  element dtype/layout as the loaded value, for example
  `gl.full(shape, 0.0, ptr.dtype.element_ty, layout=layout)`. Do not rely on a
  bare Python float literal when the verifier expects `other.dtype`.
- `buffer_store(..., stored_value=...)` must store a value whose element dtype
  matches the destination pointer element type. Cast before store when the
  compute value is an accumulator dtype.
- Generic `gl.load` / `gl.store` is the first candidate path; move to
  `buffer_load` / `buffer_store` only after these dtype and layout preconditions
  are named in the implementation plan.

### CDNA3 MFMA pattern

```python
blocked = ttgl.BlockedLayout(
    size_per_thread=[4, 4],
    threads_per_warp=[4, 16],
    warps_per_cta=[num_warps, 1],
    order=[1, 0],
)
mfma_layout = ttgl.amd.AMDMFMALayout(
    version=3,
    instr_shape=[32, 32, 8],
    transposed=True,
    warps_per_cta=[num_warps, 1],
)
a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a)
b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, mfma_layout, 4))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, mfma_layout, 4))
acc = ttgl.zeros([M, N], ttgl.float32, mfma_layout)
c = ttgl.amd.cdna3.mfma(a, b, acc)
```

### CDNA4 scaled-MFMA pattern

```python
mfma_layout = ttgl.amd.AMDMFMALayout(
    version=4,
    instr_shape=[16, 16, 32],
    transposed=True,
    warps_per_cta=[1, 4],
)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, mfma_layout, 16))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, mfma_layout, 16))
a_scale = ttgl.amd.cdna4.get_mfma_scale_layout(a.type.layout, [M, K])
b_scale = ttgl.amd.cdna4.get_mfma_scale_layout(b.type.layout, [K, N])
c = ttgl.amd.cdna4.mfma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
```

### gfx1250 WMMA pattern

```python
wmma_layout = ttgl.amd.AMDWMMALayout(
    3,
    True,
    [[0, 1], [1, 0]],
    [],
    [16, 16, instr_shape_k],
)
a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, wmma_layout, k_width))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, wmma_layout, k_width))
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

gfx1250 scaled-WMMA constraints that are easy to miss:

- Get plain `wmma` working before adding scaled WMMA.
- `wmma_scaled` expects stricter layout contracts than plain `wmma`.
- When an input format is `e2m1`, operand WMMA layout expects
  `instr_shape=[16, 16, 64]`.
- The accumulator layout for scaled WMMA expects
  `instr_shape=[16, 16, 128]`.
- `get_wmma_scale_layout` only supports scale factor `16` or `32`.
- Scale dtype combinations are limited; do not guess by name.

### gfx1250 descriptor constraints

Practical frontend-check rules:

- descriptor shape rank must be between 1 and 5;
- last tensor dimension must be contiguous;
- only `PaddedSharedLayout`, `SwizzledSharedLayout`, or
  `PartitionedSharedLayout` are valid descriptor layouts;
- currently accepted swizzled cases may require `max_phase=1`;
- only `"zero"` padding is supported.

## nvidia_quick_patterns

### Ampere or Hopper async copy

```python
cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
cp.commit_group()
cp.wait_group(0)
value = smem.load(layout)
```

### Hopper WGMMA pattern

```python
fence_async_shared()
acc = warpgroup_mma(a, b_smem, acc, is_async=True)
warpgroup_mma_wait(acc, deps=(...))
```

### Blackwell tensor-memory or TCGen05 pattern

```python
tmem = allocate_tensor_memory(...)
tcgen05_copy(...)
tcgen05_mma(...)
tcgen05_commit(...)
```

These patterns are for recognition and translation review. Do not rename them
into AMD APIs.

## version_and_compatibility_checklist

| Check | Why it matters | Typical questions |
| --- | --- | --- |
| Triton minor version | `instr_shape` and some AOT metadata differ across versions | Is this code path expecting `3.5` or `3.6+`? |
| `AMDMFMALayout.instr_shape` form | 2D vs 3D affects construction | Should this be `[M, N]` or `[M, N, K]`? |
| Execution mode | downstream code may use JIT, AOT, or both | Does this need `triton.experimental.gluon`, prebuilt kernels, or both? |
| Target backend and arch | wave size, memory path, and matrix family depend on it | `hip/gfx942`, `hip/gfx950`, `hip/gfx1250`, or CUDA? |
| Global feature availability | coarse helpers do not define per-operator support | Does "Gluon available" mean this operator is supported? |
| Operator-local support matrix | real kernels may have narrower guards | Does this attention/GEMM path support the arch? |
| AOT scratch requirements | some AOT pipelines reject scratch requirements | Does the kernel need global or profile scratch? |

Treat Triton version, execution mode, and target architecture as benchmark and
integration contract, not incidental metadata.

## common_pitfalls

General:

- Do not start from copied tutorial code without checking the target family.
- Do not treat compile or IR validation as proof that profiling will work.
- Do not mix environment changes, harness changes, and kernel changes in one
  unexplained iteration.

NVIDIA-side:

- Do not assume `tma` concepts port directly to AMD.
- Do not assume Hopper and Blackwell matrix paths are interchangeable.
- Do not forget async ordering or fence semantics around shared-memory paths.

AMD-side:

- Do not mix `mfma` and `wmma` mental models.
- Do not assume descriptor-style paths are uniformly available across CDNA3,
  CDNA4, and `gfx1250`.
- Do not assume every benchmark-valid task is profiler-ready.
- Do not assume namespace name alone determines the architecture contract.
- Do not treat host-side layout construction as optional when layout depends on
  launch configuration.
- Do not simplify away `DistributedLinearLayout` or unshuffle transforms without
  proving the algorithm is preserved.
- Do not guess `wmma_scaled` scale formats, scale factor, or accumulator layout.

## common_failures_and_fix_order

| Symptom | Inspect first | Typical fix |
| --- | --- | --- |
| layout or IR verification fails | `BlockedLayout`, `threads_per_warp`, `warps_per_cta`, `order`, `num_warps`, target arch | make layout consistent with launch contract |
| `AMDMFMALayout` or `instr_shape` construction fails | Triton version, 2D vs 3D `instr_shape`, supported intrinsic shape, layout version | add a real version guard and match expected layout form |
| `GluonSemantic.arange() missing required positional argument: 'layout'` | edited `@gluon.jit` path for leftover `tl.arange` or layout-less arange | replace the whole planned index subpath with `gl.arange(..., layout=...)` |
| `Did you forget to add @triton.jit` | helper called from the wrong JIT/language boundary | use `@gluon.jit` for Gluon device helpers and keep host helpers outside the kernel |
| import error for `_..._gluon` helper | module wiring and definitions in the patch | define the helper before host dispatch imports or calls it |
| `BlockedLayout size_per_thread` verifier failure | tile size, `threads_per_warp`, `warps_per_cta`, and power-of-two values | recompute layout from launch contract instead of patching arbitrary integers |
| `buffer_load other dtype` or `other has no dtype` | `other` argument and loaded pointer element type | create `other` with `gl.full(shape, value, ptr.dtype.element_ty, layout=...)` |
| `buffer_store stored_value` dtype mismatch | stored value dtype and destination pointer element type | cast the computed value to `out_ptr.dtype.element_ty` before `buffer_store` |
| kernel compiles but target path is wrong | backend, arch, operator-local guards, namespace vs layout version | re-check the operator support matrix |
| JIT Gluon import is missing | whether downstream expects JIT, AOT, or both | preserve or add fallback path instead of deleting it |
| AOT compilation fails with scratch-related error | `global_scratch_size` or `profile_scratch_size` | keep JIT for that path or redesign the kernel |
| correctness passes but performance regresses | baseline comparison, memory path choice, shared-memory staging order | keep plain Triton baseline and add AMD features incrementally |
| `gfx1250` descriptor construction fails | last-dimension contiguity, layout family, swizzle settings, padding mode | satisfy descriptor assertions first |
| `wmma_scaled` fails or asserts | input format, operand `instr_shape`, accumulator layout, scale factor, scale dtype combination | get plain `wmma` working first |
| preshuffled GEMM gives wrong answers | `DistributedLinearLayout`, unshuffle `reshape` / `permute` / `trans`, K divisibility assumptions | preserve transformation sequence before tuning |

Suggested debug order:

1. Confirm runtime version, backend, and arch.
2. Confirm launcher and layout alignment.
3. Confirm memory path selection.
4. Confirm matrix layout and `instr_shape`.
5. Only then add shared-memory, async, descriptor, or scheduler features.
