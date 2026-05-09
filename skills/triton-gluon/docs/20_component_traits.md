# Triton-Gluon Component Traits

Read this file when implementing or planning a concrete optimization direction.
Search policy and task allocation live in `10_search_policies.md`.

## Internal Index

- `### Trait: layout_basic`
- `### Trait: layout_slice_broadcast`
- `### Trait: layout_source_first_required`
- `layout_derivation_and_cost_model`
- `### Trait: memory_generic`
- `### Trait: memory_amd_buffer`
- `### Trait: memory_shared_async_descriptor`
- `### Trait: matrix_none`
- `### Trait: matrix_dot`
- `### Trait: matrix_scaled_dot`
- `### Trait: matrix_wmma_descriptor`
- `### Trait: shape_coverage_unknown`
- `### Trait: shape_coverage_single`
- `### Trait: shape_coverage_multi`
- `### Trait: shape_coverage_bucketed`
- `### Trait: shape_layout_constexpr_risk`
- `### Trait: shape_dispatch_required`
- `portable_component_compatibility`
- `api_quick_reference`
- `common_failures_and_fix_order`

### Trait: layout_basic

- Recover `BlockedLayout` from `tl.arange`, tile shape, `num_warps`, target
  family, and coalesced dimension.
- Construct layouts on the host when they depend on launch configuration.
- Pass host-created layouts as `constexpr`.
- Do not instantiate `BlockedLayout`, `SliceLayout`, `DotOperandLayout`, or
  other layout objects inside the `@gluon.jit` body. Creating
  `sl = gl.SliceLayout(...)` inside the kernel can be lowered as a tensor value
  and fail with errors like `cannot convert SliceLayout(...) to tensor`.
- Do not omit layout on `gl.arange`, `gl.zeros`, or similar Gluon tensors.
- In the edited `@gluon.jit` path, replace the whole planned tensor-creation
  chain. Do not mix a `gl.arange(..., layout=...)` for one axis with leftover
  `tl.arange`, `tl.zeros`, `tl.full`, `tl.load`, `tl.where`, or `tl.dot` for
  the same Gluon subpath.
- Device scalar/math in the edited Gluon subpath should also stay in Gluon
  namespace: use `gl.cdiv`, `gl.minimum`, `gl.maximum`, `gl.max`, `gl.sum`,
  `gl.exp`, `gl.where`, etc. If source uses `tl.sigmoid` and local docs do not
  prove `gl.sigmoid`, lower it with documented Gluon math such as
  `1 / (1 + gl.exp(-x))`. Host-side launch math outside `@gluon.jit` may still
  use `triton.cdiv`.
- `BlockedLayout.size_per_thread` should be derived from the tile and launch
  contract. Avoid arbitrary values; on current Gluon layouts values are expected
  to be powers of two and to multiply with `threads_per_warp` and
  `warps_per_cta` to cover the logical tile.

### Trait: layout_slice_broadcast

- Masks, broadcasts, `expand_dims`, `[:, None]`, and slicing often need
  `SliceLayout` or explicit layout conversion.
- Treat mask layout conversion as correctness-sensitive, not cosmetic cleanup.
- Do not apply `[:, None]` to arbitrary incompatible layouts.
- Every broadcast expression has a parent-layout invariant: the 1D tensors that
  become `x[:, None]` and `y[None, :]` must be derived from `SliceLayout`
  objects of the same 2D parent layout used by that expression.
- Create those `SliceLayout(axis, parent)` objects in the host layout factory
  and pass them to the kernel as `gl.constexpr`; do not construct new
  `SliceLayout` objects inside `@gluon.jit`.
- `SliceLayout` creates 1D tensors. Do not directly add or combine two 1D slice
  tensors with different lengths to build a 2D offset. First expand them with
  `[:, None]` and `[None, :]` so both operands broadcast into the same parent
  layout. A compile error like `Cannot make_shape_compatible: incompatible
  dimensions ... 16 and 32` usually means a 1D `[H]` index was combined directly
  with a 1D `[C]` / `[R]` index instead of forming `[H, C]` / `[H, R]`.
- Do not reuse a slice derived from one parent layout in another logical 2D
  context, even if the shape names look compatible. For example, an index
  derived from an `[X, Y]` parent must not be reused in an `[X, Z]` expression.
- Do not rely on `convert_layout(x, SliceLayout(...))` to turn an arbitrary 1D
  tensor into a valid slice for a different parent layout. Generate the index
  with the correct `SliceLayout(parent)` for that expression.
- For kernels with several logical 2D contexts, create separate named index
  tensors such as `idx_x_xy`, `idx_x_xz`, or `idx_x_xw` instead of one shared
  `idx_x` tensor.
- If a broadcast expression cannot name one parent layout shared by both sliced
  axes, the patch scope is too large. Split the task before editing.

### Trait: layout_source_first_required

Read operator-local source before rewriting when you see:

- `DistributedLinearLayout`
- `PartitionedSharedLayout`
- host `TensorDescriptor`
- `reshape`, `permute`, or `trans` used to unshuffle tiles
- nested 3D / 5D layout trees
- JIT/AOT gates or prebuilt-kernel loading

## layout_derivation_and_cost_model

Use this section when the task changes `BlockedLayout`, `SliceLayout`,
`convert_layout`, masks, or the layout used by load/store or matrix operands.
The goal is a correct, performance-plausible first layout, not a promised best
layout.

Derive the base layout in this order:

1. Recover the logical tile from the source: `tl.arange` bounds, pointer
   arithmetic, masks, reductions, matrix dimensions, and launcher constants.
2. Identify the physical contiguous dimension from strides or address
   expressions. Give that dimension the most useful lane/thread coverage first.
3. Pick the target family. CDNA3/CDNA4 paths normally need wave64-valid
   `threads_per_warp` products; gfx1250/RDNA WMMA paths are a separate family and
   should not inherit CDNA wave64 defaults.
4. Check `size_per_thread * threads_per_warp * warps_per_cta` against the
   logical tile per dimension. Do not patch verifier failures by inserting
   arbitrary powers of two.
5. For each logical 2D/3D expression, define one parent layout and derive
   `SliceLayout`, `DotOperandLayout`, shared, or descriptor layouts from that
   parent.
6. Keep shape-, target-, `num_warps`-, or instruction-dependent layouts in
   host-side factories and pass them as `constexpr`.

`convert_layout` decision table:

| Situation | Use `convert_layout`? | Reason |
| --- | --- | --- |
| Moving A/B or Q/K/V operands into `DotOperandLayout` for a real MFMA/WMMA hot path | yes | matrix instructions require operand layouts |
| Re-parenting a 1D slice so it can join a different 2D expression | no | regenerate the index from the correct parent `SliceLayout` |
| Fixing a catastrophic non-coalesced load/store layout after correctness is known | maybe | benchmark against the safe anchor and change one memory path at a time |
| Repeated conversion inside the innermost hot loop | avoid | layout movement can require cross-thread or cross-warp data movement |
| Converting between layouts believed to be equivalent | maybe with `assert_trivial=True` | fail early if the conversion is not a trivial reinterpretation |
| Adding a conversion only so code "looks Gluon-like" | no | it adds cost without a performance hypothesis |

Cost checks before editing:

- Does the conversion happen once per tile, once per loop iteration, or once per
  element group?
- Does it cross warp/thread ownership, or is it a trivial layout reinterpretation?
- Does it enable a measured hot-path improvement such as coalesced memory,
  `buffer_load` / `buffer_store`, MFMA, WMMA, or descriptor access?
- Can the same effect be obtained by creating the index in the right parent
  layout from the start?
- If the patch is low-latency or tiny-stage, is the conversion overhead likely
  larger than the work it removes?
- In multi-CTA layouts, layout-driven operations such as `gl.convert_layout`,
  `gl.reduce`, `gl.sum`, and `gl.max` can imply cross-CTA synchronization.
  Treat this as a separate cost/correctness boundary, and do not place a
  cross-CTA conversion/reduction inside warp-specialized regions unless the
  target tutorial/API explicitly permits it.

When in doubt, keep the first patch smaller: explicit base layout plus generic
memory access, then move one layout, memory, or matrix component at a time under
the existing patch evolution rules.

### Trait: memory_generic

- Start with `gl.load` / `gl.store` for scalar or simple vector paths.
- Do not introduce AMD buffer paths only because the target is AMD.
- Do not add shared-memory staging in the first candidate unless the source
  structure requires it.

### Trait: memory_amd_buffer

Use AMD `buffer_load` / `buffer_store` when the target family, existing AMD
structure, or access pattern benefits.

Good signs:

- streaming KV/cache loads;
- repeated query / accumulator reuse;
- memory-bound profile;
- existing AMD Gluon code path uses buffer operations.

Do not mix CDNA memory assumptions with gfx1250 descriptor paths.

Buffer dtype rules:

- `buffer_load` `other` must be a typed Gluon value compatible with the loaded
  element dtype. Prefer `gl.full(shape, 0.0, ptr.dtype.element_ty, layout=...)`
  over a bare Python literal.
- `buffer_store` `stored_value` must match the destination pointer element dtype.
  Cast accumulators or widened compute values before storing.
- If the task cannot name the loaded dtype, stored dtype, and value layout before
  editing, keep the patch on generic `gl.load` / `gl.store`.

### Trait: memory_shared_async_descriptor

Shared memory, swizzles, async copy, descriptors, `tdm`, and scheduler hints
are later-stage tools. Use them only after a simpler layout/memory candidate is
correct.

Do not translate NVIDIA TMA concepts to AMD by name.

### Trait: matrix_none

If there is no real matrix instruction path, stop at explicit layout and memory
lowering. Do not add MFMA / WMMA just because Gluon is available.

### Trait: matrix_dot

`tl.dot` is not a direct rename target. The AMD Gluon path is:

1. result layout;
2. operand layouts;
3. `convert_layout`;
4. target op such as `mfma` / `wmma`;
5. epilogue correctness.

Do not skip accumulator and operand layout compatibility.
Leaving `tl.dot(...)` inside an `@gluon.jit` body is not an L0 shortcut; it is a
mixed/invalid rewrite for a required pure AMD Gluon task.
Likewise, do not satisfy an MFMA or operand-layout hypothesis by replacing
`tl.dot` with generic `gl.dot`; use the matrix ladder or reduce the task scope.

MFMA-specific checks before editing:

- `AMDMFMALayout.instr_shape` is a matrix instruction shape, not a convenient
  tile shape. Confirm the Triton version and use the supported `(M, N, K)` form
  expected by the installed source.
- Valid CDNA MFMA result shapes are constrained; do not invent `[8, 8]`,
  `[16, 32]`, or other unsupported intrinsic shapes because they match local
  block constants.
- If result layout, operand layouts, and `convert_layout` are not all known,
  keep the task at L0 layout/memory viability instead of attempting MFMA.

### Trait: matrix_scaled_dot

Scaled matrix paths require:

- dtype checks;
- scale layout;
- target architecture;
- instruction shape;
- scale factor constraints.

Do not use scaled MFMA / WMMA on unsupported targets or dtype combinations.

### Trait: matrix_wmma_descriptor

gfx1250 WMMA, descriptor, `tdm`, cluster, and shared-layout rules are a
separate family from CDNA MFMA behavior.

Do not treat gfx1250 as CDNA with renamed APIs.

### Trait: shape_coverage_unknown

The harness did not expose a reliable shape-count signal. Treat any
single-shape speedup as provisional and do not hardcode a tile or layout shape
from an example call.

### Trait: shape_coverage_single

The harness exposes exactly one representative shape. A speedup can be useful
for triage, but do not claim portable performance and do not branch on the
observed shape literal.

### Trait: shape_coverage_multi

Correctness and performance share an ordered stream of multiple cases. Every
shape must pass correctness, and the planner should require at least one
`shape_robust` Base competitor before accepting a Gluon-specific win.

### Trait: shape_coverage_bucketed

Shapes split into distinct regimes. Prefer explicit host-side dispatch that
chooses block sizes, `num_warps`, layouts, or kernel variants from launch
attributes. Do not bury the bucket decision inside `@triton.heuristics`.

### Trait: shape_layout_constexpr_risk

Gluon layouts are usually `constexpr`: `BlockedLayout`, `SliceLayout`,
`AMDMFMALayout.instr_shape`, `DotOperandLayout`, and `warps_per_cta` must match
the launch contract. If they depend on shape or tuning choices, build them on
the host and pass them as `constexpr` arguments.

### Trait: shape_dispatch_required

When shapes are bucketed, prefer host-side dispatch that chooses block size,
num_warps, layout, or kernel variant from launch attributes.

Do not hide bucket selection inside `@triton.heuristics`; audit cannot verify
per-shape coverage from heuristic logs.

## portable_component_compatibility

Usually portable / stackable:

- memory access policy;
- mask or boundary simplification;
- indexing / pointer arithmetic cleanup;
- dtype / cast cleanup;
- explicit shape dispatch evidence.

Usually mutually exclusive without a new design:

- two different tiling schemes;
- 1D accumulator versus 2D accumulator;
- split-K / multipass reduction versus persistent single-kernel scheduling;
- full plain-Triton loop structure versus full Gluon explicit-layout rewrite;
- autotune-key dispatch versus manual host dispatch;
- two launcher signature / constexpr contract changes.

## api_quick_reference

Core rewrite surface:

| Plain Triton pattern | First Gluon rewrite | Escalate when |
| --- | --- | --- |
| `tl.arange(0, X)` | `gl.arange(0, X, layout=layout)` | always; Gluon needs explicit layout |
| `tl.load` / `tl.store` | `gl.load` / `gl.store` | switch to AMD buffer ops only when target family or existing AMD code benefits |
| `tl.zeros(...)` / `tl.full(...)` | `gl.zeros(..., layout=layout)` / `gl.full(..., layout=layout)` | always for distributed tensors |
| `tl.dot` / `tl.dot_scaled` | result layout -> operand layouts -> `convert_layout` -> target matrix op | always; never direct rename |
| descriptor / async / shared-memory paths | target-specific family | only after a simpler layout or matrix candidate is correct |

Common API names workers may need: `BlockedLayout`, `SliceLayout`,
`DotOperandLayout`, `DistributedLinearLayout`, `SwizzledSharedLayout`,
`PaddedSharedLayout`, `PartitionedSharedLayout`, `AMDMFMALayout`,
`AMDWMMALayout`, `convert_layout`, `allocate_shared_memory`, `barrier`,
`to_linear_layout`, and `set_auto_layout`.

## common_failures_and_fix_order

Typical symptoms and first checks:

- Layout or IR verification fails: check `BlockedLayout`, `threads_per_warp`,
  `warps_per_cta`, `order`, `num_warps`, and target arch alignment first.
- `AMDMFMALayout` construction fails: check Triton minor version and whether
  `instr_shape` is expected to be 2D or 3D, then check whether the intrinsic
  shape itself is supported by the AMD layout verifier.
- `GluonSemantic.arange() missing required positional argument: 'layout'`:
  a plain Triton `tl.arange` or layout-less arange survived inside the edited
  Gluon path. Return to the pre-edit layout plan and replace that whole subpath.
- `Did you forget to add @triton.jit`: a helper or nested function is being
  called through the wrong JIT/language boundary. Check that Gluon helpers are
  `@gluon.jit`, host helpers stay on the host, and plain Triton helpers are not
  called as Gluon device functions.
- Kernel compiles but targets the wrong path: re-check backend, arch,
  operator-local guards, and namespace-vs-layout-version assumptions.
- Correctness passes but performance regresses: keep the plain Triton baseline,
  then add AMD memory or matrix features incrementally.
- Preshuffled GEMM gives wrong answers: preserve `DistributedLinearLayout` and
  `reshape` / `permute` / `trans` unshuffle order before tuning.

Suggested fix order:

1. Confirm runtime version, backend, and arch.
2. Confirm launcher and layout alignment.
3. Confirm memory path selection.
4. Confirm matrix layout and `instr_shape`.
5. Add shared-memory, async, descriptor, or scheduler features only after the
   simpler path is correct.
