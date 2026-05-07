# Triton-Gluon Component Traits

Read this file when implementing or planning a concrete optimization direction.
Search policy and task allocation live in `10_search_policies.md`.

## Internal Index

- `### Trait: layout_basic`
- `### Trait: layout_slice_broadcast`
- `### Trait: layout_source_first_required`
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
- Do not omit layout on `gl.arange`, `gl.zeros`, or similar Gluon tensors.

### Trait: layout_slice_broadcast

- Masks, broadcasts, `expand_dims`, `[:, None]`, and slicing often need
  `SliceLayout` or explicit layout conversion.
- Treat mask layout conversion as correctness-sensitive, not cosmetic cleanup.
- Do not apply `[:, None]` to arbitrary incompatible layouts.

### Trait: layout_source_first_required

Read operator-local source before rewriting when you see:

- `DistributedLinearLayout`
- `PartitionedSharedLayout`
- host `TensorDescriptor`
- `reshape`, `permute`, or `trans` used to unshuffle tiles
- nested 3D / 5D layout trees
- JIT/AOT gates or prebuilt-kernel loading

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
  `instr_shape` is expected to be 2D or 3D.
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
