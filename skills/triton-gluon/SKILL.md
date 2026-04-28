---
name: triton-gluon
description: Rewrite or optimize Triton-family kernels (`plain_triton`, `nv_gluon`, or `amd_gluon`) toward valid AMD-facing Gluon candidates. Use when the task involves Gluon, `input_dialect`, AMD Triton layouts, MFMA or WMMA paths, or translating NVIDIA-facing Gluon assumptions to AMD.
tier: general
---

# Triton feature: Gluon

## When to use

- The task stays on the Triton route: `kernel_type = triton`.
- The input dialect is `plain_triton`, `nv_gluon`, or `amd_gluon`.
- The task is to rewrite, optimize, or benchmark a kernel, not only explain API
  names.

## Immediate checklist

Do **not** stop at summarizing the skill. Apply it to the current kernel:

1. Classify the input as `plain_triton`, `nv_gluon`, or `amd_gluon`.
2. Preserve source semantics first:
   - launcher shape
   - indexing
   - masks / boundary behavior
   - correctness oracle
   - benchmark intent
3. Check the runtime contract:
   - installed Triton version
   - whether `triton.experimental.gluon` JIT imports exist
   - whether the codebase depends on prebuilt AOT kernels
   - whether `AMDMFMALayout.instr_shape` is 2D or 3D
   - target backend and architecture
   - whether "Gluon available" and "this operator supports the target" are
     different questions
4. Check the host-side integration contract:
   - wrapper ABI and tensor shape assumptions
   - host-created layouts passed as `constexpr`
   - `num_warps`, `num_ctas`, block sizes, and target arch alignment
   - arch guards such as `gfx942`, `gfx950`, `gfx1250`
   - JIT/AOT fallback gates and prebuilt-kernel loading behavior
5. Choose the action path for the current dialect.
6. Benchmark the resulting candidate against the source baseline or a permitted
   fallback path.

## Output contract

- Keep `kernel_type = triton`.
- Valid optimized outputs:
  - `plain_triton`
  - `amd_gluon`
- Never create a new optimized `nv_gluon` output path.
- Treat Gluon as an additive Extension Set on top of the Base Triton search.
  Do not replace all plain Triton candidates with Gluon candidates.
- If `amd_gluon` is allowed and structurally promising, generate an
  `amd_gluon` candidate early instead of spending the whole plan on fallback
  tuning.
- Treat plain Triton winning the benchmark as a valid outcome, not as a failure
  to use Gluon.

## Multi-shape correctness == performance

Modern Triton/Gluon harnesses (AgentKernelArena PR #32, AITER repository
tasks) score correctness and performance on the **same ordered case
stream**. When the harness exposes more than one shape, GEAK preprocess
classifies the kernel as `shape_coverage_multi` or `shape_coverage_bucketed`
and the planner auto-injects a `## Shape Coverage Working Set` block listing
the first six observed cases.

In that mode:

- A patch that wins one shape but regresses another is **not** a valid
  optimization. The next planner round receives `Per-shape regressions on:
  [...]` and forces a shape-robust competitor; if the shape regression
  survives the audit it is recorded as `Gluon-positive-shape-regressed`
  and rejected.
- Hardcoded shape literals (`if M == 4096`, `BLOCK_M = 128` baked into
  layout `instr_shape`, autotune predicates that switch on shape) are
  prohibited. SYSTEM_PROMPT rule 17 enforces task self-classification:
  `single_shape_viability`, `shape_robust`, or `shape_bucketed`. Pick one
  honestly.
- For `shape_coverage_bucketed`, prefer **host-side dispatch** that picks
  block size, num_warps, layout, or even kernel variant from launch
  attributes. Do not bury the bucket selection in a `@triton.heuristics`
  lambda - the audit treats heuristic mutation as `config-shifted` even
  when aggregate speedup looks positive.
- For `shape_layout_constexpr_risk`, build Gluon layouts on the host from
  launch attributes and pass them as `constexpr`. A layout that bakes one
  shape into `AMDMFMALayout.instr_shape` will fail multi-shape correctness
  the moment the next bucket dispatches.

When the profile is `shape_coverage_unknown` or `shape_coverage_single`,
optimize for correctness first; treat any single-shape speedup as
provisional until the harness is upgraded to multi-shape.

## Planner traits, not kernel families

When planning tasks, prefer composable traits over hard-coded kernel family
labels. Kernel-family examples are useful, but the first planning layer should
be:

- dialect traits:
  - `plain_triton -> amd_gluon`
  - `nv_gluon -> amd_gluon`
  - `amd_gluon -> amd_gluon`
- layout traits:
  - base `BlockedLayout`
  - `SliceLayout` / broadcast / mask conversion
  - source-first layouts such as `DistributedLinearLayout`,
    `PartitionedSharedLayout`, host `TensorDescriptor`, 3D or 5D layout trees
- memory traits:
  - generic `gl.load` / `gl.store`
  - AMD `buffer_load` / `buffer_store`
  - shared memory, swizzle, descriptor, `tdm`, async copy
- matrix traits:
  - no matrix path
  - `tl.dot` / `tl.dot_scaled`
  - scaled MFMA / WMMA
  - gfx1250 descriptor / WMMA path
- execution traits:
  - JIT vs AOT
  - Triton minor-version compatibility
  - target arch and operator-local support matrix

Candidate task slots should follow the main GEAK planner style:

- Prefer first:
  - Base Set plain Triton candidate
  - semantics-preserving AMD Gluon viability candidate
  - plain Triton fallback or competitor when allowed
- Consider next:
  - Shared Set strategy as a `plain_triton variant`, `amd_gluon variant`, or
    paired comparison
  - memory lowering after a correct layout candidate
  - matrix lowering after result and operand layouts are clear
  - in-dialect optimization for existing AMD Gluon
- Deprioritize until later:
  - descriptor, async, scheduler, persistent, atomics, work stealing
  - launch-only or autotune-only changes
  - any path that relies on compile-only success

## Input-dialect playbook

### `plain_triton`

Use this exact order:

1. Keep the launcher recognizable.
2. Recover the implicit layout from `tl.arange`, tile shape, `num_warps`, and
   wave32 or wave64 assumptions.
3. If layout depends on launch config, build it on the host and pass it as a
   `constexpr`; keep the launcher in `kernel[grid](...)` form.
4. Decide the memory path:
   - simple scalar or vector path: prefer `gl.load` / `gl.store`
   - explicit AMD global-memory path or code already using AMD Gluon idioms:
     use `ttgl.amd.cdna3|cdna4.buffer_load` / `buffer_store`
5. Decide the matrix path:
   - if there is no real matrix instruction path, stop at explicit blocked
     layout plus AMD memory ops
   - if the kernel needs MFMA or WMMA, define `AMDMFMALayout` or
     `AMDWMMALayout`, then `DotOperandLayout`, then `convert_layout`, then the
     target-specific op
6. Add shared memory, swizzle, async copy, barriers, or scheduler hints only as
   second-stage tuning.
7. Keep plain Triton as a valid competitor unless AMD Gluon is clearly
   promising.

### `nv_gluon`

- Treat this as translation, not rename.
- Preserve common Gluon control flow, launcher shape, indexing, masks, and
  correctness scaffolding.
- Re-evaluate before carrying to AMD:
  - wave32-centric layouts
  - NVIDIA async-copy paths
  - `tma` / tensor descriptor assumptions
  - Hopper or Blackwell-specific matrix instructions
  - tensor-memory and cluster features
- Translate to AMD-facing Gluon first, then optimize.

### `amd_gluon`

- Preserve the AMD-facing structure unless benchmark evidence clearly favors a
  fallback comparison.
- Optimize inside AMD Gluon space first.
- Read operator-local architecture guards before assuming the support matrix
  from a global helper.

## API decision rules

- `tl.arange` is not enough by itself in Gluon; rewrite it as
  `gl.arange(..., layout=...)`.
- `tl.load` / `tl.store` usually become `gl.load` / `gl.store` first. Move to
  AMD `buffer_load` / `buffer_store` only when the target family or existing
  kernel structure actually requires it.
- `tl.zeros` becomes `gl.zeros(..., layout=...)`.
- `tl.dot` or `tl.dot_scaled` is **not** a direct rename target. On AMD, the
  real path is usually result layout -> operand layouts -> `convert_layout` ->
  `mfma` / `mfma_scaled` / `wmma`.
- A useful translator-derived mental model is:
  source op traits -> target family -> layout helper -> op lowering.
  For example, CDNA paths choose `AMDMFMALayout` and gfx1250 paths choose
  `AMDWMMALayout`; both still require operand layouts and `convert_layout`.
- Descriptor, tensor-memory, and async-copy concepts are vendor- and
  family-specific. Do not translate by name alone.

## Architecture notes

- `gfx942` / CDNA3:
  - prefer wave64-valid layouts
  - `buffer_load` / `buffer_store`
  - `mfma` only when the kernel is genuinely matrix-op based
- `gfx950` / CDNA4:
  - CDNA3-style paths may still appear
  - `mfma_scaled` and scale-layout helpers are high-value tools
  - use CDNA4-only async features only when required
- `gfx1250`:
  - treat `wmma`, `tdm`, cluster, and related APIs as a separate family
  - do not use them as drop-in replacements for CDNA behavior
  - prefer plain `wmma` before `wmma_scaled`
  - descriptor paths have harder constraints than CDNA layouts: contiguous last
    dimension, limited layout families, and stricter shared-memory rules
  - `wmma_scaled` has hard constraints on `instr_shape`, accumulator layout,
    scale dtype combinations, and scale factor
- `AMDMFMALayout(version=3)` maps to `gfx942`; `version=4` maps to `gfx950`.
- `AMDWMMALayout(version=3)` maps to `gfx1250`.
- The Python namespace is not the whole contract. Real code may still use
  `gl.amd.cdna3.*` helpers while layout version or feature branches target
  `gfx950`.
- A global "Gluon available" helper is not the same thing as "this operator is
  supported on this arch".

## Practical patterns from real code

- Real aiter attention kernels combine:
  - `BlockedLayout`
  - `SliceLayout`
  - `AMDMFMALayout(version=CDNA_VERSION, ...)`
  - `DotOperandLayout`
  - shared-memory allocation and swizzle
  - `buffer_load` / `buffer_store`
- Real aiter GEMM kernels on `gfx950` additionally use:
  - `mfma_scaled`
  - `get_mfma_scale_layout`
  - JSON or heuristic-selected launch configs instead of only online autotune
- Preshuffled GEMM paths may use `DistributedLinearLayout` plus
  `reshape` / `permute` / `trans` to unshuffle operand tiles. Treat that as
  algorithm structure, not cosmetic cleanup.
- Real attention kernels often carry 3D or 5D logical shapes, multiple
  `SliceLayout` layers, and mask conversions across layouts.
- JIT and AOT can coexist in one operator family.
- Optional scheduler or barrier hints are second-stage tuning tools, not
  first-pass portability requirements.

## Optimization path heuristics

- elementwise or simple vector kernels:
  - first get explicit blocked layout + correct launcher
  - then decide whether generic `gl.load` / `gl.store` is enough
  - move to AMD `buffer_load` / `buffer_store` only if the target family or the
    surrounding AMD code path really benefits
- attention or decode kernels:
  - preserve stride-rich host arguments, mask semantics, and partition logic
  - keep query, key, value layout trees intact before changing memory paths
  - treat layout conversions on masks or logits as part of correctness, not
    cleanup
- GEMM or FP8 kernels:
  - first match the real matrix family, K width, and result layout
  - then wire operand layouts and epilogue scaling
  - only then add shared-memory staging, preshuffle logic, async, or config
    tuning
- `gfx1250` WMMA or descriptor kernels:
  - first get plain `wmma` or basic descriptor use correct
  - add `wmma_scaled`, scale layouts, `tdm`, or cluster behavior only after the
    base path works

## When to read source

Stop treating the task as a generic rewrite and read operator-local source when
you see any of these:

- `DistributedLinearLayout`
- `PartitionedSharedLayout` or host `TensorDescriptor`
- `reshape` / `permute` / `trans` used to unshuffle matrix tiles
- 3D or 5D logical layouts with multiple nested `SliceLayout`
- AOT packaging, env-variable gates, or prebuilt-kernel loading
- scheduler, barrier, or priority hints that appear to affect correctness or
  launch shape

## Anti-patterns

- Introducing a new top-level `gluon` kernel type
- Producing a new optimized `nv_gluon` output path
- Keeping NVIDIA layout assumptions unchanged on AMD
- Renaming NVIDIA APIs to guessed AMD names
- Textually replacing `tl.dot` with an AMD matrix op without result and operand
  layouts
- Using MFMA when the kernel has no real matrix trait
- Starting with descriptor, async, persistent, scheduler, atomics, or work
  stealing before a simpler AMD Gluon candidate passes correctness
- Treating compile-only success as enough to claim the rewrite is valid
- Ignoring Triton minor-version compatibility when the codebase mixes JIT and
  AOT
- Assuming one global arch check is the operator support matrix
- Copying checked-in manifests, snapshots, or benchmark outputs into the answer
- Hardcoding shape literals (`if M == 4096`, `BLOCK_M = 128` baked into a
  layout `instr_shape`) on a kernel that the harness reports as
  `shape_coverage_multi` or `shape_coverage_bucketed`; this is the
  Issue #30 cheating shape and the audit will reject it
- Hiding shape-bucket selection inside a `@triton.heuristics` lambda when
  the planner trait is `shape_dispatch_required`; use host-side dispatch so
  the audit can verify per-shape coverage

## Read next

- `docs/triton_gluon.md`
  - first search for `## Quick section map for agents`
  - then search for the matching stable heading, for example
    `### Trait: matrix_dot` or `### Trait: dialect_nv_gluon`
  - read only that trait section and the headings it names; do not read the
    entire guide unless the task remains ambiguous
- `examples/triton_gluon_inputs/README.md`
