---
name: triton-gluon
description: Use when optimizing a Triton-family kernel (`plain_triton`, `nv_gluon`, or `amd_gluon`) and you need to classify the input dialect, optionally translate NVIDIA-facing Gluon, and generate or refine an `amd_gluon` candidate on AMD.
tier: general
---

# Triton feature: Gluon

## When to use

- The task stays on the Triton route: `kernel_type = triton`.
- The input dialect is `plain_triton`, `nv_gluon`, or `amd_gluon`.
- The task should automatically optimize or rewrite the kernel rather than only
  explain the API surface.

## What to do immediately after load

Do **not** stop at summarizing the skill. Apply it to the current kernel:

1. Read the task metadata and classify the input as:
   - `plain_triton`
   - `nv_gluon`
   - `amd_gluon`
2. Preserve source semantics first:
   - launcher shape
   - indexing
   - masks / boundary behavior
   - correctness oracle
   - benchmark intent
3. Choose the action path:
   - `plain_triton`: decide whether an `amd_gluon` candidate is structurally
     promising; if yes, generate one early
   - `nv_gluon`: translate vendor-specific APIs, layouts, or hardware idioms
     into AMD-facing Gluon before tuning
   - `amd_gluon`: keep the optimization path in AMD Gluon space and refine it
4. Benchmark the resulting candidate against the source baseline or permitted
   fallback path.

## Output contract

- Keep `kernel_type = triton`.
- Valid optimized outputs:
  - `plain_triton`
  - `amd_gluon`
- Never create a new optimized `nv_gluon` output path.
- If `amd_gluon` is allowed and structurally promising, prefer an early
  `amd_gluon` candidate instead of spending the whole plan on fallback-only
  tuning.

## Runtime and version checks

- Gluon in Triton uses a distinct runtime path through `GluonASTSource`, so host
  launch shape and attributes like `num_warps`, `num_ctas`, and target arch are
  part of the contract.
- Real downstream code may use:
  - pure Gluon JIT
  - Gluon AOT
  - mixed pipelines with Gluon plus plain Triton stages
- Before rewriting anything, check:
  - installed Triton version
  - whether Gluon JIT imports are available
  - whether the codebase depends on prebuilt AOT kernels
  - whether `AMDMFMALayout.instr_shape` expects old 2D or newer 3D form

## Input-dialect playbook

### `plain_triton`

- Recover implicit layout and memory decisions before lowering into Gluon.
- Generate an `amd_gluon` candidate only when explicit layouts or AMD primitives
  have a real chance to help.
- Keep plain Triton as a valid competitor when Gluon looks unlikely to pay off.

### `nv_gluon`

- Separate common Gluon syntax from NVIDIA-only assumptions.
- Preserve the common subset whenever it still makes sense on AMD.
- Re-evaluate wave32-centric layouts, async-copy paths, descriptor paths, or
  vendor-specific matrix APIs before carrying them over.
- Translate to AMD-facing Gluon, then optimize.

### `amd_gluon`

- Preserve the AMD-facing structure unless benchmark evidence clearly favors a
  fallback comparison.
- Optimize inside AMD Gluon space first.

## Layout and API notes

- Treat layout as a first-class design choice.
- Common high-value building blocks:
  - `BlockedLayout`
  - `SliceLayout`
  - `DotOperandLayout`
  - `DistributedLinearLayout`
  - `SwizzledSharedLayout`
  - `PaddedSharedLayout`
  - `PartitionedSharedLayout`
  - `AMDMFMALayout`
  - `AMDWMMALayout`
- Common execution helpers:
  - `allocate_shared_memory`
  - `convert_layout`
  - `barrier`
  - `mbarrier`
  - `cluster`
  - `fence_async_shared`
  - `warp_pipeline_stage`
- Descriptor and tensor-memory concepts are vendor- and family-specific. Do not
  assume `tma`, `tdm`, tensor descriptors, or tensor memory are interchangeable
  by name.

## Architecture notes

- `gfx942` / CDNA3:
  - prefer wave64-valid layouts
  - `buffer_load`
  - `buffer_store`
  - `mfma` only when the kernel is genuinely matrix-op based
- `gfx950` / CDNA4:
  - use CDNA4-only async-copy or scaled-MFMA paths only when they are actually
    required
  - `mfma_scaled` and scale-layout helpers are CDNA4-specific high-value tools
- `gfx1250` / RDNA-style paths:
  - treat `wmma` / `tdm` / cluster APIs as a separate family, not as drop-in
    substitutes for CDNA behavior
- NVIDIA Ampere / Hopper / Blackwell:
  - Ampere centers on `async_copy` and `mma_v2`
  - Hopper adds `tma`, `warpgroup_mma`, `cluster`, and richer barriers
  - Blackwell adds tensor-memory and `tcgen05_*` paths
- `AMDMFMALayout(version=3)` maps to `gfx942`; `version=4` maps to `gfx950`.
- `AMDWMMALayout(version=3)` maps to `gfx1250`.
- Do not assume the Python namespace alone tells you the full architecture
  contract. Real code may still use `gl.amd.cdna3.*` helpers while the layout or
  version branch targets `gfx950`.

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
  - JSON or heuristic-selected launch configs rather than only online autotune
- Optional scheduler or barrier hints may appear in production code. Treat them
  as second-stage tuning tools, not first-pass portability requirements.

## Anti-patterns

- Introducing a new top-level `gluon` kernel type
- Keeping NVIDIA layout assumptions unchanged on AMD
- Renaming NVIDIA APIs to guessed AMD names
- Treating compile-only success as enough to claim the rewrite is valid
- Ignoring Triton minor-version compatibility when the codebase mixes JIT and AOT
- Copying checked-in manifests, snapshots, or benchmark outputs into the answer

## Related docs

- `docs/triton_gluon.md`
- `examples/triton_gluon_inputs/README.md`
