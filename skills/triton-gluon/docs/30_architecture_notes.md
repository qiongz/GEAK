# Triton-Gluon Architecture Notes

Worker-routed architecture doc. Read this file when a task mentions target
backend, architecture guards, Triton versions, JIT/AOT, MFMA, WMMA, descriptors,
or prebuilt kernels.

## Profile Routing Hints

- `nv_to_amd_translation`: check target family before carrying over NVIDIA
  layout, async-copy, descriptor, or tensor-memory assumptions. Translation
  should preserve semantics first, then optimize AMD-specific paths.
- `matrix_lowering`: choose CDNA3 MFMA, CDNA4 scaled MFMA, or RDNA WMMA from
  target evidence. Do not infer `instr_shape`, K width, or scale layout from
  local block constants.
- `jit_aot_sensitive`: treat Triton version, target triple, signature, scratch,
  and prebuilt assets as execution contract. Preserve gates unless the same
  benchmark boundary proves they are irrelevant.
- `shape_bucketed_dispatch`: target family and layout version can change which
  shape buckets are valid. Keep arch-sensitive dispatch visible to audit.

## Internal Index

- `gfx942_cdna3`
- `gfx950_cdna4`
- `gfx1250_rdna_wmma`
- `amd_arch_family_quick_directions`
- `### Trait: version_sensitive`
- `### Trait: execution_jit_aot_sensitive`
- `### Trait: operator_support_sensitive`
- `runtime_target_resolution`
- `kernel_path_compatibility`
- `aot_compile_contract`
- `amd_layout_version_map`
- `gfx1250_descriptor_constraints`

## gfx942_cdna3

Typical focus:

- wave64-valid layouts;
- `buffer_load` / `buffer_store`;
- regular MFMA via `AMDMFMALayout(version=3)`;
- attention/decode memory paths;
- explicit blocked layout.

Do not use gfx942 as proof for CDNA4-only scaled MFMA behavior.

## gfx950_cdna4

Typical focus:

- CDNA3-style paths plus newer CDNA4 surfaces;
- `mfma_scaled`;
- scale-layout helpers;
- FP8 / FP4 GEMM.

Do not assume a gfx950 conclusion applies to gfx942.

## gfx1250_rdna_wmma

Typical focus:

- `wmma`;
- `wmma_scaled`;
- `tdm`;
- descriptor flows;
- cluster / barrier behavior;
- `AMDWMMALayout`.

This is not a renamed CDNA path. Do not port CDNA MFMA assumptions blindly.

## amd_arch_family_quick_directions

Use this section to choose the first AMD Gluon direction from the target family.
These are starting points for a valid candidate, not final performance answers.
If an API call or architecture feature is not backed by the routed docs, tests,
or operator-local source, record it as missing detail instead of guessing.

| Target family | First safe mental model | Matrix family | Memory/layout direction | Do not infer |
| --- | --- | --- | --- | --- |
| CDNA3 / `gfx942` | wave64-valid blocked layouts and regular MFMA | `AMDMFMALayout(version=3)` with `mfma` | generic `gl.load` / `gl.store` first, then `buffer_load` / `buffer_store` when the access pattern or existing AMD path justifies it | scaled MFMA or CDNA4-only scale layouts |
| CDNA4 / `gfx950` | CDNA-style layouts plus CDNA4-only checks | `AMDMFMALayout(version=4)`, regular MFMA, and only then scaled MFMA when dtype/scale evidence exists | CDNA3-style blocked/buffer paths may carry over, but async/scaled surfaces need explicit evidence | that a `gfx950` win applies to `gfx942` |
| RDNA / `gfx1250` | separate wave32 WMMA/descriptor family | `AMDWMMALayout(version=3)` and `wmma`; `wmma_scaled` only after plain WMMA works | descriptor/TDM/shared layout constraints are part of correctness | CDNA MFMA layouts, K widths, or buffer-path assumptions |

CDNA3 / `gfx942` quick direction:

- Start from target triple and launch attributes such as `hip:gfx942:64`,
  `num_warps`, `num_ctas`, and optional `waves_per_eu`.
- For 1D memory paths, derive a `BlockedLayout` whose
  `threads_per_warp` product is 64 and whose lane/thread coverage follows the
  contiguous memory dimension.
- For 2D/3D expressions, choose one parent layout per logical expression before
  creating `SliceLayout` or `DotOperandLayout`.
- Use MFMA only for a real matrix hot path. `tl.dot` lowering still requires
  result layout, operand layouts, `convert_layout`, target op, and epilogue
  planning.
- For `AMDMFMALayout`, choose `elem_type` from the accumulator/result layout
  types accepted by the local verifier. Do not reuse fp16/bf16 input operand
  dtype as the MFMA result layout element type.
- Treat MI300X-style CU count and HBM bandwidth as parallelism/bandwidth
  context, not as a universal partition formula.

CDNA4 / `gfx950` quick direction:

- Re-check target arch, layout version, and dtype path before using CDNA4
  features. `AMDMFMALayout(version=4)` and `mfma_scaled` are not a textual
  replacement for CDNA3 MFMA.
- Use regular MFMA or a plain Triton competitor as the anchor unless the task has
  explicit `tl.dot_scaled`, FP8/FP4, scale-layout, or `gfx950` evidence.
- For scaled MFMA, plan operand layouts, scale layouts, scale formats, K width,
  accumulator dtype, and store dtype together. Missing scale evidence should
  stop the patch at regular MFMA or plain Triton.
- Backend or CK evidence for `gfx950` support is enough to document a capability
  boundary, but not enough to invent a concrete Gluon API template.

RDNA / `gfx1250` quick direction:

- Treat `gfx1250` as WMMA/descriptor-first, not CDNA with different names.
- Get plain `wmma` or a basic descriptor path correct before adding
  `wmma_scaled`, `tdm`, async, or cluster behavior.
- Check descriptor rank, contiguous last dimension, valid shared-layout family,
  swizzle/padding restrictions, and scale-factor constraints before tuning.
- Use `hip:gfx1250:32` style target assumptions when checking launch/layout
  compatibility; do not reuse wave64 CDNA layout defaults.
- Full gfx1250 examples can include aggregate memory descriptors, multi-buffer
  shared memory, `tdm.async_load` / `tdm.async_store`, `async_wait`, split-K
  scale layouts, and codegen assertions. Treat those as mature-example evidence,
  not required L0 boilerplate.

Backend capability hints are not first-pass API recipes:

- CDNA3/CDNA4 share several buffer and async-mark lowering paths, but this does
  not mean every CDNA3 namespace call is a CDNA4 scaled feature.
- CDNA4 and GFX1250 expose newer backend capabilities such as permlane swap or
  hardware scaled-upcast support. Treat them as reasons to check source/tests,
  not as permission to add scheduler or shared-memory rewrites to an L0 patch.
- GFX1250 has separate TDM, multi-CTA, direct LDS/scatter, and descriptor
  surfaces. These are capability boundaries for later tasks after plain WMMA or
  descriptor correctness, not defaults for every RDNA Gluon candidate.
- If evidence is backend-only or CK-only, write a verification checklist and
  keep the first candidate on documented Gluon APIs.

### Trait: version_sensitive

Triton minor version can affect:

- `AMDMFMALayout.instr_shape` form;
- Gluon import and JIT behavior;
- AOT metadata compatibility;
- descriptor validation.

When generating version-sensitive code, prefer explicit guards and keep the
fallback path intact.

### Trait: execution_jit_aot_sensitive

Real downstream code can use:

- pure JIT Gluon kernels;
- AOT-compiled Gluon kernels;
- mixed pipelines with both Gluon and normal Triton kernels.

Do not delete JIT/AOT gates without understanding the package and benchmark
contract.

JIT/AOT compatibility is a kernel-path question:

- one operator may use direct JIT;
- another may use AOT assets;
- one family may mix a Gluon main kernel with normal Triton helpers;
- prebuilt kernels can fail across Triton minor versions even when source JIT
  imports work.

Do not delete fallback gates or package-loading behavior just because a local
JIT path compiles.

### Trait: operator_support_sensitive

Global "Gluon available" is not enough. Operator-local support may depend on:

- target backend;
- architecture guard;
- Triton minor version;
- layout version;
- dtype and scale format;
- prebuilt-kernel availability.

For required AMD Gluon tasks, use the supported import path:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

Do not use `from triton import gluon` as an availability probe.

## runtime_target_resolution

Planner and worker tasks should treat target selection as part of the benchmark
contract. GEAK resolves target backend in this order:

1. explicit `target_backend` from CLI, task text, or discovery metadata;
2. `GEAK_TARGET_BACKEND`;
3. best-effort `rocminfo` detection without `sudo`;
4. default `hip/gfx942`.

Normal non-Docker shells often need `GEAK_TARGET_BACKEND` when `rocminfo` is not
available. Docker or ROCm shells can usually detect `gfx*` before profiling
writes `profile.json`.

## kernel_path_compatibility

Architecture is a decision boundary, not a naming convention:

- `gfx942` / CDNA3 is the safest first AMD Gluon target for current GEAK work.
- `gfx950` / CDNA4 can reuse CDNA3-style paths but adds scaled-MFMA and newer
  surfaces; do not backport those conclusions to `gfx942`.
- `gfx1250` is a WMMA/descriptor family with stricter frontend checks, not CDNA
  with renamed APIs.

Module path and architecture version are not always the same thing. Seeing
`gl.amd.cdna3.*` does not prove the whole path targets CDNA3; layout version,
target arch, and feature guards may still point at `gfx950`.

## aot_compile_contract

Gluon AOT is a separate integration contract from JIT. Route AOT tasks here when
the prompt mentions `AOT`, `prebuilt`, `compile_gluon`, `signature`,
`waves_per_eu`, `matrix_instr_nonkdim`, `kpack`, `sanitize_overflow`,
`global_scratch`, or `profile_scratch`.

Required checks:

- Confirm the Triton version and commit. `triton.experimental.gluon` is an
  experimental API surface, and Triton `3.5` / `3.6+` metadata can differ.
- Confirm target triple: backend, arch, and warp size, e.g. `hip:gfx942:64` or
  `hip:gfx1250:32`.
- Confirm launch attributes: `num_warps`, `num_ctas`, `waves_per_eu`, and
  target arch must match layout assumptions.
- Confirm AOT signature: pointer type hints such as `*fp32:16` encode
  divisibility/alignment assumptions; constexpr values disappear from the
  generated C prototype.
- Confirm scratch behavior: current downstream AOT wrappers may reject kernels
  with nonzero `global_scratch_size` or `profile_scratch_size`.

Mixed AOT/JIT pipelines are common. A real operator may compile one Gluon stage
with `GluonASTSource` and a helper/reduce stage with normal Triton `ASTSource`.
Do not debug or classify the whole operator as a single AOT mechanism.

## amd_layout_version_map

Real layout-version mapping:

- `AMDMFMALayout(version=1)` -> `gfx908`
- `AMDMFMALayout(version=2)` -> `gfx90a`
- `AMDMFMALayout(version=3)` -> `gfx942`
- `AMDMFMALayout(version=4)` -> `gfx950`
- `AMDWMMALayout(version=1)` -> RDNA3
- `AMDWMMALayout(version=2)` -> RDNA4
- `AMDWMMALayout(version=3)` -> `gfx1250`

The Python namespace is not the whole architecture contract. Code may use
`gl.amd.cdna3.*` helpers while a layout or feature branch targets `gfx950`.

## gfx1250_descriptor_constraints

Descriptor-style paths are stricter than generic CDNA-style blocked layouts:

- descriptor rank must be between 1 and 5;
- the last tensor dimension must be contiguous;
- only `PaddedSharedLayout`, `SwizzledSharedLayout`, or
  `PartitionedSharedLayout` are valid descriptor shared layouts;
- accepted swizzled descriptor cases may require `max_phase=1`;
- only `"zero"` padding is supported in the current path.
