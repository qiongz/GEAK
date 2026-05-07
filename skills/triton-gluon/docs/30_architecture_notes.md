# Triton-Gluon Architecture Notes

Read this file when a task mentions target backend, architecture guards,
Triton versions, JIT/AOT, MFMA, WMMA, descriptors, or prebuilt kernels.

## Internal Index

- `gfx942_cdna3`
- `gfx950_cdna4`
- `gfx1250_rdna_wmma`
- `### Trait: version_sensitive`
- `### Trait: execution_jit_aot_sensitive`
- `### Trait: operator_support_sensitive`
- `runtime_target_resolution`
- `kernel_path_compatibility`
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
