---
layer: "3"
category: "triton"
subcategory: "gluon"
tags: ["triton", "gluon", "rocm", "amd", "kernel", "layout", "mfma", "wmma", "gfx942", "gfx950", "gfx1250"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: false
last_updated: 2026-04-22
difficulty: "expert"
estimated_time: "60min"
---

# Triton Gluon on AMD GPUs

Guide to using Triton Gluon on AMD GPUs through ROCm, with emphasis on layout-driven kernel design, AMD matrix paths, and version-sensitive behavior.

**This documentation targets ROCm 7.0+ only.**

**Important scope note**: in practice, Gluon feature availability depends more directly on the installed Triton build than on ROCm alone. Treat the Triton version and source tree as part of the compatibility contract.

**Primary public references**:

- Triton Gluon tutorials: [https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)
- Triton Gluon intro tutorial: [https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py)
- Triton Gluon layouts tutorial: [https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/02-layouts.py](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/02-layouts.py)
- Triton repository: [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
- ROCm blog for plain Triton on AMD GPUs: [https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)

## Overview

Gluon is a lower-level GPU programming language built on the same compiler stack as Triton. It keeps Triton's host-side launcher model, but exposes more of the device-side decisions that plain Triton often leaves implicit:

- explicit tensor layouts
- explicit shared-memory layouts and staging
- target-specific memory paths
- target-specific matrix instruction paths
- more visible JIT versus AOT trade-offs

On AMD GPUs, this matters most for kernels where performance depends on controlling:

- blocked layouts and slice layouts
- MFMA or WMMA operand mapping
- shared-memory swizzle or partitioning
- descriptor or tensor-memory-like flows on supported targets

This is **not** a new top-level kernel type. Think of Gluon as a Triton-family extension for lower-level kernels.

In GEAK, Triton kernels default to an automatic candidate search that can
include AMD Gluon. The `off` and `force` controls are primarily for ablation or
forced-path debugging; normal inputs do not need to declare an input dialect or
explicitly turn Gluon on.

## When to Use Gluon

Stay with plain Triton first when:

- the kernel is simple and already expresses the right execution shape
- `@triton.jit` plus autotuning is likely enough
- the optimization target is mostly launch configuration or fusion

Escalate to Gluon when:

- layout choice is central to performance
- you need explicit control over register, lane, warp, or block mapping
- you need AMD matrix paths such as MFMA, scaled MFMA, or WMMA
- shared-memory staging, swizzle, or descriptor paths are part of the design
- a production kernel already uses Gluon and you are modifying or porting it

## Core Mental Model

### Same launcher model, stricter device contract

The host-side launcher remains Triton-like:

- `@gluon.jit`
- `kernel[grid](...)`
- `triton.cdiv`
- `num_warps`
- `num_ctas`

The big difference is that device code no longer treats layout as implicit. In Gluon:

- `gl.arange(..., layout=...)` carries layout information into later operations
- tensors and accumulators often need explicit layouts
- matrix instructions require explicit result and operand layouts
- launch attributes and layout legality are tied together during IR verification

### Layout is first-class

The most important shift from plain Triton to Gluon is that layout becomes a design object rather than an emergent property.

Common layout concepts on AMD Gluon paths:

| Layout or helper | Purpose |
|------------------|---------|
| `BlockedLayout` | Base thread, warp, and CTA distribution |
| `SliceLayout` | View or index a sub-dimension of a parent layout |
| `DotOperandLayout` | Operand layout required by matrix instructions |
| `DistributedLinearLayout` | Explicit register and lane mapping |
| `SwizzledSharedLayout` | Shared-memory swizzle for conflict-aware staging |
| `PaddedSharedLayout` | Shared-memory layout with padding semantics |
| `PartitionedSharedLayout` | gfx1250-style partitioned shared-memory layout |
| `AMDMFMALayout` | AMD MFMA result layout |
| `AMDWMMALayout` | AMD WMMA result layout |

### JIT and AOT are both real

Real Gluon deployments may use:

- pure JIT Gluon kernels
- AOT-compiled Gluon kernels
- mixed pipelines where only some stages are Gluon

Do not assume every downstream project is JIT-only.

## AMD Target Families

Current Triton Gluon practice on AMD is best understood by target family rather than by one generic "AMD" bucket.

| Family | Typical focus | Practical notes |
|--------|----------------|-----------------|
| `gfx942` / CDNA3 | `buffer_load`, `buffer_store`, `mfma`, `AMDMFMALayout(version=3)` | Good first target for MI3xx-class Gluon bring-up |
| `gfx950` / CDNA4 | CDNA3-style paths plus `mfma_scaled`, scale-layout helpers, newer async surfaces | Some code still uses `gl.amd.cdna3.*` names while targeting `version=4` layouts |
| `gfx1250` | `wmma`, `wmma_scaled`, `tdm`, descriptor flows, cluster or barrier APIs, `AMDWMMALayout(version=3)` | Treat as a separate family, not "CDNA with renamed APIs" |

### Architecture mapping

Current layout-version mappings that matter in Gluon code:

- `AMDMFMALayout(version=3)` -> `gfx942`
- `AMDMFMALayout(version=4)` -> `gfx950`
- `AMDWMMALayout(version=3)` -> `gfx1250`

Read the actual target arch, layout version, and version guards together. The Python namespace alone does not tell the full story.

## API Decision Table

This is the most useful mental table for `plain_triton -> amd_gluon` rewrites.

| Plain Triton pattern | First Gluon rewrite | Escalate when |
|----------------------|---------------------|---------------|
| `tl.arange(0, XBLOCK)` | `gl.arange(0, XBLOCK, layout=layout)` | Always. Gluon needs the layout to be explicit. |
| `tl.load` / `tl.store` | `gl.load` / `gl.store` | Move to AMD `buffer_load` / `buffer_store` when the target family or the existing AMD code path actually depends on it. |
| `tl.zeros((M, N), dtype=...)` | `gl.zeros((M, N), dtype=..., layout=layout)` | Always. Accumulator layout matters. |
| `tl.dot` / `tl.dot_scaled` | Result layout -> operand layouts -> `convert_layout` -> target-specific matrix op | Always. This is not a direct rename target. |
| Implicit shared-memory staging | `allocate_shared_memory` after the first correct candidate | Only after the blocked-layout or matrix path is already correct. |
| Descriptor helper usage | Target-specific descriptor family | Only when the target family really supports or benefits from it. |

## Planner Traits and Candidate Slots

GEAK should treat Gluon as a candidate strategy inside the Triton route, not as
a separate kernel type. Planning should be trait-based rather than one recipe
per kernel family.

### Core planning traits

- **Semantics contract**: preserve launcher shape, indexing, masks, boundary
  behavior, correctness oracle, and benchmark intent before changing the
  algorithm.
- **Dialect path**:
  - `plain_triton -> amd_gluon`: start with a minimal viability rewrite.
  - `nv_gluon -> amd_gluon`: translate vendor assumptions; do not rename APIs.
  - `amd_gluon -> amd_gluon`: optimize in dialect first.
- **Layout exposure**:
  - recover `BlockedLayout` from `tl.arange`, tile shape, and `num_warps`;
  - use `SliceLayout` or layout conversions for broadcast, masks, or slices;
  - switch to source-first planning when you see `DistributedLinearLayout`,
    `PartitionedSharedLayout`, host `TensorDescriptor`, nested layout trees, or
    reshape / permute / trans unshuffle sequences.
- **Memory lowering**:
  - start with `gl.load` / `gl.store`;
  - move to AMD `buffer_load` / `buffer_store` only when useful;
  - delay shared memory, swizzles, descriptors, `tdm`, and async paths until a
    simpler candidate is correct.
- **Matrix lowering**:
  - `tl.dot` and `tl.dot_scaled` require result layout, operand layouts,
    `convert_layout`, and a target op;
  - scaled paths require dtype, scale-layout, target-arch, and scale-factor
    checks;
  - gfx1250 WMMA / descriptor paths are separate from CDNA3 / CDNA4 MFMA paths.
- **Execution contract**:
  - check JIT versus AOT;
  - check Triton minor version and `instr_shape`;
  - check target backend and operator-local arch guards;
  - do not equate global Gluon availability with per-operator support.

Target backend resolution happens before task planning where possible:

1. explicit `target_backend`
2. `GEAK_TARGET_BACKEND`
3. `rocminfo` detection without `sudo`
4. default `hip/gfx942`

Set `GEAK_TARGET_BACKEND` in restricted environments. Docker or ROCm shells may
allow GEAK to read `rocminfo` output such as `Name: gfx942` before profiling
artifacts exist.

### Candidate slot policy

Early task generation should prefer:

1. a semantics-preserving AMD Gluon viability candidate;
2. a trait-specific AMD Gluon candidate that names the traits it addresses;
3. a plain Triton fallback or competitor when allowed.

Only consider advanced descriptors, async copy, scheduler hints, persistent
kernels, atomics, or work stealing after a simpler Gluon candidate passes
correctness. If plain Triton wins the benchmark, that is a valid selected
result rather than a failed Gluon run.

### Base / Shared / Extension search space

For GEAK planning, AMD Gluon should be an additive extension to the existing
Triton search.

- **Base Set**: preserve plain Triton candidates from the main planner, such as
  algorithmic rewrites, fusion, shape-specialized variants, memory/layout
  cleanup, and low-priority launch or autotune work.
- **Shared Set**: when a strategy can apply to both dialects, identify whether
  the task is a `plain_triton variant`, an `amd_gluon variant`, or a paired
  comparison under the same benchmark.
- **Extension Set**: add AMD Gluon-only candidates such as minimal viability,
  trait-specific lowering, `nv_gluon -> amd_gluon` translation, in-dialect AMD
  Gluon optimization, MFMA / WMMA / scaled, or descriptor paths.

Do not let Extension Set tasks replace all Base Set tasks. If Gluon compile or
correctness fails, shrink the next Gluon attempt to layout-only,
translation-only, or memory-only work. If Gluon is correct but slower, refine
memory or matrix lowering before trying scheduler, persistent, async, or
descriptor-heavy tasks.

## Plain Triton to AMD Gluon Rewrite Order

For most rewrites, the safest order is:

1. preserve launcher shape, indexing, masks, and correctness behavior
2. reconstruct the base tile and choose a correct blocked layout
3. align host launch config and layout
4. decide the memory path
5. decide the matrix path if needed
6. add shared-memory, async, or tuning features only after the first candidate works

### Host-side layout construction matters

Many first Gluon rewrites fail because they keep the launcher but forget that layout may depend on launch choices.

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
    x = gl.amd.cdna3.buffer_load(ptr=x_ptr, offsets=offsets, mask=mask, other=0.0)
    gl.amd.cdna3.buffer_store(ptr=y_ptr, offsets=offsets, stored_value=x, mask=mask)

def launch(x, y, XBLOCK=256, num_warps=4):
    layout = make_cdna_1d_layout(XBLOCK, num_warps)
    grid = (triton.cdiv(x.numel(), XBLOCK),)
    kernel[grid](x, y, x.numel(), XBLOCK=XBLOCK, layout=layout, num_warps=num_warps)
```

The key point is not the specific vector-copy operation. It is that layout, `XBLOCK`, and `num_warps` are one contract.

## Version and Execution-Mode Traps

### Triton 3.5 versus 3.6+

One of the most important practical boundaries is:

- before Triton `3.6.0`, some AMD MFMA layouts use 2D `instr_shape` such as `[M, N]`
- from Triton `3.6.0` onward, many paths expect 3D `instr_shape` such as `[M, N, K]`

This is not cosmetic. It changes whether layout construction and compilation succeed.

### Treat JIT and AOT as part of the contract

Before editing a Gluon kernel, confirm:

- whether `triton.experimental.gluon` is available in the runtime
- whether the codebase expects JIT, AOT, or both
- whether there are prebuilt-kernel loading paths behind environment-variable gates
- whether the AOT path has scratch-related restrictions

## Optimization Paths by Kernel Family

The best optimization order depends on the kernel family.

### Elementwise or vector-style kernels

Recommended path:

1. preserve launcher and baseline behavior
2. reconstruct a correct blocked layout
3. decide whether generic `gl.load` / `gl.store` is enough
4. move to AMD `buffer_load` / `buffer_store` only if the target family or existing code path benefits
5. benchmark against plain Triton before adding any more structure

### Attention or decode kernels

Recommended path:

1. preserve stride-rich host arguments and partition logic
2. preserve query, key, and value logical shapes before touching matrix ops
3. keep nested `SliceLayout` trees and mask conversions semantically intact
4. wire result layout and operand layouts only after the logical shape story is still correct
5. treat scheduler, barrier, or staging hints as second-stage tuning

This family is where shape rewrites, mask layout conversions, and partition logic often become correctness-critical.

### GEMM or FP8 kernels

Recommended path:

1. identify the real matrix family and K width
2. pick the result layout first
3. derive operand layouts and `convert_layout` steps
4. make the epilogue correct: scales, bias, accumulation dtype, store layout
5. only then add shared-memory staging, async features, preshuffle support, or tuned config selection

### Preshuffled GEMM

This deserves separate caution. In practice it often stops being a generic matrix recipe and becomes an operator-specific layout transformation problem.

Watch for:

- `DistributedLinearLayout`
- `reshape` / `permute` / `trans` unshuffle sequences
- assumptions that K is a multiple of the preshuffled block shape

Preserve the transformation structure before tuning loads or matrix ops.

### `gfx1250` WMMA or descriptor kernels

Recommended path:

1. get plain `wmma` or basic descriptor load and store working
2. validate layout family, contiguous-last-dimension requirements, and shared-layout constraints
3. only then add `wmma_scaled`, scale layouts, `tdm`, async paths, or cluster logic

## gfx1250-Specific Constraints

This is the area where general "AMD Gluon" advice is least sufficient.

### WMMA and scaled WMMA

Practical rules from the frontend checks and current source tree:

- prefer getting plain `wmma` working before adding scaled WMMA
- `wmma_scaled` expects stricter layout contracts than plain `wmma`
- when an input format is `e2m1`, the operand WMMA layout expects `instr_shape=[16, 16, 64]`
- the accumulator layout for scaled WMMA expects `instr_shape=[16, 16, 128]`
- `get_wmma_scale_layout` only supports scale factor `16` or `32`
- scale dtype combinations are limited; do not guess them by name

### Descriptor constraints

Current frontend checks enforce rules such as:

- descriptor rank must be between 1 and 5
- the last tensor dimension must be contiguous
- only `PaddedSharedLayout`, `SwizzledSharedLayout`, or `PartitionedSharedLayout` are accepted descriptor layouts
- for currently accepted swizzled cases, `max_phase` must stay at `1`
- only `"zero"` padding is supported

Treat descriptor setup as a correctness contract, not as a later optimization detail.

## When to Read Source First

Some patterns are strong signals that the general playbook is no longer enough on its own.

Read operator-local source before rewriting the kernel if you see:

- `DistributedLinearLayout`
- `PartitionedSharedLayout`
- host `TensorDescriptor`
- `reshape` / `permute` / `trans` used to unshuffle matrix tiles
- 3D or 5D logical layouts with multiple nested `SliceLayout`
- environment-variable gates around JIT versus AOT
- prebuilt-kernel loading, AOT packaging, or generated asset lookup
- scheduler, barrier, or priority hints that may affect correctness or launch shape

## Common Mistakes

- Treating Gluon as a top-level kernel type rather than a Triton-family extension
- Starting from copied tutorials without checking the target family
- Treating compile success as proof that the rewrite is benchmark-valid
- Renaming NVIDIA APIs to guessed AMD names
- Assuming `gl.amd.cdna3.*` means the kernel cannot target `gfx950`
- Assuming one global "feature available" helper is the operator support matrix
- Mixing environment changes, harness changes, and kernel changes in one unexplained iteration

## Related Guides

### In this knowledge base

- [Triton on AMD GPUs](triton-on-rocm.md)
- [Custom Kernels with Triton](../../../layer-5-llm/05-advanced/custom-kernels/triton-kernels.md)
- [CDNA Architecture and MI Series Guide](../../../layer-1-hardware/amd-gpu-arch/cdna-architecture.md)
- [RDNA Architecture Guide](../../../layer-1-hardware/amd-gpu-arch/rdna-architecture.md)
- [Kernel Optimization for AMD GPUs](../../../best-practices/performance/kernel-optimization.md)
- [GPU Performance Optimization Best Practices](../../../best-practices/performance/gpu-optimization.md)

### GEAK-specific workflow docs

- [GEAK Triton-Gluon workflow guide](../../../../docs/triton_gluon.md)

For GEAK agents, use the workflow guide as a single indexed reference. Search
for `## Quick section map for agents`, then jump to stable headings such as
`### Trait: matrix_dot`, `### Trait: memory_amd_buffer`, or
`### Trait: execution_jit_aot_sensitive`. Do not read the whole guide before
checking the relevant trait sections.

### External resources

- Triton public Gluon tutorials: [https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)
- Triton language reference: [https://triton-lang.org/main/python-api/triton.language.html](https://triton-lang.org/main/python-api/triton.language.html)
- ROCm blog on developing Triton kernels on AMD GPUs: [https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
