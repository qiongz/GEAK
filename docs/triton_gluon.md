# Triton-Gluon

This is the single GEAK agent-facing document for the Triton-family Gluon
feature. It is primarily read by planner and worker agents through
`gluon_guide_path`, not by end users during normal GEAK invocation.

The goal is not just to explain the API surface. The goal is to help GEAK
correctly read, translate, generate, and optimize:

- `plain_triton`
- `nv_gluon`
- `amd_gluon`

while staying grounded in what current Triton-family and AMD-facing Gluon
implementations actually support today.

## Repository wiring

Latest `main` keeps structured optimization knowledge under the RAG MCP tree:

```text
mcp_tools/rag-mcp/knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-gluon-on-rocm.md
```

GEAK still injects this document explicitly into Triton-Gluon task prompts as
`gluon_kb_path`. The RAG `query` / `optimize` tools may also retrieve it when
RAG is enabled, but planner prompts should not depend on RAG alone. For
Triton-Gluon tasks, the deterministic context files are:

- `docs/triton_gluon.md` as `gluon_guide_path`;
- the RAG MCP AMD KB entry above as `gluon_kb_path`;
- `examples/triton_gluon_inputs/README.md` as `gluon_examples_path`.

## Quick section map for agents

Do not read this whole document first. Use the detected planning traits to jump
to the short trait reference below, then view only the headings it names.

- `semantics_contract` -> `### Trait: semantics_contract`
- `dialect_plain_triton` -> `### Trait: dialect_plain_triton`
- `dialect_nv_gluon` -> `### Trait: dialect_nv_gluon`
- `dialect_amd_gluon` -> `### Trait: dialect_amd_gluon`
- `layout_basic` -> `### Trait: layout_basic`
- `layout_slice_broadcast` -> `### Trait: layout_slice_broadcast`
- `layout_source_first_required` -> `### Trait: layout_source_first_required`
- `memory_generic` -> `### Trait: memory_generic`
- `memory_amd_buffer` -> `### Trait: memory_amd_buffer`
- `memory_shared_async_descriptor` -> `### Trait: memory_shared_async_descriptor`
- `matrix_none` -> `### Trait: matrix_none`
- `matrix_dot` -> `### Trait: matrix_dot`
- `matrix_scaled_dot` -> `### Trait: matrix_scaled_dot`
- `matrix_wmma_descriptor` -> `### Trait: matrix_wmma_descriptor`
- `execution_jit_aot_sensitive` -> `### Trait: execution_jit_aot_sensitive`
- `version_sensitive` -> `### Trait: version_sensitive`
- `operator_support_sensitive` -> `### Trait: operator_support_sensitive`
- `shape_coverage_unknown` -> `### Trait: shape_coverage_unknown`
- `shape_coverage_single` -> `### Trait: shape_coverage_single`
- `shape_coverage_multi` -> `### Trait: shape_coverage_multi`
- `shape_coverage_bucketed` -> `### Trait: shape_coverage_bucketed`
- `shape_layout_constexpr_risk` -> `### Trait: shape_layout_constexpr_risk`
- `shape_dispatch_required` -> `### Trait: shape_dispatch_required`
- `search_space_allocation` -> `### Search space: base_shared_extension`

## Trait reference for targeted reading

### Trait: semantics_contract

Read this first: preserve launcher shape, indexing, masks and boundaries,
correctness behavior, and benchmark intent before changing algorithms.

Then view these headings:

- `## 1. Product contract`
- `## 3. How GEAK should apply this feature`
- `### 8.2 Host and runtime contract`
- `## 11. Benchmark-aware rules`

Do not:

- claim success from compile-only validation;
- change the evaluation contract or harness behavior.

### Trait: dialect_plain_triton

Read this first: start from a minimal AMD Gluon viability rewrite with explicit
layout while keeping plain Triton as a benchmarked competitor.

Then view these headings:

- `### 4.2 `plain_triton` input`
- `### 9.1 Example: `plain_triton -> amd_gluon` row-wise affine transform`
- `### A.2 JIT entry and host launcher`

Do not:

- replace the whole algorithm before a correctness-passing baseline candidate;
- assume AMD Gluon must beat plain Triton.

### Trait: dialect_nv_gluon

Read this first: treat NVIDIA-facing Gluon as translation, not API renaming.
Preserve common Gluon semantics before moving vendor-specific paths to AMD.

Then view these headings:

- `### 4.3 `nv_gluon` input`
- `### 6.2 NVIDIA families`
- `### 6.6 Concept map`
- `### 9.2 Example: `nv_gluon -> amd_gluon` tile reader rewrite`
- `### A.7 NVIDIA quick patterns`

Do not:

- output an optimized `nv_gluon` path;
- rename TMA, WGMMA, tensor-memory, or cluster APIs into guessed AMD names.

### Trait: dialect_amd_gluon

Read this first: preserve the existing AMD-facing structure and optimize inside
AMD Gluon unless benchmark evidence favors a plain Triton fallback.

Then view these headings:

- `### 4.4 `amd_gluon` input`
- `### 6.3 AMD families`
- `### 7.1 Attention path on `gfx942` / `gfx950``
- `### 7.2 GEMM and FP8 paths on `gfx950``

Do not:

- discard operator-local architecture guards;
- drift back to plain Triton without benchmark evidence.

### Trait: layout_basic

Read this first: recover `BlockedLayout` from `tl.arange`, tile shape,
`num_warps`, target family, and coalesced dimension.

Then view these headings:

- `#### Recover implicit layout before changing APIs`
- `#### Align host launcher and layout`
- `### 5.1 Layout is first-class`
- `### A.3 Core language and layout surface`

Do not:

- construct layout independently from launch attributes;
- omit `layout` on `gl.arange`, `gl.zeros`, or similar distributed tensors.

### Trait: layout_slice_broadcast

Read this first: masks, broadcasts, `expand_dims`, and slicing often need
compatible `SliceLayout` or explicit layout conversions.

Then view these headings:

- `### 5.1 Layout is first-class`
- `#### Attention or decode kernels`
- `### A.3 Core language and layout surface`

Do not:

- treat mask layout conversions as cosmetic cleanup;
- apply `[:, None]` or `expand_dims` on arbitrary incompatible layouts.

### Trait: layout_source_first_required

Read this first: source-first planning is required for distributed layouts,
descriptors, nested layout trees, JIT/AOT packaging, and unshuffle sequences.

Then view these headings:

- `### 7.7 When docs are not enough`
- `#### Preshuffled GEMM`
- `#### `gfx1250` WMMA or descriptor kernels`
- `### 8.2 Host and runtime contract`

Do not:

- simplify `reshape` / `permute` / `trans` unshuffle sequences blindly;
- rewrite descriptor or prebuilt-kernel paths without reading operator source.

### Trait: memory_generic

Read this first: start with `gl.load` / `gl.store` for scalar or simple vector
paths before moving to AMD-specific memory operations.

Then view these headings:

- `#### Choose the memory path deliberately`
- `### 6.1 Common Gluon layer`
- `### A.3 Core language and layout surface`

Do not:

- introduce `buffer_load` / `buffer_store` only because the target is AMD;
- add shared-memory staging in the initial layout rewrite unless source requires it.

### Trait: memory_amd_buffer

Read this first: use AMD `buffer_load` / `buffer_store` when target family,
existing AMD structure, or access pattern actually benefits.

Then view these headings:

- `#### Choose the memory path deliberately`
- `#### `gfx942` / CDNA3`
- `#### `gfx950` / CDNA4`
- `### A.6 AMD quick patterns`

Do not:

- infer full architecture support from the Python namespace alone;
- mix CDNA memory assumptions with gfx1250 descriptor paths.

### Trait: memory_shared_async_descriptor

Read this first: shared memory, swizzles, async copy, descriptors, and `tdm`
are second-stage tools after a simpler candidate is correct.

Then view these headings:

- `### 5.2 Synchronization and pipeline concepts`
- `### 5.3 Descriptor and tensor-memory concepts`
- `### A.4 Shared memory, synchronization, and cluster surface`
- `### A.5 Descriptor and tensor-memory surface`

Do not:

- start with descriptor, async, scheduler, or persistent work as the first candidate;
- translate NVIDIA TMA concepts to AMD by name.

### Trait: matrix_none

Read this first: if there is no real matrix instruction path, stop at explicit
layout and memory lowering rather than forcing MFMA or WMMA.

Then view these headings:

- `#### Treat matrix lowering as a separate step`
- `#### Elementwise or vector-style kernels`
- `## 11. Benchmark-aware rules`

Do not:

- add MFMA or WMMA when the source has no matrix trait.

### Trait: matrix_dot

Read this first: `tl.dot` lowers through result layout, operand layouts,
`convert_layout`, and target matrix op.

Then view these headings:

- `#### Treat matrix lowering as a separate step`
- `### 6.3 AMD families`
- `#### CDNA3 MFMA pattern`
- `#### `gfx1250` WMMA pattern`

Do not:

- textually replace `tl.dot` with `mfma` or `wmma`;
- skip accumulator and operand layout compatibility.

### Trait: matrix_scaled_dot

Read this first: scaled matrix paths require dtype, scale layout, target arch,
instruction shape, and scale factor checks.

Then view these headings:

- `#### GEMM or FP8 kernels`
- `#### CDNA4 scaled-MFMA pattern`
- `#### `gfx1250` WMMA pattern`
- `### A.10 Common failures and fix order`

Do not:

- use `mfma_scaled` on targets or dtype combinations that do not support it;
- guess scale formats from names alone.

### Trait: matrix_wmma_descriptor

Read this first: gfx1250 WMMA, descriptor, `tdm`, cluster, and shared-layout
rules are separate from CDNA MFMA behavior.

Then view these headings:

- `#### `gfx1250` and RDNA-style paths`
- `#### `gfx1250` WMMA or descriptor kernels`
- `#### `gfx1250` descriptor constraints`
- `### A.5 Descriptor and tensor-memory surface`

Do not:

- treat gfx1250 as CDNA with renamed APIs;
- add `wmma_scaled`, `tdm`, or cluster behavior before plain WMMA or descriptor works.

### Trait: execution_jit_aot_sensitive

Read this first: JIT availability, AOT packaging, prebuilt kernels, and scratch
constraints are part of the integration contract.

Then view these headings:

- `### 2.2 JIT and AOT are both real`
- `### 7.3 JIT and AOT can coexist in one operator family`
- `### 9.4 Example: JIT with explicit AOT fallback`
- `### A.8 Version and compatibility checklist`

Do not:

- delete existing fallback gates without understanding runtime packaging.

### Trait: version_sensitive

Read this first: Triton minor version can change layout construction, especially
`AMDMFMALayout.instr_shape`.

Then view these headings:

- `### 2.3 Triton version compatibility matters`
- `### 9.3 Example: Triton-version-compatible MFMA layout guard`
- `### A.8 Version and compatibility checklist`
- `### A.10 Common failures and fix order`

Do not:

- assume a 2D or 3D `instr_shape` without checking the expected Triton version.

### Trait: operator_support_sensitive

Read this first: global "Gluon available" checks are not the same as an
operator-local support matrix.

Then view these headings:

- `### 6.4 Practical differences by target`
- `### 6.5 Module path vs architecture version is not always the same thing`
- `### 7.5 Feature availability is not operator support`
- `### A.8 Version and compatibility checklist`

Do not:

- infer support from a global helper or namespace name alone.

### Trait: shape_coverage_unknown

Read this first: the benchmark harness did not expose any shape-count signal
(no `build/performance_report.json`, no `(M,K,N): X ms` lines, no `N shapes`
report). Treat the kernel as if it could be any of single / multi / bucketed
until proven otherwise; do not optimize for a specific tile size.

Then view these headings:

- `## 11. Benchmark-aware rules`
- `### Trait: semantics_contract`

Do not:

- guess block sizes from a single example call;
- emit a Gluon candidate that hardcodes a launch shape before running
  correctness on the actual harness stream.

### Trait: shape_coverage_single

Read this first: the harness exposes exactly one representative shape. A
single-shape `Gluon-positive` is allowed but never sufficient by itself;
treat any speedup as fragile until a multi-shape harness confirms it.

Then view these headings:

- `### Trait: semantics_contract`
- `## 11. Benchmark-aware rules`

Do not:

- claim a portable speedup from one shape;
- branch the kernel on `M == X` constants when the harness is single-shape;
  this is exactly the pattern that becomes a hidden regression once the
  harness is upgraded to multi-shape.

### Trait: shape_coverage_multi

Read this first: correctness and performance share an ordered case stream of
multiple shapes. Every candidate task must classify itself as
`single_shape_viability`, `shape_robust`, or `shape_bucketed` (SYSTEM_PROMPT
rule 17). Per-shape correctness must all pass; per-shape speedup is reported
back to the planner and any shape with ratio `< 0.9` becomes a forced
coverage requirement on the next round.

Then view these headings:

- `### Trait: layout_basic`
- `### Trait: semantics_contract`
- `### Search space: base_shared_extension`

Do not:

- hardcode shape literals to win one case at the cost of another;
- collapse the case stream to a single representative shape;
- skip Base Set plain Triton candidates - planner allocates at least one
  Base Set + one Shared Set slot when `shape_coverage_profile == multi` and
  the run has `>= 4` GPU.

### Trait: shape_coverage_bucketed

Read this first: shapes split into qualitatively different buckets (small
vs large M, contiguous vs strided K, etc.) and a single launch config almost
never wins all buckets. Prefer host-side dispatch (`if M < threshold:
kernel_v1[grid](...)` else `kernel_v2[grid](...)`) over heuristic
`@triton.heuristics` predicates; the planner will pair `shape_dispatch_required`
strong trait with this profile, so Extension Set has at least 2 slots when
`>= 5` GPU.

Then view these headings:

- `### Trait: shape_dispatch_required`
- `### Trait: shape_layout_constexpr_risk`
- `### Trait: layout_basic`
- `### Search space: base_shared_extension`

Do not:

- collapse buckets to one tile config and then claim a global speedup;
- modify `@triton.heuristics({...})` lambdas to silently change which
  bucket falls into which config - audit treats this as `config-shifted`
  and rejects the patch even if aggregate speedup looks positive;
- omit Base Set plain Triton dispatch fallback - bucketed Gluon must beat
  bucketed plain Triton, not single-config plain Triton.

### Trait: shape_layout_constexpr_risk

Read this first: Gluon layouts (`BlockedLayout`, `AMDMFMALayout.instr_shape`,
`SliceLayout`) are passed as `constexpr`; if the layout is derived from a
hardcoded shape it baked-in rather than passed from host, multi-shape
correctness breaks the moment a different shape is dispatched. Construct
layouts on the host from launch attributes and pass them as `constexpr`
arguments.

Then view these headings:

- `### Trait: layout_basic`
- `#### Recover implicit layout before changing APIs`
- `#### Align host launcher and layout`

Do not:

- instantiate `AMDMFMALayout(version=3, instr_shape=[16,16])` inside the
  kernel for a multi-shape harness without a host-side dispatch wrapper;
- hardcode `tl.arange` upper bounds based on a single-shape assumption.

### Trait: shape_dispatch_required

Read this first: this is a strong Gluon-extension signal. A single in-kernel
config cannot serve all shape buckets, so the patch must add a host-side
selector that picks block size, num_warps, layout, or even kernel variant
based on launch attributes. The planner reserves an Extension Set slot for
this trait when the GPU budget allows.

Then view these headings:

- `### Trait: shape_coverage_bucketed`
- `### Trait: shape_layout_constexpr_risk`
- `### Search space: base_shared_extension`

Do not:

- substitute autotune for explicit dispatch when the harness already
  enumerates the cases; autotune adds compile-time overhead and the planner
  cannot verify per-shape coverage from autotune logs;
- write a single `@triton.heuristics` predicate that switches block size on
  shape - `compare.py` audit treats heuristic mutation as config-shift even
  when speedup looks positive.

## 1. Product contract

GEAK treats Gluon as a **feature extension of Triton**, not as a new top-level
kernel type.

- keep `kernel_type = triton`
- infer `input_dialect` internally to classify the input:
  - `plain_triton`
  - `nv_gluon`
  - `amd_gluon`
- if discovery, task frontmatter, or config metadata mentions a Gluon source,
  keep the top-level route as `kernel_type = triton`; represent Gluon-specific
  state through `input_dialect`, output dialect policy, and planner search
  guidance inside the Triton-family path
- default Triton runs use `gluon_feature_mode = auto`, so AMD Gluon is part of
  the candidate search space when the planner receives Triton-family metadata
- use `gluon_feature_mode` only as an explicit control for ablation or forced
  debugging:
  - `off`
  - `auto`
  - `force`
- prefer `amd_gluon` output when it is allowed, while keeping plain Triton as a
  benchmarked fallback

Valid optimized outputs:

- `plain_triton`
- `amd_gluon`

GEAK should never create a new optimized `nv_gluon` output path.

Separate translation paths, such as latest-main `kernel_type = pytorch2flydsl`
or detected `kernel_type = flydsl`, are independent of this contract. They may
trigger FlyDSL translation / homogeneous routing, but they should not change how
Triton or Triton-Gluon kernels are classified.

## 2. Runtime, version, and compilation model

### 2.1 What Gluon is in Triton

In Triton itself, Gluon is not just syntax sugar. It has its own JIT source
path and language mode.

- `triton.experimental.gluon.jit` produces a `GluonJITFunction`
- `GluonJITFunction` uses `GluonASTSource`
- `GluonASTSource` sets the language to `Language.GLUON`
- the generated module is marked with launch-time attributes such as:
  - `ttg.target`
  - `ttg.num-warps`
  - `ttg.num-ctas`
  - `ttg.threads-per-warp`

That means host launch shape is part of the correctness contract, not a
secondary afterthought.

### 2.2 JIT and AOT are both real

Real code in aiter uses both:

- direct `@gluon.jit` kernels launched from Python
- AOT-compiled Gluon kernels wrapped from C++ / PyTorch integration code

Do not assume every Gluon path is "just JIT". In practice:

- one operator may use Gluon JIT
- another may use Gluon AOT
- a mixed pipeline may use a Gluon main kernel and a normal Triton reduce kernel

GEAK should therefore think in terms of **kernel-path compatibility**, not just
"can I write the kernel body".

### 2.3 Triton version compatibility matters

The real code under aiter explicitly carries version branches.

High-value compatibility facts:

- Gluon is expected only on newer Triton builds in practice
- Triton `3.5` and `3.6+` differ in `AMDMFMALayout.instr_shape`
- some AOT kernels built against Triton `3.5` may not load under Triton `3.6`
  because metadata requirements changed

Practical implication for GEAK:

- when reading or generating code that uses `AMDMFMALayout`, check whether the
  codebase expects a 2D or 3D `instr_shape`
- when interacting with prebuilt kernels or AOT assets, treat Triton minor
  version as part of the benchmark and integration contract

## 3. How GEAK should apply this feature

When the Gluon skill or this guide is in context, GEAK should **act**, not stop
at explanation.

1. Classify the input as `plain_triton`, `nv_gluon`, or `amd_gluon`.
2. Preserve source semantics first:
   - launcher shape
   - indexing
   - masks and boundaries
   - correctness behavior
   - benchmark intent
3. Choose the path:
   - `plain_triton`: decide whether generating an `amd_gluon` candidate is
     structurally promising
   - `nv_gluon`: translate vendor-specific APIs, layouts, descriptors, and
     memory paths into AMD-facing Gluon first, then optimize
   - `amd_gluon`: keep optimizing inside AMD Gluon space unless benchmark
     evidence strongly favors a fallback
4. Compare the result against the source baseline or an allowed fallback path.

## 4. Writing model and input paths

### 4.1 General writing model

Gluon keeps Triton's host-side launcher model:

- `@gluon.jit`
- `kernel[grid](...)`
- `triton.cdiv`
- `program_id`
- `constexpr`

What changes is that Gluon makes low-level decisions explicit:

- layout selection
- shared-memory usage
- synchronization
- descriptor and tensor-memory usage
- target-specific memory paths
- target-specific matrix instruction paths

Start from **target family + layout**, not from copied tutorial syntax.

### 4.2 `plain_triton` input

Use this path when the source is normal Triton and you want to decide whether an
AMD-facing Gluon candidate is worth generating.

Recommended mechanical rewrite order:

1. keep launcher and correctness semantics recognizable
2. reconstruct the base tile and layout before touching target-specific ops
3. align host launch config and layout
4. choose the memory path
5. choose the matrix path if needed
6. add shared-memory or pipeline features only after the first candidate works

#### Recover implicit layout before changing APIs

In plain Triton, `tl.arange`, tile sizes, pointer arithmetic, and `num_warps`
implicitly define how work is distributed. In Gluon, that distribution becomes
an explicit layout object.

- start from the logical tile shape from the original Triton kernel
- derive a `BlockedLayout` that matches the original execution shape
- for 1D kernels, satisfy:
  `size_per_thread[0] * threads_per_warp[0] * warps_per_cta[0] == XBLOCK`
- on MI3xx-style CDNA targets, prefer a wave64-valid first candidate such as
  `threads_per_warp=[64]`
- for 2D kernels, choose `order` so the fastest-varying logical dimension still
  matches the coalesced dimension from the original pointer arithmetic
- derive `SliceLayout` or `DotOperandLayout` from that base layout instead of
  inventing them independently

#### Align host launcher and layout

Gluon keeps Triton's host-side launcher model, but host code may now need to
construct and pass layouts.

- if layout depends only on fixed compile-time constants, it can live inside the
  kernel
- if layout depends on `BLOCK_*`, `num_warps`, target family, or a tuning
  choice, build it on the host and pass it as a `constexpr`
- keep the launcher in `kernel[grid](...)` form
- keep `num_warps`, `num_ctas`, and target architecture aligned with the layout,
  because Gluon IR verification checks layouts against launch attributes

#### Choose the memory path deliberately

Use the simplest correct path first:

- `gl.load` / `gl.store` is the right first rewrite for scalar or simple vector
  loads and stores
- move to AMD `buffer_load` / `buffer_store` when the target family, existing
  AMD Gluon code structure, or performance-sensitive access pattern actually
  calls for it
- do not introduce descriptor paths, async copy, or shared-memory staging in the
  same step as the initial layout rewrite unless the source already depends on
  them

#### Treat matrix lowering as a separate step

`tl.dot` and `tl.dot_scaled` are not direct rename targets.

The usual AMD Gluon path is:

1. choose the result layout: `AMDMFMALayout` or `AMDWMMALayout`
2. derive `DotOperandLayout` for each operand
3. `convert_layout` into the operand layouts
4. call the target-specific op such as `mfma`, `mfma_scaled`, or `wmma`

Only after that first matrix candidate is correct should you add:

- shared-memory swizzles
- async copy
- scale-layout helpers
- scheduler or barrier hints

Stay in plain Triton when:

- the kernel is simple and already expresses the right execution shape
- explicit layouts add complexity without a plausible performance upside
- the target-specific path would only be compile-valid, not benchmark-valid

### 4.3 `nv_gluon` input

Treat this as a translation problem, not a rename problem.

Preserve first:

- common Gluon control flow
- launcher shape
- indexing semantics
- masks
- correctness scaffolding

Re-evaluate before carrying to AMD:

- wave32-centric layouts
- NVIDIA async-copy paths
- `tma` / descriptor assumptions
- Hopper or Blackwell-specific matrix instructions
- tensor-memory and cluster control features

The goal is not to keep a better `nv_gluon` output. The goal is to translate
into a valid `amd_gluon` candidate and then optimize it.

### 4.4 `amd_gluon` input

This is the most direct optimization path.

- keep the search inside AMD Gluon space first
- preserve the AMD-facing structure unless benchmark evidence clearly says a
  fallback should win
- prefer wave64-valid decomposition on MI3xx-class targets
- keep architecture-specific capabilities aligned with the actual target family

## 5. Layout, synchronization, and descriptor mental model

### 5.1 Layout is first-class

Real Triton Gluon code uses more than just `BlockedLayout`.

Common high-value layout concepts:

| Layout / helper | Main purpose |
|-----------------|--------------|
| `BlockedLayout` | Base thread/warp/CTA distribution |
| `SliceLayout` | Select a sub-dimension from a parent layout |
| `DotOperandLayout` | Operand layout for matrix instructions |
| `DistributedLinearLayout` | Explicit thread/register mapping when the default blocked view is not enough |
| `SwizzledSharedLayout` | Shared-memory swizzle for conflict-aware access |
| `PaddedSharedLayout` | Shared-memory layout with padding semantics |
| `PartitionedSharedLayout` | gfx1250 / RDNA-style partitioned shared-memory path |
| `AMDMFMALayout` | AMD MFMA result layout |
| `AMDWMMALayout` | AMD WMMA result layout |
| `TensorMemoryLayout` | Blackwell tensor-memory layout |

Recommended workflow:

1. choose the target family
2. choose the base layout
3. derive slice, dot-operand, shared, or descriptor layouts
4. build indices with that layout
5. lower memory or instruction paths

### 5.2 Synchronization and pipeline concepts

Useful concepts across the codebase:

- `allocate_shared_memory`
- `barrier`
- `mbarrier`
- `cluster`
- `fence_async_shared`
- `warp_pipeline_stage`

Do not assume these are interchangeable across vendors.

- Hopper and Blackwell expose one cluster/barrier family
- gfx1250 exposes another
- AMD also has optional wave or warp pipeline guidance

### 5.3 Descriptor and tensor-memory concepts

This is one of the easiest places to make wrong assumptions.

NVIDIA-side concepts:

- host `TensorDescriptor`
- device-side `tma`
- Blackwell tensor memory
- `TensorMemoryLayout`
- `tcgen05_*`

AMD-side concepts:

- gfx1250 host `TensorDescriptor`
- `tdm`
- `wmma`
- `PartitionedSharedLayout`

These are **not** 1:1 renames. Treat them as separate capability families.

## 6. NVIDIA, AMD, and architecture differences

### 6.1 Common Gluon layer

Reusable syntax usually comes from `triton.experimental.gluon.language`:

- `program_id`
- `constexpr`
- `BlockedLayout`
- `SliceLayout`
- `DotOperandLayout`
- `arange`
- `load`
- `store`
- `zeros`
- `convert_layout`
- `allocate_shared_memory`
- `barrier`

This layer gives you the kernel skeleton. Vendor-specific modules provide the
performance-critical memory or instruction paths.

### 6.2 NVIDIA families

| Family | Typical focus |
|--------|----------------|
| Ampere | `async_copy`, `mma_v2` |
| Hopper | `tma`, `mbarrier`, `cluster`, `warpgroup_mma` |
| Blackwell | `tensor_memory_descriptor`, `TensorMemoryLayout`, `tcgen05_*`, `clc`, richer TMA patterns |

Important practical notes:

- Hopper is the first place where `warpgroup_mma` and richer `tma` flows become
  central
- Blackwell adds tensor-memory and `tcgen05_*` concepts that are not just
  "Hopper but larger"
- some NVIDIA APIs are re-exported through later-generation modules, so always
  verify the actual import path in your local Triton build

NVIDIA-oriented Gluon code often assumes wave32-style execution and newer
descriptor or warpgroup features. Those assumptions must be rechecked before
moving to AMD.

### 6.3 AMD families

| Family | Typical focus |
|--------|----------------|
| CDNA3 / `gfx942` | `buffer_load`, `buffer_store`, `mfma`, `AMDMFMALayout` |
| CDNA4 / `gfx950` | CDNA3 ops plus `async_copy`, `mfma_scaled`, `get_mfma_scale_layout` |
| RDNA3 / RDNA4 | `wmma` |
| `gfx1250` | `wmma`, `wmma_scaled`, `tdm`, `async_copy`, `mbarrier`, `cluster`, `AMDWMMALayout` |

Real layout-version mapping in Triton matters:

- `AMDMFMALayout(version=1)` -> `gfx908`
- `AMDMFMALayout(version=2)` -> `gfx90a`
- `AMDMFMALayout(version=3)` -> `gfx942`
- `AMDMFMALayout(version=4)` -> `gfx950`

and:

- `AMDWMMALayout(version=1)` -> RDNA3
- `AMDWMMALayout(version=2)` -> RDNA4
- `AMDWMMALayout(version=3)` -> `gfx1250`

### 6.4 Practical differences by target

#### `gfx942` / CDNA3

Best first target for GEAK's current feature work.

- prefer explicit wave64-valid layouts
- prefer `buffer_load` / `buffer_store`
- use `mfma` only when the kernel is actually matrix-op based
- do not force descriptor or async-copy parity just because the source used it

#### `gfx950` / CDNA4

Use when the kernel truly benefits from CDNA4-only surfaces.

- can expose newer async-copy and scaled-MFMA paths
- `mfma_scaled` and `get_mfma_scale_layout` are specific high-value additions
- should not be treated as mandatory for every AMD Gluon optimization

#### `gfx1250` and RDNA-style paths

Treat these as separate families rather than as "CDNA with different names".

- `wmma` mental model differs from `mfma`
- `tdm` is not a direct rename of NVIDIA `tma`
- cluster and barrier APIs must be re-evaluated on their own terms
- current aiter Gluon code does not represent gfx1250 as a drop-in extension of
  the current CDNA3/4 kernels
- host `TensorDescriptor` paths are stricter than generic CDNA-style blocked
  layouts: the last dimension must be contiguous, only certain shared-memory
  layout families are valid, and some swizzle settings are rejected
- `wmma_scaled` carries extra contract checks around `instr_shape`,
  accumulator layout, scale dtype combinations, and scale-factor selection
- practical first path: get plain `wmma` or a basic descriptor path working
  before adding `wmma_scaled`, `tdm`, or cluster behavior

### 6.5 Module path vs architecture version is not always the same thing

One subtle but important practical detail from real aiter code:

- you may see `gl.amd.cdna3.*` memory or instruction namespaces
- while the layout or arch branch still passes `AMDMFMALayout(version=4)` for
  `gfx950`

Do not assume the Python namespace name alone tells you the full architecture
contract. Read:

- the actual target arch
- the layout version
- the version guard
- any feature branch around scaled ops or async-copy

### 6.6 Concept map

These are conceptual correspondences, not 1:1 substitutions:

| Concept | NVIDIA | AMD |
|--------|---------|-----|
| Matrix path | `mma_v2`, `warpgroup_mma`, `tcgen05_mma` | `mfma`, `wmma` |
| Async global -> shared | `async_copy_*`, `tma.*` | family-specific `async_copy.*` |
| Descriptor-like path | `tma`, `TensorDescriptor` | `tdm` |
| Barrier / cluster | `mbarrier`, `cluster` | family-specific barrier / cluster APIs |
| Tensor memory | Blackwell tensor memory | no direct global AMD equivalent; use gfx1250-specific `tdm` / WMMA family where appropriate |

## 7. Real patterns from aiter

### 7.1 Attention path on `gfx942` / `gfx950`

The paged-attention Gluon code in aiter is useful because it shows real
production-style composition:

- `BlockedLayout`
- `SliceLayout`
- `AMDMFMALayout(version=CDNA_VERSION, ...)`
- `DotOperandLayout`
- `allocate_shared_memory`
- shared-memory swizzle
- `buffer_load` / `buffer_store`
- explicit stride-rich host arguments

This is more representative than a toy vector-add kernel.

### 7.2 GEMM and FP8 paths on `gfx950`

The GEMM side adds patterns that the current GEAK docs should explicitly cover:

- `mfma_scaled`
- `get_mfma_scale_layout`
- architecture-conditioned K widths and instruction shapes
- JSON-config or heuristic-driven launch selection instead of only online
  autotune

### 7.3 JIT and AOT can coexist in one operator family

aiter uses:

- pure JIT Gluon kernels
- AOT-compiled Gluon kernels
- mixed pipelines where only some stages are Gluon

So GEAK should not force every downstream rewrite into one execution mode.

### 7.4 Optional scheduling hints are real but secondary

aiter also carries optional scheduling or barrier-style hints in some paths.

Treat these as:

- second-stage tuning features
- target-specific hints
- not first-pass mandatory portability features

### 7.5 Feature availability is not operator support

Real downstream trees often have both:

- a coarse "Gluon is available on this machine" helper
- operator-local arch guards inside specific kernels

Do not assume these are the same thing.

Common examples in aiter-style code:

- one global helper may only mark `gfx950` and `gfx1250` as Gluon-available
- a specific attention kernel may still support `gfx942` and `gfx950`
- a CDNA4 GEMM may require `gfx950` even though another Gluon operator runs on
  `gfx942`

When reviewing or generating code, read operator-local guards before concluding
that an architecture is supported or unsupported.

### 7.6 Optimization paths by kernel family

The most effective optimization order depends on the kernel family. Real aiter
code strongly suggests the following progression.

#### Elementwise or vector-style kernels

Recommended path:

1. preserve launcher, masks, and baseline behavior
2. reconstruct a correct blocked layout
3. decide whether generic `gl.load` / `gl.store` is enough
4. move to AMD `buffer_load` / `buffer_store` only if the target family or the
   existing code path actually benefits
5. benchmark against plain Triton before adding any more structure

This is the safest family for early `plain_triton -> amd_gluon` candidates.

#### Attention or decode kernels

Recommended path:

1. preserve stride-rich host arguments and partition logic
2. preserve the query, key, value logical shapes before touching matrix ops
3. keep nested `SliceLayout` trees and mask conversions semantically intact
4. wire `AMDMFMALayout` and operand layouts only after the logical shape story
   is still correct
5. treat optional scheduler, barrier, or staging hints as second-stage tuning

The main trap here is thinking that reshape, mask conversion, or layout changes
on scores are cosmetic. In real attention kernels, they are often part of
correctness.

#### GEMM or FP8 kernels

Recommended path:

1. identify the actual matrix instruction family and K width
2. pick the result layout first
3. derive operand layouts and `convert_layout` steps
4. make the epilogue correct: scales, bias, accumulation dtype, store layout
5. only then add shared-memory staging, async features, preshuffle support, or
   tuned config selection

In real `gfx950` code, config selection may come from heuristics or checked-in
JSON, not only from online autotune.

#### Preshuffled GEMM

This deserves separate caution because it often stops being a generic matrix
recipe and becomes an operator-specific layout transformation problem.

Watch for:

- `DistributedLinearLayout`
- `reshape` / `permute` / `trans` unshuffle sequences
- assumptions that K is a multiple of the preshuffled block shape

When these appear, preserve the transformation structure first and optimize only
after the unshuffle is still correct.

#### `gfx1250` WMMA or descriptor kernels

Recommended path:

1. get plain `wmma` or plain descriptor load and store working
2. validate layout family, contiguous-last-dimension requirements, and shared
   layout constraints
3. only then add `wmma_scaled`, scale layouts, `tdm`, async paths, or cluster
   logic

This family has more frontend assertions than the current CDNA3/4 paths, so it
punishes speculative rewrites more quickly.

### 7.7 When docs are not enough

Even with this guide, some patterns should immediately trigger source-first
reading rather than generic rewriting.

Read operator-local source before changing the kernel if you see:

- `DistributedLinearLayout`
- `PartitionedSharedLayout`
- host `TensorDescriptor`
- `reshape` / `permute` / `trans` used to unshuffle matrix tiles
- 3D or 5D logical layouts with multiple nested `SliceLayout`
- environment-variable gates around JIT versus AOT paths
- prebuilt-kernel loading, AOT packaging, or generated asset lookup
- scheduler, barrier, or priority hints that appear to affect launch shape or
  correctness

These are the cases where the general playbook is still useful, but it is no
longer sufficient on its own.

## 8. Planner traits for candidate generation

GEAK's planner should not hard-code one Gluon recipe per kernel family. It
should infer a small set of composable planning traits, then allocate candidate
task slots using the same `Prefer First` / `Consider Next` / `Deprioritize`
style used by the HIP and plain Triton planner.

These traits are internal planning signals. They are not user-facing input
requirements.

### 8.1 Lowering mental model

The most reusable pattern from Triton's translator helpers is:

1. identify source operation traits,
2. choose the target family,
3. select layout helpers,
4. lower the operation with explicit layout conversion.

For example, AMD dot lowering is not a text substitution. A plain `tl.dot`
usually becomes:

1. target-family dispatch:
   - CDNA3 / CDNA4: `AMDMFMALayout`
   - gfx1250: `AMDWMMALayout`
2. result layout selection
3. `DotOperandLayout` for each operand
4. `convert_layout` into those operand layouts
5. target op such as `mfma`, `wmma`, `mfma_scaled`, or `wmma_scaled`
6. conversion back to the caller's expected layout when needed

Likewise, generic Triton helpers such as `tl.arange`, `tl.full`, and
`expand_dims` become layout-aware operations. The first planner question should
therefore be "what traits does this source operation expose?" rather than "what
kernel family is this?".

### 8.2 Host and runtime contract

Real AMD Gluon code often has a host-side contract as important as the kernel
body:

- wrapper ABI and tensor shape assumptions
- stride-rich arguments and logical tensor rank
- host-created layouts passed as `constexpr`
- `num_warps`, `num_ctas`, block sizes, and target arch alignment
- arch guards such as `gfx942`, `gfx950`, and `gfx1250`
- Triton minor-version guards, especially `AMDMFMALayout.instr_shape`
- JIT versus AOT gates and prebuilt-kernel loading behavior

Planner tasks should not ask an agent to "just rewrite the kernel body" when
these contracts are visible. In those cases the first candidate should be
source-first and semantics-preserving.

GEAK resolves the target backend before the first task-planning pass. The
priority is:

1. explicit `target_backend` from CLI, task text, or discovery metadata,
2. `GEAK_TARGET_BACKEND`,
3. best-effort `rocminfo` detection without `sudo`,
4. default `hip/gfx942`.

Normal non-Docker environments should prefer `GEAK_TARGET_BACKEND` when
`rocminfo` is unavailable or restricted. Docker or ROCm shells can usually
provide early `gfx*` detection before profiling creates `profile.json`.

### 8.3 Core traits

- `semantics_contract`: every Gluon candidate must preserve launcher shape,
  indexing, masks and boundaries, correctness behavior, and benchmark intent.
  Compile-only success is not enough.
- `dialect_plain_triton`, `dialect_nv_gluon`, `dialect_amd_gluon`: choose
  whether the task is a minimal layout rewrite, an NVIDIA-to-AMD translation, or
  an in-dialect AMD Gluon optimization.
- `layout_basic`, `layout_slice_broadcast`, `layout_source_first_required`:
  decide whether the candidate can start from `BlockedLayout`, needs
  `SliceLayout` / mask-preserving conversion, or must read operator-local layout
  code before planning.
- `memory_generic`, `memory_amd_buffer`, `memory_shared_async_descriptor`:
  stage memory lowering from `gl.load` / `gl.store`, to AMD
  `buffer_load` / `buffer_store`, to shared-memory, descriptor, `tdm`, or async
  paths only after simpler candidates are correct.
- `matrix_none`, `matrix_dot`, `matrix_scaled_dot`,
  `matrix_wmma_descriptor`: decide whether matrix lowering is absent, requires
  result/operand layout plus `mfma` / `wmma`, needs scaled-op constraints, or
  belongs to the gfx1250 descriptor/WMMA family.
- `execution_jit_aot_sensitive`, `version_sensitive`,
  `operator_support_sensitive`: make JIT/AOT, Triton minor version,
  `instr_shape`, target backend, and operator-local arch support part of the
  planning contract.

### 8.4 Candidate slot policy

When AMD Gluon is in the output search space, early planning should allocate:

1. a minimal AMD Gluon viability candidate,
2. one trait-specific AMD Gluon candidate that names the traits it addresses,
3. a plain Triton fallback or competitor when allowed.

Only after a simpler AMD Gluon candidate passes correctness should the planner
escalate to shared-memory swizzles, async copy, descriptors, scheduler hints,
persistent kernels, atomics, or work stealing.

For later rounds, previous tasks and verified evaluations should narrow the
next candidate:

- if Gluon failed to compile or pass correctness, shrink the next attempt to a
  layout-only, translation-only, or memory-only step;
- if Gluon passed correctness but regressed performance, try memory or matrix
  lowering before scheduler-level changes;
- if plain Triton wins, keep it as the selected fallback rather than forcing
  more Gluon work.

### 8.5 Strong negative examples

The planner should reject or deprioritize tasks that:

- introduce a new top-level `gluon` kernel type;
- produce a new optimized `nv_gluon` output path;
- keep NVIDIA layout assumptions unchanged on AMD;
- rename NVIDIA `tma`, WGMMA, tensor-memory, or cluster APIs to guessed AMD
  names;
- replace `tl.dot` with an AMD matrix op without result and operand layouts;
- add MFMA when the source has no real matrix trait;
- start with descriptor, async, persistent, scheduler, atomics, or work stealing
  before a simpler AMD Gluon candidate passes correctness;
- claim success from compile-only validation without benchmark comparison.

### Search space: base_shared_extension

GEAK should plan Triton-family optimization as a union of three sets:

- **Base Set: plain Triton**. Preserve the main Triton planner path:
  algorithmic rewrites, memory and layout cleanup, fusion, shape-specialized
  variants, and lower-priority autotune or launch tuning.
- **Shared Set: common strategy, dialect-specific implementation**. Use this for
  tiling, blocking, mask simplification, memory coalescing, shape
  specialization, split/decomposition, and fusion strategies that could be
  implemented as either `plain_triton` or `amd_gluon`.
- **Extension Set: AMD Gluon**. Use this for minimal AMD Gluon viability,
  trait-specific lowering, `nv_gluon -> amd_gluon` translation, existing
  `amd_gluon` in-dialect optimization, and MFMA / WMMA / scaled / descriptor
  paths.

Gluon is additive. It should expand the search space, not replace the Base Set.
When GPU budget is small, preserve at least one Base Set candidate and one
minimal Extension Set candidate if AMD Gluon is allowed. With larger budgets,
add Shared Set paired variants only when they can be compared under the same
benchmark contract.

Planning rules:

- Do not replace all plain Triton tasks with Gluon tasks.
- Shared Set tasks must state whether they are a `plain_triton variant`, an
  `amd_gluon variant`, or a paired comparison.
- If a plain Triton candidate wins the benchmark, accept it as the selected
  fallback rather than forcing more Gluon work.
- If a Triton strategy wins and maps cleanly to Gluon traits, a later round may
  create an AMD Gluon variant of that winning strategy.
- If Gluon fails compile or correctness, shrink the next Extension Set attempt
  to layout-only, translation-only, or memory-only work.
- If Gluon passes correctness but is slower, refine memory or matrix lowering
  before scheduler, persistent, async, or descriptor work.

## 9. Representative examples

These examples are intentionally different from the checked-in test fixtures.
They are documentation examples, not golden outputs.

### 9.1 Example: `plain_triton -> amd_gluon` row-wise affine transform

Suppose the source kernel applies `output[row, col] = input[row, col] * scale[row] + bias[row]`.
The plain Triton version may leave layout implicit. The AMD-facing Gluon version
should make the one-dimensional row tile explicit before lowering memory ops.
This example is intentionally end-to-end: it includes the host-side layout
builder and launcher, because that is where many first rewrites go wrong.

```python
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

def make_cdna_1d_layout(block_size: int, num_warps: int):
    lanes_per_cta = 64 * num_warps
    assert block_size % lanes_per_cta == 0
    return ttgl.BlockedLayout(
        size_per_thread=[block_size // lanes_per_cta],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )

@gluon.jit
def row_affine_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_cols,
    COL_BLOCK: ttgl.constexpr,
    layout: ttgl.constexpr,
):
    row = ttgl.program_id(0)
    cols = ttgl.arange(0, COL_BLOCK, layout=layout)
    mask = cols < n_cols

    x = ttgl.amd.cdna3.buffer_load(ptr=x_ptr + row * n_cols, offsets=cols, mask=mask, other=0.0)
    scale = ttgl.load(scale_ptr + row)
    bias = ttgl.load(bias_ptr + row)
    y = x * scale + bias
    ttgl.amd.cdna3.buffer_store(
        ptr=out_ptr + row * n_cols,
        offsets=cols,
        stored_value=y,
        mask=mask,
    )

def launch_row_affine(x, scale, bias, out, col_block=256, num_warps=4):
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    layout = make_cdna_1d_layout(col_block, num_warps)
    grid = (n_rows,)
    row_affine_kernel[grid](
        x,
        scale,
        bias,
        out,
        n_cols,
        COL_BLOCK=col_block,
        layout=layout,
        num_warps=num_warps,
    )
```

What this example is showing:

- the launcher stays Triton-like
- host code may need to construct the layout and pass it into the kernel
- the layout must agree with `col_block` and `num_warps`
- the main row tile uses an AMD-facing memory path while scalar row parameters
  can remain generic
- for non-contiguous tensors, pass explicit strides instead of relying on
  `row * n_cols`

### 9.2 Example: `nv_gluon -> amd_gluon` tile reader rewrite

Suppose an NVIDIA-facing Gluon kernel reads a two-dimensional tile using a
wave32-centric blocked layout and later plans to use an async-copy path. The
first AMD rewrite should often keep the tile logic but replace the layout and
drop the unsupported fast path until parity is justified.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.jit
def tile_reader_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    n_cols,
    ROW_BLOCK: ttgl.constexpr,
    COL_BLOCK: ttgl.constexpr,
    layout: ttgl.constexpr,
):
    pid = ttgl.program_id(0)
    row_offsets = pid * ROW_BLOCK + ttgl.arange(0, ROW_BLOCK, ttgl.SliceLayout(1, layout))
    col_offsets = ttgl.arange(0, COL_BLOCK, ttgl.SliceLayout(0, layout))
    mask = (row_offsets < n_rows)[:, None] & (col_offsets < n_cols)[None, :]

    tile = ttgl.amd.cdna3.buffer_load(
        ptr=in_ptr,
        offsets=[row_offsets[:, None], col_offsets[None, :]],
        mask=mask,
        other=0.0,
    )
    ttgl.amd.cdna3.buffer_store(
        ptr=out_ptr,
        offsets=[row_offsets[:, None], col_offsets[None, :]],
        stored_value=tile,
        mask=mask,
    )
```

What this example is showing:

- preserve the common tile logic first
- switch to an AMD-valid layout and memory path
- do not mechanically carry over an NVIDIA-only async or descriptor path

### 9.3 Example: Triton-version-compatible MFMA layout guard

Some real code has to survive Triton minor-version differences in
`AMDMFMALayout.instr_shape`.

```python
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

def parse_triton_version(version: str) -> tuple[int, ...]:
    version = version.split("+")[0].split("-")[0]
    parts = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)

TRITON_VERSION_GE_3_6_0 = tl.constexpr(parse_triton_version(triton.__version__) >= (3, 6, 0))

@gluon.jit
def mfma_guarded_kernel(x_ptr, y_ptr, out_ptr, K_BLOCK: gl.constexpr, CDNA_VERSION: gl.constexpr):
    if TRITON_VERSION_GE_3_6_0:
        instr_shape: gl.constexpr = [16, 16, K_BLOCK]
    else:
        instr_shape: gl.constexpr = [16, 16]

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=instr_shape,
        transposed=True,
        warps_per_cta=[1, 4],
    )
```

What this example is showing:

- Triton version can be part of the kernel-generation contract
- layout generation may need compatibility guards
- this is a real module-level version check, not placeholder pseudocode
- this is especially relevant for mixed JIT and AOT environments

### 9.4 Example: JIT with explicit AOT fallback

Some production code wants a JIT path when `triton.experimental.gluon` exists,
but a fallback path when only prebuilt kernels are available.

```python
try:
    from triton.experimental import gluon
    GLUON_JIT_ENABLED = True
except ImportError:
    GLUON_JIT_ENABLED = False

def launch_or_fallback(x, y, out):
    if GLUON_JIT_ENABLED:
        grid = (1,)
        return gluon_kernel[grid](x, y, out, num_warps=4)
    return call_prebuilt_gluon_aot_kernel(x, y, out)
```

What this example is showing:

- JIT availability is not guaranteed in every downstream environment
- some real integrations intentionally carry both paths
- GEAK should preserve that decision surface when the codebase already depends
  on it

## 10. Current repo-local notes

GEAK's current checked-in defaults for this feature are intentionally small:

- default target backend: `hip/gfx942`
- default Triton feature mode: `auto`
- default Triton output search space: `plain_triton` and `amd_gluon`
- current repo-local profile focus: `mi3xx`
- example surface: `examples/triton_gluon_inputs/`
- no checked-in golden outputs
- no checked-in run manifests, preprocess snapshots, profile JSON, or benchmark
  dumps in `examples/`

The accepted checked-in input forms are:

- `plain_triton`
- `nv_gluon`
- `amd_gluon`

GEAK should generate and benchmark candidate outputs at run time.

## 11. Benchmark-aware rules

- compile-only success is not enough to claim the path is valid
- preserve correctness before optimizing for speed
- when a harness supports multiple modes, keep one ordered case stream across
  correctness, profile, and benchmark
- if `plain_triton` remains an allowed output, compare it against `amd_gluon`
  instead of assuming AMD Gluon wins automatically
- treat Triton minor version, JIT vs AOT availability, and target architecture
  as part of the benchmark contract

## 12. Anti-patterns

- introducing a new top-level `gluon` kernel type
- treating conceptual correspondences as 1:1 API renames
- keeping NVIDIA layout assumptions unchanged on AMD
- mixing AMD and NVIDIA layout families in one kernel path
- hardcoding repo-local absolute paths, branch names, or container names in
  reusable syntax guidance
- treating checked-in examples or optimization logs as benchmark truth
- assuming every AMD target should use the same CDNA3/4-style recipe
- assuming descriptor, tensor-memory, or cluster APIs are portable by name

## Appendix A: API usage

### A.1 Imports

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

AMD-heavy code often uses:

```python
from triton.experimental.gluon import language as ttgl
```

### A.2 JIT entry and host launcher

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

Relevant runtime facts:

- the launcher stays Triton-like
- when layout depends on launch choices, build it on the host and pass it in
- `GluonASTSource` uses `Language.GLUON`
- launch attributes include target, warps, CTAs, and threads-per-warp
- if you intentionally stay vendor-neutral, the host layout pattern is the same;
  only the `load` / `store` lines change back to `gl.load` / `gl.store`

### A.3 Core language and layout surface

| API | Purpose |
|-----|---------|
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
| `convert_layout` | Move a tensor into another layout |
| `zeros(..., layout=...)` | Create accumulator/register tensors with explicit layout |
| `to_linear_layout` | Convert to a linearized layout view when needed |
| `set_auto_layout` | Allow automatic layout selection in constrained flows |

Recommended order:

1. choose the layout
2. build indices with that layout
3. load values
4. convert layout if needed
5. run the target-specific op

Decision table for common Triton rewrites:

| Plain Triton pattern | First Gluon rewrite | Escalate when |
|----------------------|---------------------|---------------|
| `tl.arange(0, XBLOCK)` | `gl.arange(0, XBLOCK, layout=layout)` | always; Gluon needs the layout to be explicit |
| `tl.load` / `tl.store` | `gl.load` / `gl.store` | switch to AMD `buffer_load` / `buffer_store` when the target family or the existing AMD code path actually depends on it |
| `tl.zeros((M, N), dtype=...)` | `gl.zeros((M, N), dtype=..., layout=layout)` | always; accumulators need explicit layout too |
| `tl.dot` / `tl.dot_scaled` | result layout + operand layouts + `convert_layout` + target-specific matrix op | always; this is not a direct rename target |
| tensor descriptor helpers | target-specific descriptor family | only when the target family actually supports descriptors or tensor memory |
| implicit shared-memory staging from a tuned Triton kernel | `allocate_shared_memory` after the first correct candidate | only after the blocked-layout or matrix path is already correct |

### A.4 Shared memory, synchronization, and cluster surface

```python
shared_a: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[1, 0])
shared_b: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[0, 1])

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

This is the simplest concrete shared-memory staging pattern to copy from.

If the target family later adds an async path, keep the same high-level phase
structure:

1. issue async transfer
2. commit
3. wait
4. consume from shared memory or continue compute

Do not copy literal async function names between NVIDIA, CDNA, and `gfx1250`
families. The phase structure is common; the APIs are not.

### A.5 Descriptor and tensor-memory surface

NVIDIA-side:

- host `TensorDescriptor`
- device `tma`
- `tensor_memory_descriptor`
- `TensorMemoryLayout`
- `tcgen05_*`

AMD-side:

- gfx1250 host `TensorDescriptor`
- `tdm`
- `async_load`
- `async_wait`
- `prefetch`
- `async_scatter`

### A.6 AMD quick patterns

#### CDNA3 MFMA pattern

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

#### CDNA4 scaled-MFMA pattern

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

#### `gfx1250` WMMA pattern

```python
wmma_layout = ttgl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, instr_shape_k])
a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, wmma_layout, k_width))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, wmma_layout, k_width))
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

Key constraints that are easy to miss:

- prefer getting plain `wmma` working before adding scaled WMMA
- `wmma_scaled` expects stricter layout contracts than plain `wmma`
- when an input format is `e2m1`, the operand WMMA layout expects
  `instr_shape=[16, 16, 64]`
- the accumulator layout for scaled WMMA expects `instr_shape=[16, 16, 128]`
- `get_wmma_scale_layout` only supports scale factor `16` or `32`
- scale dtype combinations are limited; do not guess them by name

#### `gfx1250` descriptor constraints

Practical rules from the frontend checks:

- the descriptor shape rank must be between 1 and 5
- the last tensor dimension must be contiguous
- only `PaddedSharedLayout`, `SwizzledSharedLayout`, or
  `PartitionedSharedLayout` are valid descriptor layouts
- for the currently accepted swizzled cases, `max_phase` must stay at `1`
- only `"zero"` padding is supported

Treat descriptor setup as a correctness contract, not as a later optimization.

### A.7 NVIDIA quick patterns

#### Ampere or Hopper async copy

```python
cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
cp.commit_group()
cp.wait_group(0)
value = smem.load(layout)
```

#### Hopper WGMMA pattern

```python
fence_async_shared()
acc = warpgroup_mma(a, b_smem, acc, is_async=True)
warpgroup_mma_wait(acc, deps=(...))
```

#### Blackwell tensor-memory or TCGen05 pattern

```python
tmem = allocate_tensor_memory(...)
tcgen05_copy(...)
tcgen05_mma(...)
tcgen05_commit(...)
```

### A.8 Version and compatibility checklist

| Check | Why it matters | Typical questions |
|-------|----------------|-------------------|
| Triton minor version | `instr_shape` and some AOT metadata differ across versions | is this code path expecting `3.5` or `3.6+`? |
| `AMDMFMALayout.instr_shape` form | 2D vs 3D affects whether layout construction succeeds | should this be `[M, N]` or `[M, N, K]`? |
| Execution mode | some downstreams use JIT, AOT, or both | does the code need `triton.experimental.gluon`, prebuilt kernels, or both? |
| Target backend and arch | wave size, memory path, and matrix family all depend on it | `hip/gfx942`, `hip/gfx950`, `hip/gfx1250`, or CUDA target? |
| Global feature availability | coarse helpers do not define per-operator support | does "Gluon available" really mean this operator is supported here? |
| Operator-local support matrix | real kernels may have narrower or different guards | does this specific attention or GEMM path support the arch? |
| AOT scratch requirements | some AOT pipelines reject kernels with scratch requirements | does the kernel need global or profile scratch? |

Treat Triton version, execution mode, and target architecture as part of the
benchmark and integration contract, not as incidental metadata.

### A.9 Common pitfalls

#### General

- do not start from copied tutorial code without checking the target family
- do not treat compile or IR validation as proof that profiling will work
- do not mix environment changes, harness changes, and kernel changes in one
  unexplained iteration

#### NVIDIA-side

- do not assume `tma` concepts port directly to AMD
- do not assume Hopper and Blackwell matrix paths are interchangeable
- do not forget async ordering or fence semantics around shared-memory paths

#### AMD-side

- do not mix `mfma` and `wmma` mental models
- do not assume descriptor-style paths are uniformly available across CDNA3,
  CDNA4, and `gfx1250`
- do not assume every benchmark-valid task is profiler-ready
- do not assume the namespace name alone fully determines the architecture
  contract
- do not treat host-side layout construction as optional when layout depends on
  launch configuration
- do not assume `DistributedLinearLayout` and unshuffle transforms can be
  simplified away without changing the algorithm
- do not guess `wmma_scaled` scale formats, scale factor, or accumulator layout
  from naming alone

### A.10 Common failures and fix order

| Symptom | Inspect first | Typical fix |
|---------|---------------|-------------|
| layout or IR verification fails | `BlockedLayout`, `threads_per_warp`, `warps_per_cta`, `order`, `num_warps`, target arch | make the layout consistent with the launch contract before changing APIs again |
| `AMDMFMALayout` or `instr_shape` construction fails | Triton version, 2D vs 3D `instr_shape`, layout version | add a real version guard and match the expected layout form |
| kernel compiles but target path is wrong | backend, arch, operator-local guards, namespace vs layout version | re-check the actual support matrix for the operator, not only a global helper |
| JIT Gluon import is missing | whether the downstream expects JIT, AOT, or both | preserve or add the existing fallback path instead of deleting it |
| AOT compilation fails with scratch-related error | `global_scratch_size` or `profile_scratch_size` | keep JIT for that path or redesign the kernel; current AOT helpers may not support it |
| correctness is fine but performance regresses | baseline comparison, memory path choice, shared-memory staging order | keep the plain Triton baseline and add AMD-specific features incrementally |
| `gfx1250` descriptor construction fails | contiguity of the last dimension, layout family, swizzle settings, padding mode | satisfy descriptor assertions first instead of weakening the layout model |
| `wmma_scaled` fails or asserts | input format, operand `instr_shape`, accumulator layout, scale factor, scale dtype combination | get plain `wmma` working first, then match the exact scaled-WMMA contract |
| preshuffled GEMM gives wrong answers | `DistributedLinearLayout`, unshuffle `reshape` / `permute` / `trans`, K divisibility assumptions | preserve the transformation sequence before tuning loads or matrix ops |

Suggested debug order:

1. confirm runtime version, backend, and arch
2. confirm launcher and layout alignment
3. confirm memory path selection
4. confirm matrix layout and `instr_shape`
5. only then add shared-memory, async, or scheduler features
