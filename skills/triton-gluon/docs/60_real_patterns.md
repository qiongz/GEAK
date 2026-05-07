# Triton-Gluon Real Patterns And Benchmark Rules

Read this file when docs or examples are not enough: architecture families,
real aiter patterns, benchmark rules, repo-local defaults, and source-first
triggers live here.

Do not read this whole file by default. Use the routed section(s) below.

## Read Only These Sections

| If the task needs... | Read |
| --- | --- |
| product/runtime context beyond `00_always_read.md` | `product_and_runtime_context` |
| plain Triton -> Gluon rewrite order and implicit layout recovery | `writing_model` |
| first-pass scope for layout-heavy kernels | `extension_l0_scope_for_layout_heavy_kernels` |
| L1 memory/buffer lowering after a viable Gluon patch | `extension_l1_memory_lowering_anchor` |
| layout/sync/descriptor concepts without exact API snippets | `layout_sync_descriptor_mental_model` |
| AMD/NVIDIA family comparison or namespace-vs-arch nuance | `nvidia_amd_family_differences` |
| translator/current_target-based AMD lowering | `translator_derived_amd_dispatch` |
| aiter attention/GEMM/MQA examples and operator-local support | `real_patterns_from_aiter`, `operator_local_support_matrix` |
| elementwise/attention/GEMM/preshuffled/gfx1250 strategy order | `optimization_paths_by_kernel_family` |
| source-first triggers | `source_first_triggers` |
| repo-local defaults or benchmark interpretation | `repo_local_notes`, `benchmark_aware_rules` |
| final sanity check for risky patterns | `anti_patterns` |

## Internal Index

- `product_and_runtime_context`
- `writing_model`
- `extension_l0_scope_for_layout_heavy_kernels`
- `extension_l1_memory_lowering_anchor`
- `layout_sync_descriptor_mental_model`
- `nvidia_amd_family_differences`
- `translator_derived_amd_dispatch`
- `real_patterns_from_aiter`
- `operator_local_support_matrix`
- `optimization_paths_by_kernel_family`
- `source_first_triggers`
- `repo_local_notes`
- `benchmark_aware_rules`
- `anti_patterns`

## product_and_runtime_context

GEAK treats Gluon as a feature extension of Triton:

- keep `kernel_type = triton`;
- infer `input_dialect` as `plain_triton`, `nv_gluon`, or `amd_gluon`;
- default Triton runs use `gluon_feature_mode = auto`;
- use `gluon_feature_mode = off | auto | force` only for ablation or forced
  debugging;
- prefer `amd_gluon` when allowed and structurally promising, while keeping
  plain Triton as a benchmarked fallback;
- never create a new optimized `nv_gluon` output path.

In Triton, Gluon is not just syntax sugar:

- `triton.experimental.gluon.jit` produces a `GluonJITFunction`;
- `GluonJITFunction` uses `GluonASTSource`;
- `GluonASTSource` sets the language to `Language.GLUON`;
- generated modules carry launch attributes such as `ttg.target`,
  `ttg.num-warps`, `ttg.num-ctas`, and `ttg.threads-per-warp`.

JIT and AOT are both real downstream modes:

- direct `@gluon.jit` kernels launched from Python;
- AOT-compiled Gluon kernels wrapped from C++ / PyTorch integration code;
- mixed pipelines with a Gluon main kernel and normal Triton helper kernels.

Triton minor version is part of the contract. Real code carries branches for
layout construction and AOT metadata, especially around Triton `3.5` versus
`3.6+` `AMDMFMALayout.instr_shape`.

## writing_model

Gluon keeps Triton's host-side launcher model:

- `@gluon.jit`;
- `kernel[grid](...)`;
- `triton.cdiv`;
- `program_id`;
- `constexpr`.

What changes is that low-level choices become explicit:

- layout selection;
- shared-memory usage;
- synchronization;
- descriptor and tensor-memory usage;
- target-specific memory paths;
- target-specific matrix instruction paths.

Start from target family and layout, not copied tutorial syntax.

For `plain_triton -> amd_gluon`, use this order:

1. Keep launcher and correctness semantics recognizable.
2. Reconstruct base tile and layout before touching target-specific ops.
3. Align host launch config and layout.
4. Choose memory path.
5. Choose matrix path if needed.
6. Add shared-memory or pipeline features only after the first candidate works.

The first Gluon candidate should usually be a mechanical,
semantics-preserving rewrite unless the source is already AMD Gluon. The goal is
not to use the most advanced Gluon feature immediately; it is to create a
benchmark-valid candidate whose relationship to the baseline is easy to audit.

For implicit layout recovery:

- start from original logical tile shape;
- derive `BlockedLayout` matching original execution shape;
- for 1D kernels, satisfy
  `size_per_thread[0] * threads_per_warp[0] * warps_per_cta[0] == XBLOCK`;
- on MI3xx-style CDNA targets, prefer a wave64-valid first candidate such as
  `threads_per_warp=[64]`;
- for 2D kernels, choose `order` so the fastest-varying logical dimension still
  matches the coalesced dimension;
- derive `SliceLayout` or `DotOperandLayout` from the base layout instead of
  inventing them independently.

For plain Triton, implicit execution shape is hidden in `tl.arange` bounds,
tile/block constants, pointer arithmetic, coalesced dimension, `num_warps`, and
wave32/wave64 assumptions. Recover those before choosing target-specific memory
or matrix ops.

Host launcher and layout alignment:

- fixed compile-time layout can live in the kernel;
- layout depending on `BLOCK_*`, `num_warps`, target family, or tuning choices
  should be built on the host and passed as `constexpr`;
- keep launcher in `kernel[grid](...)` form;
- align `num_warps`, `num_ctas`, target arch, and layout because Gluon IR
  verification checks the launch attributes.

Stay in plain Triton when:

- the kernel is simple and already expresses the right execution shape;
- explicit layouts add complexity without plausible performance upside;
- the target-specific path would only be compile-valid, not benchmark-valid.

`gl.load` / `gl.store` are not a failure to use AMD. They are often the right
first Gluon rewrite for scalar or simple vector paths. Move to
`buffer_load` / `buffer_store` only when target family, existing AMD structure,
or access pattern justifies it.

## extension_l0_scope_for_layout_heavy_kernels

For layout-heavy kernels, Extension L0 should establish a small compileable
Gluon viability path. It should not translate the whole algorithm in one patch.

Layout-heavy signals include:

- multiple logical 2D contexts that need different `SliceLayout` parents;
- `[:, None]` / `[None, :]` broadcasts;
- masks built from multiple axes;
- reductions such as `sum`, `max`, softmax, or online softmax;
- more than one matrix path (`QK` and `PV`, for example);
- RoPE, preshuffle, descriptor, or nested layout transformations.

Prefer one of these L0 scopes:

- index and mask layout smoke path for one logical 2D expression;
- one load/store subpath with explicit layout;
- one small matrix-layout skeleton without full epilogue;
- one source-first extraction of layout contracts and host launcher alignment.

For the chosen L0 scope, fully convert that subpath to Gluon. Do not leave a
plain Triton island such as `tl.arange(0, BLOCK_R)` in a RoPE or mask branch
inside `@gluon.jit`; if a branch is too hard to convert, it is outside the L0
scope and should remain outside the patch.

Treat L0 as an execution and layout anchor, not a performance promise. A useful
L0 patch proves that the selected Gluon path really runs, preserves correctness,
and exposes layout/memory evidence for later tasks. If L0 is slower than the
Base path, record the overhead source when visible and avoid repeated launch
constant tuning unless the next patch has a concrete reason it should remove
that overhead.

Patch evolution model:

- `patch_0`: smallest real executed Gluon path that can compile and pass
  correctness.
- `patch_1+`: one change at a time, such as one layout repair, one memory op,
  one matrix subpath, one launch constant, or one dispatch condition.
- Before the next patch, record `Changed component`, `Expected effect`,
  `Observed effect`, and `Keep / revert / compose later`.
- If a patch bundles unrelated changes, later rounds cannot tell whether Gluon
  helped or was masked by another regression.

Defer full attention/decode/GEMM rewrites until after L0 proves the relevant
layout family compiles. L1 tasks can then add memory lowering, matrix lowering,
or shared/descriptor features one at a time.

## extension_l1_memory_lowering_anchor

Extension L1 memory or buffer lowering is not a second attempt to wrap the
original plain Triton kernel in `@gluon.jit`.

Before adding `buffer_load`, `buffer_store`, or other target-specific memory
ops, identify the verified Gluon anchor:

- the last correctness-passing Gluon or mixed patch;
- its parent layouts for each logical expression;
- its host-created layouts and `constexpr` launch contract;
- its known slow path or memory-bound section.

Rules:

- Preserve the verified anchor's layout plan. Do not regenerate indices with
  plain `tl.arange` inside `@gluon.jit`.
- Before editing, write why the memory change should improve performance:
  fewer global transactions, better coalescing, less masking overhead, or a
  known memory-bound hot path. If the reason is only "AMD buffer ops might be
  faster", keep generic `gl.load` / `gl.store`.
- Change one memory path at a time, such as one KV/cache load, one streaming
  vector load, or one output store.
- For a task named or scoped as buffer/load/store lowering, keep
  `Matrix path: none` unless the task explicitly says matrix lowering or
  `bundle_allowed=true`. Do not introduce MFMA as a side quest in a buffer-load
  patch.
- Keep masks and broadcast indices in the same parent-layout context as the
  anchor.
- If no Gluon anchor has passed correctness, downgrade the L1 memory task to a
  narrower layout/memory smoke path and report that no anchor exists.
- Compare against the verified Gluon anchor as well as the original baseline.

Extension L1 matrix/MFMA lowering follows the same anchor rule. Do not use MFMA
as the first real Gluon attempt for a layout-heavy kernel. The task must name the
anchor layout, result layout, operand layouts, and exact single matrix subpath
being lowered; otherwise keep it as an L0 layout skeleton.

MFMA performance viability checklist before editing:

- the selected dot/matrix subpath is on the benchmark hot path;
- result and operand layouts avoid repeated `convert_layout` inside a loop;
- the patch does not add a second launch or host dispatch branch that dominates
  the measured case;
- accumulator dtype and store dtype are planned before the epilogue;
- expected speedup comes from replacing real matrix work, not from merely making
  a small MFMA skeleton compile.

## layout_sync_descriptor_mental_model

Real Triton Gluon code uses more than `BlockedLayout`.

| Layout / helper | Main purpose |
| --- | --- |
| `BlockedLayout` | Base thread/warp/CTA distribution |
| `SliceLayout` | Select a sub-dimension from a parent layout |
| `DotOperandLayout` | Operand layout for matrix instructions |
| `DistributedLinearLayout` | Explicit thread/register mapping |
| `SwizzledSharedLayout` | Shared-memory swizzle |
| `PaddedSharedLayout` | Shared-memory padding semantics |
| `PartitionedSharedLayout` | gfx1250 / RDNA-style partitioned shared-memory path |
| `AMDMFMALayout` | AMD MFMA result layout |
| `AMDWMMALayout` | AMD WMMA result layout |
| `TensorMemoryLayout` | Blackwell tensor-memory layout |

Recommended workflow:

1. choose target family;
2. choose base layout;
3. derive slice, dot-operand, shared, or descriptor layouts;
4. build indices with that layout;
5. lower memory or instruction paths.

Useful synchronization/pipeline concepts:

- `allocate_shared_memory`;
- `barrier`;
- `mbarrier`;
- `cluster`;
- `fence_async_shared`;
- `warp_pipeline_stage`.

Do not assume these are interchangeable across vendors:

- Hopper and Blackwell expose one cluster/barrier family;
- gfx1250 exposes another;
- AMD also has optional wave or warp pipeline guidance.

Descriptor and tensor-memory concepts are not 1:1 substitutions:

- NVIDIA side: host `TensorDescriptor`, device-side `tma`, Blackwell tensor
  memory, `TensorMemoryLayout`, `tcgen05_*`;
- AMD side: gfx1250 host `TensorDescriptor`, `tdm`, `wmma`,
  `PartitionedSharedLayout`.

## nvidia_amd_family_differences

Common Gluon layer from `triton.experimental.gluon.language`:

- `program_id`;
- `constexpr`;
- `BlockedLayout`;
- `SliceLayout`;
- `DotOperandLayout`;
- `arange`;
- `load`;
- `store`;
- `zeros`;
- `convert_layout`;
- `allocate_shared_memory`;
- `barrier`.

NVIDIA families:

| Family | Typical focus |
| --- | --- |
| Ampere | `async_copy`, `mma_v2` |
| Hopper | `tma`, `mbarrier`, `cluster`, `warpgroup_mma` |
| Blackwell | `tensor_memory_descriptor`, `TensorMemoryLayout`, `tcgen05_*`, `clc`, richer TMA patterns |

NVIDIA-oriented Gluon often assumes wave32-style execution and newer descriptor
or warpgroup features. Recheck those assumptions before moving to AMD.

AMD families:

| Family | Typical focus |
| --- | --- |
| CDNA3 / `gfx942` | `buffer_load`, `buffer_store`, `mfma`, `AMDMFMALayout` |
| CDNA4 / `gfx950` | CDNA3 ops plus `async_copy`, `mfma_scaled`, `get_mfma_scale_layout` |
| RDNA3 / RDNA4 | `wmma` |
| `gfx1250` | `wmma`, `wmma_scaled`, `tdm`, `async_copy`, `mbarrier`, `cluster`, `AMDWMMALayout` |

Practical target differences:

- `gfx942` / CDNA3: best first target for current GEAK feature work; prefer
  wave64-valid layouts, `buffer_load` / `buffer_store`, and MFMA only for real
  matrix-op kernels.
- `gfx950` / CDNA4: use CDNA4-only surfaces when the kernel truly benefits;
  `mfma_scaled` and `get_mfma_scale_layout` are high-value additions, not
  mandatory defaults.
- `gfx1250`: treat as a separate WMMA/descriptor family, not CDNA with renamed
  APIs. Get plain `wmma` or a basic descriptor path working before adding
  `wmma_scaled`, `tdm`, async, or cluster behavior.

Module path versus architecture version can differ. Real code may use
`gl.amd.cdna3.*` helpers while layout or feature branches target
`AMDMFMALayout(version=4)` / `gfx950`. Read target arch, layout version, version
guards, and feature branches before deciding support.

Concept map:

| Concept | NVIDIA | AMD |
| --- | --- | --- |
| Matrix path | `mma_v2`, `warpgroup_mma`, `tcgen05_mma` | `mfma`, `wmma` |
| Async global -> shared | `async_copy_*`, `tma.*` | family-specific `async_copy.*` |
| Descriptor-like path | `tma`, `TensorDescriptor` | `tdm` |
| Barrier / cluster | `mbarrier`, `cluster` | family-specific barrier / cluster APIs |
| Tensor memory | Blackwell tensor memory | no direct global AMD equivalent; use gfx1250-specific `tdm` / WMMA where appropriate |

## translator_derived_amd_dispatch

Triton's `triton_to_gluon_translator` AMD helper code is a useful mental model
for migration tasks:

- Use `current_target()` or equivalent target metadata to dispatch, not string
  replacement.
- `gfx1250` routes matrix lowering through `AMDWMMALayout`, `DotOperandLayout`,
  `convert_layout`, and `wmma`.
- CDNA3/CDNA4 routes matrix lowering through `AMDMFMALayout`,
  `DotOperandLayout`, `convert_layout`, and `mfma`.
- CDNA3/CDNA4 MFMA instruction K width depends on target generation and element
  bitwidth.
- `tl.dot_scaled` style paths may decompose through target-specific dot helpers
  plus scale layouts; do not translate by name alone.
- `tl_make_tensor_descriptor` / descriptor object load/store are gfx1250-only
  AMD TDM concepts in this model.

Planner implication: if a task mentions `translator`, `current_target`,
`tl_make_tensor_descriptor`, `tdm`, or `TensorDescriptor`, route it to both
`30_architecture_notes.md` and this file before implementation.

## real_patterns_from_aiter

Paged-attention Gluon code on `gfx942` / `gfx950` shows production-style
composition:

- `BlockedLayout`;
- `SliceLayout`;
- `AMDMFMALayout(version=CDNA_VERSION, ...)`;
- `DotOperandLayout`;
- `allocate_shared_memory`;
- shared-memory swizzle;
- `buffer_load` / `buffer_store`;
- explicit stride-rich host arguments.

This is more representative than a toy vector-add kernel.

Toy vector examples are useful for syntax but weak evidence for real
optimization decisions.

GEMM and FP8 paths on `gfx950` additionally use:

- `mfma_scaled`;
- `get_mfma_scale_layout`;
- architecture-conditioned K widths and instruction shapes;
- JSON-config or heuristic-selected launch configs instead of only online
  autotune.

JIT and AOT can coexist in one operator family:

- pure JIT Gluon kernels;
- AOT-compiled Gluon kernels;
- mixed pipelines where only some stages are Gluon.

Optional scheduling, barrier, or priority hints exist in real paths. Treat them
as second-stage tuning features and target-specific hints, not first-pass
portability requirements.

Feature availability is not operator support:

- a coarse helper may mark only `gfx950` and `gfx1250` as Gluon-available;
- a specific attention kernel may still support `gfx942` and `gfx950`;
- a CDNA4 GEMM may require `gfx950` while another Gluon operator runs on
  `gfx942`.

Always read operator-local guards before concluding that an architecture is
supported or unsupported.

## operator_local_support_matrix

Do not use one global Gluon-available helper as the operator support matrix.
Observed downstream patterns differ by operator:

| Operator family | Observed Gluon path | Support notes |
| --- | --- | --- |
| Paged-attention decode | JIT Gluon main attention + normal Triton reduce, plus AOT wrapper path | CDNA3/CDNA4 style path; operator-local guards may include `gfx942` and `gfx950` |
| GEMM A8W8 / blockscale / AFP4WFP4 | Gluon GEMM with checked-in JSON configs | Existing configs may be `gfx950`-specific; do not infer `gfx942` support |
| PA MQA logits | JIT Gluon or prebuilt/AOT artifact selected by Triton version/env | Artifact shape can be zip/config/env driven rather than Jinja `.so` |
| gfx1250 descriptor/WMMA examples | WMMA/TDM/descriptor-oriented examples and tests | Separate family from CDNA; not a drop-in replacement for CDNA attention/GEMM |

Artifact shapes also differ:

- JIT-only: Python `@gluon.jit` launched directly.
- Jinja `.so`: Gluon AOT stage plus generated C++/pybind wrapper.
- Zip/config/env: prebuilt artifacts selected by environment variables and
  config lookup.

Treat these as operator-local integration contracts. A worker should preserve
existing artifact selection and environment gates unless benchmark evidence
proves they are irrelevant.

## optimization_paths_by_kernel_family

Elementwise or vector-style kernels:

1. preserve launcher, masks, and baseline behavior;
2. reconstruct correct blocked layout;
3. decide whether generic `gl.load` / `gl.store` is enough;
4. move to AMD `buffer_load` / `buffer_store` only if target family or existing
   code benefits;
5. benchmark against plain Triton before adding more structure.

This is the safest family for early `plain_triton -> amd_gluon` candidates.

Attention or decode kernels:

1. preserve stride-rich host arguments and partition logic;
2. preserve query, key, and value logical shapes before touching matrix ops;
3. keep nested `SliceLayout` trees and mask conversions semantically intact;
4. wire `AMDMFMALayout` and operand layouts only after the logical shape story
   remains correct;
5. treat optional scheduler, barrier, or staging hints as second-stage tuning.

The trap is treating reshape, mask conversion, or score-layout changes as
cosmetic. In real attention kernels they often affect correctness.

GEMM or FP8 kernels:

1. identify the actual matrix instruction family and K width;
2. pick result layout first;
3. derive operand layouts and `convert_layout` steps;
4. make epilogue correct: scales, bias, accumulation dtype, store layout;
5. only then add shared-memory staging, async features, preshuffle support, or
   tuned config selection.

In real `gfx950` code, config selection may come from heuristics or checked-in
JSON, not only online autotune.

Preshuffled GEMM:

- watch for `DistributedLinearLayout`;
- preserve `reshape` / `permute` / `trans` unshuffle sequences;
- preserve assumptions that K is a multiple of the preshuffled block shape;
- optimize only after the unshuffle is still correct.

gfx1250 WMMA or descriptor kernels:

1. get plain `wmma` or plain descriptor load/store working;
2. validate layout family, contiguous-last-dimension requirements, and shared
   layout constraints;
3. only then add `wmma_scaled`, scale layouts, `tdm`, async paths, or cluster
   logic.

This family has more frontend assertions than current CDNA3/4 paths, so
speculative rewrites fail quickly.

## source_first_triggers

Stop generic rewriting and read operator-local source when you see:

- `DistributedLinearLayout`;
- `PartitionedSharedLayout`;
- host `TensorDescriptor`;
- `reshape` / `permute` / `trans` used to unshuffle matrix tiles;
- 3D or 5D logical layouts with multiple nested `SliceLayout`;
- environment-variable gates around JIT versus AOT paths;
- prebuilt-kernel loading, AOT packaging, or generated asset lookup;
- scheduler, barrier, or priority hints that appear to affect launch shape or
  correctness.

The general playbook still helps in these cases, but it is no longer sufficient
on its own.

## repo_local_notes

GEAK's current checked-in defaults for this feature are intentionally small:

- default target backend: `hip/gfx942`;
- default Triton feature mode: `auto`;
- default Triton output search space: `plain_triton` and `amd_gluon`;
- current repo-local profile focus: `mi3xx`;
- example surface: `examples/triton_gluon_inputs/`;
- no checked-in golden outputs;
- no checked-in run manifests, preprocess snapshots, profile JSON, or benchmark
  dumps in `examples/`.

Accepted checked-in input forms:

- `plain_triton`;
- `nv_gluon`;
- `amd_gluon`.

GEAK should generate and benchmark candidate outputs at run time.

Do not hardcode repo-local absolute paths, branch names, containers, manifests,
or optimization logs into reusable syntax guidance.

## benchmark_aware_rules

- Compile-only success is not enough.
- Preserve correctness before optimizing speed.
- When a harness supports multiple modes, keep one ordered case stream across
  correctness, profile, and benchmark.
- If `plain_triton` remains an allowed output, compare it against `amd_gluon`
  instead of assuming AMD Gluon wins.
- Treat Triton minor version, JIT/AOT availability, and target architecture as
  part of the benchmark contract.
- Treat checked-in examples and optimization logs as examples, not benchmark
  truth.
- Plain Triton winning is valid. It is not a failure to use Gluon.

## anti_patterns

- Introducing a new top-level `gluon` kernel type.
- Treating conceptual correspondences as 1:1 API renames.
- Producing optimized `nv_gluon`.
- Keeping NVIDIA layout assumptions unchanged on AMD.
- Mixing AMD and NVIDIA layout families in one kernel path.
- Textually replacing `tl.dot` with AMD matrix ops without result and operand
  layouts.
- Wrapping the original plain Triton body in `@gluon.jit` and then fixing
  compiler errors one by one instead of starting from a scoped layout plan.
- Treating L1 MFMA/buffer work as a fresh rewrite from the original plain
  Triton kernel when no correctness-passing Gluon anchor exists.
- Adding MFMA / WMMA when the source has no matrix trait.
- Assuming every AMD target should use the same CDNA3/4 recipe.
- Assuming descriptor, tensor-memory, or cluster APIs are portable by name.
- Starting with descriptor, async, persistent, scheduler, atomics, or work
  stealing before a simpler AMD Gluon candidate passes correctness.
- Treating checked-in examples, manifests, snapshots, or benchmark outputs as
  ground truth for a new run.
- Ignoring Triton minor-version compatibility when the codebase mixes JIT and
  AOT.
- Assuming one global arch check is the operator support matrix.
- Hardcoding shape literals or hiding bucket selection inside
  `@triton.heuristics` for multi-shape / bucketed harnesses.
