# Triton-Gluon Real Patterns And Benchmark Rules

Worker-routed evidence and benchmark doc. Read this file when docs or examples
are not enough: source-first triggers, benchmark boundary, real operator
patterns, repo-local defaults, and negative evidence live here. API syntax and
small skeletons belong in `50_api_reference.md`.

## Profile Routing Hints

- `nv_to_amd_translation`: use `nvidia_amd_family_differences` and source-first
  triggers to separate concepts that translate from APIs that must not be
  renamed.
- `shape_bucketed_dispatch` / `hybrid_dispatch*`: use benchmark boundary and
  real-pattern evidence to justify visible host dispatch. A kernel-only Gluon win
  does not replace a fair/full-operator path without same-ABI evidence.
- `shared_transplant` / `gluon_variant_from_anchor`: use safe-anchor evidence and
  portable-component rules. Preserve the anchor algorithm before changing memory,
  matrix, or dispatch behavior.
- Real attention/GEMM/descriptor paths are source-first. If the task shows
  `DistributedLinearLayout`, host `TensorDescriptor`, unshuffle transforms, or
  JIT/AOT gates, read operator-local source before generic rewriting.

Do not read this whole file by default. Use the routed section(s) below.

## Read Only These Sections

| If the task needs... | Read |
| --- | --- |
| product/runtime context beyond `00_always_read.md` | `product_and_runtime_context` |
| checking whether a guide claim is backed by real samples, backend support, or operator-local evidence | `evidence_inventory_for_guide_authoring` |
| plain Triton -> Gluon rewrite order and implicit layout recovery | `writing_model` |
| first-pass scope for layout-heavy kernels | `extension_l0_scope_for_layout_heavy_kernels` |
| L1 memory/buffer lowering after a viable Gluon patch | `extension_l1_memory_lowering_anchor` |
| layout/sync/descriptor concepts without exact API snippets | `layout_sync_descriptor_mental_model` |
| AMD/NVIDIA family comparison or namespace-vs-arch nuance | `nvidia_amd_family_differences` |
| translator/current_target-based AMD lowering | `translator_derived_amd_dispatch` |
| generalized writing patterns from real code, without copying one operator | `generalized_real_code_patterns` |
| real attention, GEMM, descriptor, or wrapper-heavy operator patterns | `real_operator_patterns`, `operator_local_support_matrix` |
| elementwise/attention/GEMM/preshuffled/gfx1250 strategy order | `optimization_paths_by_kernel_family` |
| end-to-end, wrapper-heavy, fair benchmark, or same-ABI comparison | `benchmark_boundary_and_integration_costs` |
| source-first triggers | `source_first_triggers` |
| repo-local defaults or benchmark interpretation | `repo_local_notes`, `benchmark_aware_rules` |
| final sanity check for risky patterns | `anti_patterns` |

## Internal Index

- `product_and_runtime_context`
- `evidence_inventory_for_guide_authoring`
- `writing_model`
- `extension_l0_scope_for_layout_heavy_kernels`
- `extension_l1_memory_lowering_anchor`
- `layout_sync_descriptor_mental_model`
- `nvidia_amd_family_differences`
- `translator_derived_amd_dispatch`
- `generalized_real_code_patterns`
- `real_operator_patterns`
- `operator_local_support_matrix`
- `optimization_paths_by_kernel_family`
- `benchmark_boundary_and_integration_costs`
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
- choose the optimization direction from the main Triton strategy first,
  then decide whether Gluon is a useful implementation layer for that direction;
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

## evidence_inventory_for_guide_authoring

Quick directions are guide rails for a first correct and plausible candidate,
not proof of the fastest possible kernel. Before turning a pattern into reusable
guidance, classify its evidence:

| Evidence source | Use it for | Do not use it for |
| --- | --- | --- |
| `skills/triton-gluon/docs/*.md`, `docs/triton_gluon.md`, and the ROCm Gluon knowledge-base | existing product contracts, routed headings, and known API names | inventing new API calls that are absent from docs or source |
| `40_examples.md` | reasoning shape, host/layout wiring, version guards, and fallback patterns | benchmark truth or constants to copy |
| upstream Gluon tutorials | common Gluon mental model: layouts, host launch, shared-memory phases, barriers | assuming target-specific tutorial APIs are AMD APIs |
| upstream block-scaled matrix tutorials | scale-packing reasoning for scaled matrix variants | assuming one scale packing works for every instruction shape |
| upstream AMD Gluon tests | WMMA, scaled WMMA, descriptor, and constraint examples | extrapolating CDNA MFMA behavior to RDNA/GFX1250 |
| upstream AMD Gluon examples | descriptor setup, split-K scale-layout rank changes, and codegen-check patterns | treating advanced example structure as required L0 scope |
| upstream AMD backend/tests with target-specific matrix, async, or descriptor support | architecture capability boundaries and verifier/lowering constraints | writing a concrete Gluon API template when no Gluon example exists |
| downstream real-operator Gluon sources | real operator composition, guards, layouts, JIT/AOT wiring, and launch/config practices | treating one operator's constants as universal defaults |
| external support matrices, low-precision GEMM configs, and integration tests | architecture/operator support side evidence | deriving Gluon API syntax directly |

Use evidence labels when writing new guidance:

- `exact_sample`: the API shape appears in a Gluon doc, tutorial, test, or real
  Gluon source.
- `backend_support`: the architecture/lowering appears in Triton AMD backend or
  tests, but the exact Gluon API call still needs sample confirmation.
- `operator_local`: the pattern is valid for a specific operator family or
  wrapper contract; generalize only the decision rule.
- `hypothesis`: the idea follows from hardware or source structure but needs
  profiling or a missing-doc report before becoming a rule.

If only backend or CK evidence exists for a `gfx950` feature, document the
capability boundary and verification checklist, not a concrete Gluon call
template. If a claim cannot be tied to one of the evidence labels above, do not
put it in a quick direction.

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

For existing production AMD Gluon operators, do not force the plain overlay
model. Use `source_origin=existing_amd_gluon_operator` only when the measured
baseline already executes that operator path, then preserve source contracts
first: JIT/AOT fallback, target guards, layout declarations, artifact selection,
and stage dependencies.

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
- derived layouts such as `SliceLayout` and `DotOperandLayout` should also be
  built on the host and passed as `constexpr`; do not create layout objects
  inside `@gluon.jit`;
- keep launcher in `kernel[grid](...)` form;
- align `num_warps`, `num_ctas`, target arch, and layout because Gluon IR
  verification checks the launch attributes.

Stay in plain Triton when:

- the kernel is simple and already expresses the right execution shape;
- explicit layouts add complexity without plausible performance upside;
- the target-specific path would only be compile-valid, not benchmark-valid.
- the Gluon task cannot name the optimization direction, plain Triton
  competitor, and measured hot path it improves.

Stay on the production Gluon in-dialect path when the source is already a
measured AMD Gluon operator. In that case, plain Triton is comparison or
fallback evidence, not a mandatory same-batch competitor for every direction.

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
- more than one matrix path or dataflow branch in the same stage;
- positional transforms, preshuffle/unshuffle transforms, descriptor paths, or
  nested layout transformations;
- boundary-specific transform branches such as "last block", "last token",
  "tail tile", "split boundary", or other special-case data paths.

Prefer one of these L0 scopes:

- scalar or 1D stage that feeds the measured output;
- index and mask layout smoke path for one logical 2D expression;
- one load/store subpath with explicit layout;
- one small matrix-layout skeleton without full epilogue;
- one source-first extraction of layout contracts and host launcher alignment.

Anti-example pattern:

- A stage that combines a positional transform, boundary-specific replacement,
  multiple matrix/dataflow paths, online reduction, split-boundary logic, and
  indirect memory access is not a good Round-1 full-stage L0 target.
- Prefer a smaller scoped path: one reduction, one load/store path, one
  index/mask layout probe, or one transform branch that executes and feeds the
  measured output.

For the chosen L0 scope, fully convert that subpath to Gluon. Do not leave a
plain Triton island inside the edited `@gluon.jit` branch. For example, if the
selected path is one transform or mask branch, its index creation, masks,
loads/stores, tensor creation, and reductions must all be layout-aware Gluon.
If a branch is too hard to convert completely, it is outside the L0 scope and
should remain outside the patch.

Treat L0 as an execution and layout anchor, not a performance promise. A useful
L0 patch proves that the selected Gluon path really runs, preserves correctness,
and exposes layout/memory evidence for later tasks. If L0 is slower than the
Base path, record the overhead source when visible and avoid repeated launch
constant tuning unless the next patch has a concrete reason it should remove
that overhead.

If `extension_intent=execution_anchor` and L0 passes correctness but is slower
on every benchmark shape, record `observed_speedup`, `overhead_source`, and
`not_viable_for_l1=true` unless the next task names a concrete removable
overhead. Do not automatically generate same-scope MFMA, buffer, shared-memory,
or scheduler follow-up.

If the selected scoped path cannot be implemented without touching forbidden
paths, do not reinterpret the task as a larger Gluon rewrite. Use the task's
execution path policy:

- `inline_scoped_helper`: only when the scoped Gluon code can be wired into the
  measured path without changing forbidden code.
- `separate_gluon_kernel`: only when the task accepts second-launch/temp-buffer
  overhead and marks the result as an execution anchor.
- `whole_jit_kernel`: only when the task explicitly states that the whole
  helper/kernel is the minimum executable unit. Keep this as a mechanical anchor
  unless the task separately allows matrix, buffer, scheduler, or dispatch
  lowering.
- `infeasible`: report the scope as infeasible and keep the next search width on
  Base/Shared or a separately named full-kernel performance candidate.

Do not satisfy L0 by adding an unused `@gluon.jit` helper next to an unchanged
plain Triton path. If the helper is the scoped L0 path, the measured host
dispatch must launch it and its output must feed the correctness result.

Low-latency / tiny-stage rule:

- If the benchmark case or scoped stage is already very short (roughly sub-100us),
  explicit Gluon layout and launch overhead can dominate. L0 should be the
  smallest executed anchor, not a tuning campaign.
- After a slower correctness-passing L0 on such a path, do not keep sweeping
  `num_warps`, block size, `num_stages`, or dispatch guards. Record
  `overhead_source`, `observed_speedup`, and `not_viable_for_l1` when the
  evidence shows regression or material per-shape loss.
- A later task may revisit the anchor only if it names the overhead it removes;
  otherwise spend the search width on Base or Shared candidates.

Patch evolution model:

- `patch_0`: smallest real executed Gluon path that can compile and pass
  correctness.
- `patch_1+`: one change at a time, such as one layout repair, one memory op,
  one matrix subpath, one launch constant, or one dispatch condition.
- The next patch is chosen by the previous patch result. If `patch_N` compiles,
  executes, and passes correctness, `patch_N+1` may make one normal improvement
  to the same task scope. If `patch_N` fails, `patch_N+1` fixes only the current
  failure layer and must not add a new optimization.
- Before the next patch, record `Changed component`, `Expected effect`,
  `Observed effect`, and `Keep / revert / compose later`.
- After helper-not-executed, fix only wiring/launch/output feeding. After a
  forbidden-scope failure, revert the forbidden change before any other edit.
  After a slow correctness pass, change only one named overhead source.
- Helper-only failure is not a layout or matrix failure. The next patch should
  only connect the already-written Gluon helper to the declared target path and
  measured output. Do not add new layout factories, MFMA paths, or wrapper API
  changes while the helper is still unexecuted.
- If a patch bundles unrelated changes, later rounds cannot tell whether Gluon
  helped or was masked by another regression.
- Quick directions and checked-in examples are starting points for `patch_0` or
  a single L1 component, not final answers. After correctness, follow this same
  patch evolution model instead of adding a new tuning workflow.

`patch_evolution_by_task_type`:

- Base/plain Triton tasks are not forced into this Gluon state machine. Keep
  them narrow, use normal strategy notes for attempts, and let tested/profiled
  patches compete through the existing postprocess path.
- Any task that enters Gluon worker docs must use the pass/fail dual track
  above. This includes L0, L1, Gluon variants, Hybrid/mixed dispatch, and
  existing production AMD Gluon tasks that receive Gluon worker context.
- Generated L0 overlays use `patch_0` only as a compile/execute/correctness
  anchor. If that anchor passes, later patches change one allowed variable. If
  it fails, later patches fix the classified failure layer first.
- Branch A local smoke/probe L0 uses `patch_0` to prove exactly one primary
  component can execute and feed measured output. If the helper is not executed,
  the next patch is wiring-only. If source evidence proves the local path cannot
  execute independently, record `Task correction` and shrink/report; do not
  promote the worker patch to a whole-kernel rewrite.
- Whole-kernel Gluon anchors are compile-risk tasks. `patch_0` proves launcher,
  layout factory, ABI, and minimum execution wiring before any performance
  tuning. `patch_1+` works through one failure layer at a time.
- Branch B whole-helper skeleton L0 is a whole-kernel anchor whose `patch_0`
  proves compile, wiring, parent-layout lineage, and measured-output feeding.
  Passing `patch_0` does not permit immediate MFMA, buffer, scheduler, epilogue,
  or performance tuning unless the next patch names one removable overhead.
- L1 trait-specific tasks start from a verified anchor and change only the
  named trait, such as one memory path, one layout conversion, or one matrix
  subpath. A failed L1 patch repairs that trait's failure layer before trying
  another trait.
- Gluon variant and Hybrid/mixed tasks preserve the safe anchor semantics.
  Passing patches may change one portable component or one dispatch decision;
  failing patches first restore anchor semantics or repair wrapper/dispatch
  wiring.
- Existing production AMD Gluon tasks preserve source contracts first. Passing
  patches may refine one in-dialect component; failing patches first repair the
  observed source, layout, API, artifact, or integration layer.

`failure_to_next_patch_map`:

- broadcast/layout failure -> fix parent layout, slice axis, expand direction,
  tensor rank, or host layout factory only.
- matrix/dot failure -> fix result layout, `DotOperandLayout`, `convert_layout`,
  target matrix op, accumulator dtype, or epilogue/store layout only.
- layout verifier / target arch failure -> fix host layout construction, target
  family, wave-size assumptions, `num_warps`, or launch attributes only.
- dtype/load/store failure -> fix `gl.full` dtype, casts, masks, generic
  `gl.load` / `gl.store`, buffer op preconditions, or store dtype only.
- reduction/accumulator failure -> fix accumulator layout, reduction order,
  loop-carried state, or identity values only.
- helper not executed / fallback success / wrong target -> fix wrapper,
  launch wiring, output feeding, or target-symbol association only.
- scope or forbidden-path failure -> revert the forbidden change before any
  further optimization.

If task failure-layer metadata is missing or incomplete, record a `Task
correction` in strategy notes and summary, choose the smallest matching layer
from this map, and continue within the existing task boundary. Do not introduce
new patch-evolution metadata fields for this correction.

Task consistency check:

- Before editing, compare the task body, task metadata, source code, and routed
  docs. Confirm that `Target component`, `Allowed change`,
  `minimum_executable_unit`, and `allowed_execution_path` describe the same
  scoped work.
- Classify the task as Branch A local smoke/probe or Branch B whole-helper
  skeleton using `10_search_policies.md::l0_scope_decision_before_emit`. Branch
  A stays within one primary component; Branch B starts from a whole-helper
  layout/compile skeleton. Do not silently switch branches inside a worker patch.
- If the source shows an unavoidable matrix, reduction, layout, or wrapper layer
  that is absent from `failure_layers` / `expected_failure_layers`, record a
  `Task correction` in strategy notes and summary, then use
  `failure_to_next_patch_map` to choose the next failure-fix patch.
- If the task would require widening beyond allowed scope, follow
  `scope_infeasible_policy` and report/shrink instead of silently expanding the
  patch.
- If `Plain competitor` names a different direction or target component, stop
  and report the mismatch. Do not use a different competitor to justify success.

Whole-kernel L0 comparison anchor:

- Use `whole_jit_kernel` only when the whole helper or kernel is the smallest
  executable Gluon unit. This is a comparison-boundary decision, not a reason to
  drop the same-direction plain evidence.
- When a whole helper is the L0 target, prefer a plain competitor whose
  `Optimization direction`, `Target component`, and measurement boundary describe
  the same helper or execution boundary.
- If the only available plain competitor is a narrow local cleanup, prefer
  shrinking the Gluon L0 to that local component or asking the planner for a
  matching whole-helper Base task.
- Gluon-specific mechanisms such as explicit layouts, `DotOperandLayout`, or
  buffer ops stay in `Gluon overlay reason`, `Performance hypothesis`, and
  `Allowed change`. They should explain how the Gluon implementation tests the
  same goal, not create a separate optimization direction.

Public API freeze:

- The harness-visible wrapper is part of the benchmark contract. Keep exported
  function names, module import paths, wrapper arguments, and return behavior
  stable.
- If a Gluon path needs host dispatch, add the dispatch inside the existing
  wrapper or behind an internal helper. Do not rename `decode_*`, `matmul_*`,
  `softmax_*`, or other public entrypoints that tests import.
- A patch that cannot wire Gluon without changing the public API should report
  the integration boundary and request a separate task, not mutate the harness
  contract.

## broadcast_heavy_whole_kernel_l0

Broadcast-heavy whole-kernel L0 is a compile-risk anchor, not a performance
candidate. Use it only when a smaller `index_map`, `mask_boundary`,
`load_store`, or `layout_broadcast` smoke path cannot execute and feed measured
output.

Before editing, write a layout-map skeleton:

```text
parent expression | parent layout | slice tensor | slice axis | expanded form | consumer
```

Rules:

- `patch_0` should establish host-created parent layouts, slice tensors, and
  launch wiring. It should not also introduce MFMA, buffer ops, scheduler
  changes, epilogue fusion, or wrapper rewrites unless the task explicitly
  marks a bundle.
- If the skeleton shows that one symbolic dimension is consumed by multiple
  parents, create separate tensors per parent. Do not reuse a single 1D index
  across `[H, R]`, `[R, N]`, `[H, N]`, `[H, C]`, or `[C, N]` contexts.
- A failure in RoPE, mask, or reduction layout before the target load/store path
  means the task was broader than the local component. The next patch should fix
  that parent-layout layer or report shrink/infeasible.
- If matrix/dot operands are unavoidable in `patch_0`, the task should include
  matrix/dot in expected failure layers and set `matrix_lowering_required`
  consistently with the scope.

## l0_scope_by_kernel_family

Use kernel families as routing hints, then use `atomic_component_lattice` to
pick the primary component.

| Family | Typical signals | Preferred L0 scope | Required failure layers to consider |
| --- | --- | --- | --- |
| matrix/GEMM/scaled_mm | `tl.dot`, scales, fp8/fp4, epilogue | matrix skeleton or scale-layout skeleton | matrix_operand, scale_dtype, epilogue_fusion, shape_dispatch |
| attention/decode | Q/K/V, paged KV, RoPE, masks, online softmax | index/mask/load smoke path or layout-map skeleton | index_map, mask_boundary, layout_broadcast, matrix_operand, reduction_accumulator, wrapper_integration |
| softmax/reduction/norm | row/block reduction, max/sum/rms, normalization | accumulator/reduction layout anchor | reduction_accumulator, mask_boundary, load_store, epilogue_fusion |
| topk/sampler/routing | compare/select, index update, partial reduction | compare/select state anchor | selection_update, reduction_accumulator, index_map, mask_boundary |
| elementwise/memory | activation, RoPE elementwise, cache copy, lora path | scoped index/load/store or dtype smoke path | index_map, mask_boundary, load_store, scale_dtype |
| scan/stateful | SSM, prefix, recurrent update | one state transition smoke path | state_update, reduction_accumulator, shape_dispatch |
| integration/dispatch | JIT/AOT/prebuilt, shape bucket, module wiring | launch/wiring or explicit host dispatch evidence | wrapper_integration, shape_dispatch |

Unknown family fallback:

- Do not emit `whole_jit_kernel` as the default Gluon L0.
- Prefer a Base/plain Triton task, or a smallest `index_map`, `load_store`, or
  `layout_broadcast` smoke task.
- Record the missing classification in task notes so later docs can be extended.

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
- its measured speedup against the true baseline and whether it has material
  per-shape regression.

The task prompt must include:

```text
Anchor patch: <task>/<patch or input_baseline>
Anchor speedup: <number>x
Anchor execution: true
Comparison target: anchor_patch
Allowed change: <one memory/matrix/layout/dispatch component>
Reject if: <conditions that invalidate this L1 patch>
```

If the anchor is below `0.5x`, has significant per-shape regression, or did not
execute AMD Gluon, do not add L1 memory/MFMA lowering. First shrink or diagnose
the anchor overhead.

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

## generalized_real_code_patterns

Use real Gluon code to extract decisions, not constants:

1. Bind the Gluon idea to the same optimization direction as a plain Triton
   competitor. Do not start from "use Gluon" as the optimization direction.
2. State the measurement boundary (`kernel_only`,
   `fair_make_inputs_run_kernel`, or `full_operator`) before interpreting a
   Gluon win or loss.
3. Identify the physical continuous memory dimension and assign the most useful
   lane/thread coverage there first.
4. Choose the target matrix family from the architecture and dtype path before
   choosing result and operand layouts.
5. Build one parent layout per logical 2D/3D expression, then derive
   `SliceLayout`, `DotOperandLayout`, shared, or descriptor layouts from that
   parent.
6. Keep shape-dependent layouts, `num_warps`, instruction shapes, and dispatch
   buckets in host-side code or explicit `constexpr` arguments.
7. Treat `convert_layout` as a paid operation unless the source or layout output
   proves it is a trivial reinterpretation.
8. Treat cross-CTA layouts as synchronization-sensitive. Upstream tutorials say
   layout-driven operations such as `convert_layout`, reductions, sums, and maxes
   can emit CGA barriers when they cross CTAs; do not move these into
   warp-specialized regions as a generic optimization.
9. Preserve operator-local JIT/AOT, artifact, and environment gates until the
   benchmark contract proves they are not part of the measured path.

Generalizable patterns:

- Elementwise/vector paths: start with explicit `BlockedLayout` plus generic
  `gl.load` / `gl.store`; move to AMD buffer ops only when memory-bound evidence
  or existing AMD structure justifies it.
- Attention/decode-style paths: preserve stride-rich host arguments, partition
  logic, masks, query/key/value logical shapes, and separate parent layouts
  before attempting MFMA or shared-memory staging.
- GEMM/FP8/FP4 paths: choose the matrix family, result layout, operand layouts,
  scale layout, accumulator dtype, and store dtype before adding shared-memory
  or launch/config tuning.
- RDNA/gfx1250 descriptor paths: get a plain WMMA or descriptor path correct
  before adding scaled WMMA, `tdm`, async, or cluster behavior.
- Split-K or block-scaled paths: derive scale layouts and output/reduction
  shapes from the chosen split regime. Do not reuse a single 2D scale layout
  when `SPLIT_K` introduces an extra logical rank.
- Config-driven GEMM paths: treat checked-in JSON configs, `NUM_KSPLIT`,
  `SPLITK_BLOCK_SIZE`, `GROUP_K`, `GROUP_N`, `num_warps`, `num_stages`, and
  `waves_per_eu` as a host-side configuration contract. The reusable pattern is
  shape/config dispatch plus same-ABI comparison, not the literal config values.
- Codegen-audited examples: source mapping checks for `wmma`, LDS loads,
  permlane swaps, or absence of `convert_layout` are useful evidence for
  validating a mature path. They are not mandatory for first L0 candidates.

Do not generalize:

- numeric partition formulas from one operator;
- `instr_shape`, `k_width`, `num_warps`, `waves_per_eu`, or JSON config choices
  without checking target, dtype, and shape regime;
- module namespace names such as `gl.amd.cdna3.*` as a complete architecture
  contract;
- CK or backend-only support notes into Gluon API syntax.
- a kernel-only win into a full-operator replacement when packing, cache layout,
  wrapper dispatch, or artifact lookup changes between paths.
- scheduler priority calls, backend assembly checks, or example-only static
  profiling helpers as portable Gluon requirements.

## real_operator_patterns

Real operator Gluon code shows production-style composition:

- `BlockedLayout`;
- `SliceLayout`;
- `AMDMFMALayout(version=CDNA_VERSION, ...)`;
- `DotOperandLayout`;
- `allocate_shared_memory`;
- shared-memory swizzle;
- `buffer_load` / `buffer_store`;
- explicit stride-rich host arguments.

This is more representative than a toy vector-add kernel, but still only gives
operator-local evidence.

Toy vector examples are useful for syntax but weak evidence for real
optimization decisions.

GEMM and FP8 paths on `gfx950` additionally use:

- `mfma_scaled`;
- `get_mfma_scale_layout`;
- architecture-conditioned K widths and instruction shapes;
- JSON-config or heuristic-selected launch configs instead of only online
  autotune.
- split-K partial outputs and separate reduce stages when `NUM_KSPLIT > 1`;
- scale-group parameters such as `GROUP_K` and `GROUP_N` derived from scale
  tensor shapes, not guessed from the main matrix tile alone.

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
| Attention-style operator | Gluon main kernel plus plain Triton helper stages, or an AOT wrapper path | CDNA-family path; operator-local guards decide exact architecture support |
| Low-precision GEMM operator | Gluon GEMM with checked-in or heuristic configs | Configs may be architecture-specific; do not infer support across families |
| Prebuilt-artifact operator | JIT Gluon or prebuilt/AOT artifact selected by version/env | Artifact shape can be package/config/env driven rather than generated locally |
| RDNA descriptor/WMMA operator | WMMA/TDM/descriptor-oriented path | Separate family from CDNA; not a drop-in replacement for CDNA attention/GEMM |

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

Composite high-coupling kernels:

Use this category when a single candidate combines several failure layers, for
example broadcast-heavy layout, matrix/dot lowering, reduction or accumulator
state, conditional/source-first logic, and wrapper/integration boundaries.

Generic failure layers:

1. broadcast/layout layer;
2. memory/load-store layer;
3. matrix/dot lowering layer;
4. reduction/accumulator layer;
5. conditional/source-first layer;
6. wrapper/integration layer.

Round-1 strategy:

- Prefer a micro-anchor from one failure layer before a compile-risk whole-kernel
  task.
- If a whole-kernel task is truly the minimum executable unit, mark it as a
  compile-risk anchor and make `patch_0` a compile goal, not a performance goal.
- `patch_1+` should declare and change one failure layer at a time. Record the
  declared layer and observed failure so later rounds can shrink or escalate
  without guessing.

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

In real low-precision GEMM code, config selection may come from heuristics or
checked-in JSON, not only online autotune.

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

## benchmark_boundary_and_integration_costs

End-to-end operator wrappers are often pipelines rather than a single hot
kernel. Before choosing a Gluon rewrite, classify the measured boundary:

- `kernel_only`: payloads are already prepared and the benchmark measures the
  hot kernel body. Use this to judge whether a Gluon kernel implementation is
  locally useful.
- `fair_make_inputs_run_kernel`: input preparation plus kernel launch is in
  scope. Packing, cache layout conversion, temporary allocation, and wrapper
  dispatch may dominate.
- `full_operator`: the whole operator path is in scope. Attribute wins to the
  component that changed, not automatically to the kernel dialect.

Same-ABI rule:

- Compare plain Triton and AMD Gluon under the same host ABI whenever possible:
  same input tensor layout, cache format, wrapper signature, grid semantics, and
  correctness oracle.
- If the Gluon path requires a different packed cache or prebuilt artifact,
  record that as an integration change and benchmark it under the same
  measurement boundary as the plain Triton path.
- If `kernel_only` Gluon wins but `fair_make_inputs_run_kernel` loses, the next
  task should target pack/unpack, cache ABI, wrapper dispatch, or artifact lookup
  cost. Do not keep tuning the Gluon kernel body until the integration cost is
  accounted for.
- If `fair_make_inputs_run_kernel` wins but `kernel_only` loses, report the win
  as integration/ABI evidence. It is not proof that the Gluon kernel body is
  faster.

Hot-path task strategy:

- For pure hot-path kernels, Gluon must reduce real measured work: fewer memory
  transactions, a better matrix instruction path, less masking/index overhead,
  or shape-specialized dispatch that beats the plain competitor.
- For pipeline kernels, keep the Gluon scope to one stage or subpath unless the
  task explicitly allows a bundled end-to-end rewrite.
- Hybrid dispatch must preserve the plain Triton path for shapes or subpaths
  where it wins. A kernel-only Gluon win cannot replace a fair/full-operator
  path unless the same boundary also preserves no-regression against the safe
  anchor.

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
- Gluon must beat the safe plain Triton anchor to count as `Gluon-positive`.
  Faster-than-original but slower-than-safe-anchor is evidence, not a win.
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
