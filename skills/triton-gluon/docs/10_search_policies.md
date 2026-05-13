# Triton-Gluon Search Policies

Planner-facing policy doc. Read this file when planning tasks, reviewing prior
rounds, or deciding how a Triton optimization direction should be tried as plain
Triton, AMD Gluon, paired comparison, shared transplant, or hybrid dispatch.
Implementation details live in worker-routed docs (`20_component_traits.md`,
`30_architecture_notes.md`, `50_api_reference.md`, and `60_real_patterns.md`).

## Internal Index

- `### Search policy: optimization_direction_metadata_sets`
- `### Search policy: optimization_direction_dialect_overlay`
- `### Search policy: existing_amd_gluon_refinement_policy`
- `### Search policy: overlay_direction_vs_mechanism`
- `### Search policy: atomic_component_lattice`
- `### Search policy: l0_scope_decision_before_emit`
- `### Search policy: overlay_priority_routing`
- `### Search policy: l0_scope_classification`
- `### Search policy: evidence_anchored_composition`
- `### Search policy: measurement_boundary_policy`
- `### Search policy: dialect_contract_metadata`
- `gluon_doc_gate_metadata`
- `trait_policy_separation`
- `round_progression`
- `result_attribution`

### Search policy: optimization_direction_metadata_sets

Plan by optimization direction first: split/decomposition, fusion,
memory/layout cleanup, shape specialization, persistent/launch amortization, or
another Triton family from the main planner. Legacy Base/Shared/Extension labels
are compatibility metadata for audit, attribution, scheduling, and selector
contracts; do not use them as task ideas.

Plain Triton coverage is mandatory for every high-value direction. Gluon is an
additional implementation option, not a reason to remove the plain Triton
competitor.
Base tasks are real no-regression performance candidates. Do not create a small
plain Triton task only to satisfy a Gluon `Plain competitor` field: it needs a
standalone performance mechanism at that scope, and its first patch should stay
small enough to be judged against the baseline.
Round 1 L0 overlays must bind to a concrete same-batch plain task through
`Plain competitor`, not merely to a family name. The `Plain competitor` task's
`Base family` must match the overlay's `Source Base family`. It must be a Base
Set task with `required_output_dialect=plain_triton`; do not point
`Plain competitor` at Shared, paired-comparison, shared-transplant, Gluon, mixed,
or hybrid tasks.

Required prompt fields by layer:

```text
Plain Triton competitor:
Base family: <family_id>
Optimization direction: <main Triton strategy>

Paired mapping:
Shared source family: <base_family_id>
Optimization direction: <main Triton strategy>
Implementation layer: plain_triton variant | amd_gluon variant | paired comparison | shared_transplant
Comparison target: true_baseline | safe_anchor
Allowed change: <one portable component>
Reject if: <conditions that invalidate the patch>

AMD Gluon overlay:
Extension layer: L0 | L1 | Hybrid
Optimization direction: <main Triton strategy>
Source Base family: <family_id>
Plain competitor: <plain Triton task label in this batch>
Gluon overlay reason: explicit_layout | buffer_path | matrix_lowering | shape_bucket | dialect_specific_memory | local_subpath_win
Overlay priority: Prefer | high-confidence Consider
Implementation layer: amd_gluon overlay | paired comparison | mixed/hybrid dispatch
Performance hypothesis: <why this scoped overlay might beat the safe plain/base path>
Measurement boundary: kernel_only | fair_make_inputs_run_kernel | full_operator
Comparison target: true_baseline | safe_anchor | anchor_patch
Allowed change: <one component or one dispatch decision>
Target symbol: <required when scoped to a specific stage or helper>
Target component: <required when scoped to named local expressions or data paths>
Reject if: <conditions that invalidate the patch>
```

### Search policy: optimization_direction_dialect_overlay

Use the standard Triton priority order to choose what to optimize first.
Then decide which implementation layer should try that direction:

- `plain_triton`: the default implementation layer and required no-regression
  competitor.
- `amd_gluon`: use only when the same direction has a concrete Gluon mechanism:
  explicit layout control, AMD buffer path, matrix/MFMA lowering, shape-bucketed
  dispatch, or prior local sub-operation evidence.
- `paired comparison`: use when the same strategy can be implemented in both
  dialects and the benchmark should decide which layer wins.
- `mixed` / `hybrid`: use only after evidence shows different winners by shape
  or sub-operation.

Examples:

- `split-K` may have a plain Triton split/decomposition task and, only with a
  real matrix/layout reason, an AMD Gluon split/decomposition variant.
- `memory/layout cleanup` may have a plain Triton pointer/mask simplification
  and an AMD Gluon explicit-layout or buffer-load overlay.
- `persistent / launch amortization` is usually plain Triton/HIP first. Add
  Gluon only when kernel-only or sub-operation evidence shows that explicit
  layout removes more overhead than it adds.

Do not emit a generic "rewrite to Gluon" task. A Gluon task must name the
optimization direction it implements and the single allowed component it changes
in the next patch.

### Search policy: existing_amd_gluon_refinement_policy

Use this policy only when `input_dialect=amd_gluon` and
`source_origin=existing_amd_gluon_operator`. Missing or unknown origin stays on
the generated-overlay path.

Plain Triton family names remain useful, but they become performance taxonomy
labels rather than mandatory Base tasks. For production AMD Gluon input, plan by:

```text
kernel_family_signal
-> atomic_component_graph
-> coupling / execution-boundary decision
-> task_type
-> doc_profile + failure_layers
```

Search width is a planner recommendation, not a new metadata schema. Use source
evidence, shape profile, prior Gluon signal, and component coupling to decide
whether to emit one narrow refinement or up to three distinct refinements, but
do not add persistent fields such as `gluon_depth`, `complexity_level`, or
`recommended_slots`. Keep the task contract on existing fields: `Task type`,
`Kernel family signal`, `Target component`, `Allowed change`,
`required_output_dialect`, `gluon_doc_profile`, and concrete target symbols.

Kernel family signals only route search focus:

- `attention_decode_kv_cache`: paged KV, QK/PV, softmax, partition reduce, shape
  or persistent dispatch.
- `logits_or_small_reduction`: logits, tiny stages, row/block reductions, low
  latency paths.
- `gemm_or_scaled_dot`: GEMM, scaled dot, MFMA/WMMA, accumulator and epilogue.
- `memory_or_elementwise`: load/store, masks, casts, dtype, output store.
- `wrapper_shape_dispatch`: public ABI, shape bucket, partition, PS/persistent
  path, fallback selection.
- `descriptor_or_source_first`: descriptor, preshuffle/unshuffle, JIT/AOT,
  artifact or target guard.

Atomic components produce dispatchable tasks:

- `wrapper_shape_dispatch`
- `layout_parent_slice`
- `load_store_buffer`
- `matrix_operand_mfma`
- `reduction_accumulator`
- `state_update_softmax`
- `epilogue_output_store`
- `scheduler_launch_runtime`
- `source_contract_integration`

Task types:

```text
amd_gluon_in_dialect_refine
amd_gluon_layout_or_matrix_refine
amd_gluon_shape_dispatch_refine
plain_subkernel_refine
shared_or_plain_comparison
defer_composition
```

Existing AMD Gluon refinement prompt contract:

```text
source_origin: existing_amd_gluon_operator
Task type: amd_gluon_in_dialect_refine | amd_gluon_layout_or_matrix_refine | amd_gluon_shape_dispatch_refine
Implementation layer: amd_gluon in-dialect refinement
required_output_dialect: amd_gluon
gluon_tl_policy: production_source_allowed
layout_construction_policy: source_preserve
Kernel family signal: <family signal>
Target component: <one atomic component>
Allowed change: <one component-local change>
Failure layers: <route/layout/matrix/reduction/memory/wrapper as applicable>
Measurement boundary: kernel_only | fair_make_inputs_run_kernel | full_operator
Comparison target: true_baseline | safe_anchor
Reject if: correctness fails or any benchmark shape regresses
```

`Target component` is a routing label, not a concrete diff target. Required patch
targets must come from `Target symbol`, backticked local names, or scoped groups
such as `components(...)`, `expressions(...)`, `loads(...)`, or `stores(...)`.
Do not let abstract atomic components such as `state_update_softmax` or
`load_store_buffer` become `required_patch_target_symbols`.

Coupling rule:

- Low coupling single-component changes may dispatch in round 1.
- Medium coupling changes must state measurement boundary, ABI risk, and rollback
  condition.
- High coupling combinations that change wrapper dispatch, matrix layout,
  softmax/reduction state, partition policy, or scheduler policy together should
  become `defer_composition` until prior evidence exists.
- If the target symbol is still a plain `@triton.jit` subkernel, use
  `plain_subkernel_refine` and `required_output_dialect=plain_triton`.
- For round 1, give each dispatchable refinement exactly one `Allowed change`.
  If the prompt lists softmax state, MFMA accumulator, value-load reordering,
  quant scale, scheduler, and shape policy together, it is a composition note,
  not a dispatchable refinement.

### Search policy: overlay_direction_vs_mechanism

Use `Optimization direction` for the shared performance or algorithmic goal that
both the plain Triton competitor and the Gluon overlay are trying to test. Use
Gluon-specific fields for the implementation mechanism.

Preferred split:

```text
Optimization direction: <shared performance goal>
Gluon overlay reason: explicit_layout | buffer_path | matrix_lowering | shape_bucket | ...
Implementation layer: amd_gluon overlay
Performance hypothesis: <why the Gluon mechanism might help this same goal>
Allowed change: <same target component or dispatch decision>
```

Examples:

- Prefer `Optimization direction: reduce K_Buffer memory transactions` with
  `Gluon overlay reason: explicit_layout for the same K_Buffer load path`.
- Prefer `Optimization direction: improve the stage1 QK data path` with
  `Gluon overlay reason: DotOperandLayout / matrix lowering for the same dot
  component`.
- Avoid making implementation mechanisms such as "use explicit layout",
  "convert to @gluon.jit", "use DotOperandLayout", or "try buffer ops" the
  optimization direction by themselves. Those can be good overlay reasons when
  attached to a shared direction and a comparable target component.
- A whole-kernel Gluon translation mechanism is still not a same-direction
  overlay unless it implements the plain competitor's core mechanism. If the
  plain competitor is about parallelization, fusion, a memory path, scheduling,
  or an algorithmic rewrite, a Gluon patch that only changes syntax, layout
  explicitness, or matrix dialect is execution evidence, not same-direction
  performance evidence.

For high-coupling L0 work, the same rule still applies: if the Gluon path needs
`whole_jit_kernel`, the comparison anchor should describe a comparable helper or
execution boundary. If the only available Base task is a narrow local cleanup,
prefer shrinking the Gluon target to that local component or generating a
matching Base task before emitting the overlay.

Same-direction L0 binding is strict:

- `Plain competitor`, `Optimization direction`, `Allowed change`, `Target
  symbol`, and `Target component` must name the same stage, helper, component,
  or execution boundary.
- A Base task for stage1 register pressure is not a valid competitor for a Gluon
  L0 that edits a stage2 reduction, even when both belong to the same operator
  family.
- A broad whole-kernel Gluon anchor must still name the same helper/execution
  boundary and the same optimization mechanism as its plain competitor. If it
  only proves that a Gluon whole-kernel path can execute, keep it as execution
  evidence and do not promote it to L1/MFMA/buffer follow-up.
- If the planner cannot produce a same-component competitor, it should drop the
  overlay, emit diagnostic/no-dispatch evidence, or generate a matching Base
  competitor for that boundary.
- `force_l0_anchor` only relaxes overlay priority. It does not relax
  same-direction, same-component, same-helper, or same-boundary requirements.

### Search policy: atomic_component_lattice

Kernel family names are routing hints, not a complete coverage mechanism. Plan
Gluon work by decomposing the target into atomic components, then choose one
primary component for `patch_0`. Other components should be listed as
secondary components or blockers in the task body / notes, not turned into new
hard metadata fields.

Atomic components:

| Component | Covers | Typical first Gluon scope |
| --- | --- | --- |
| `index_map` | program ids, `tl.arange`, offsets, strides, pointer arithmetic | one index expression or address-family smoke path |
| `mask_boundary` | tail masks, causal/window masks, page/block boundaries | one mask parent layout and its guarded load/store |
| `load_store` | global loads/stores, coalescing, cache path, store dtype | one load/store value layout with generic `gl.load` / `gl.store` first |
| `layout_broadcast` | `SliceLayout`, parent-layout map, `[:, None]`, `[None, :]` | one parent layout and its slice/broadcast expression |
| `matrix_operand` | `tl.dot`, MFMA/WMMA, operand/result layout | matrix skeleton with result layout and operand layouts |
| `scale_dtype` | fp8/fp4 scales, quant/dequant, packed dtype, casts | one scale-layout or dtype conversion boundary |
| `reduction_accumulator` | sum/max/softmax/rms/topk accumulators, loop-carried state | accumulator layout anchor for one reduction axis |
| `selection_update` | topk/sampler compare-select-index update | compare/select state update without also changing storage ABI |
| `state_update` | scan, SSM, recurrent state, prefix update | one state transition or scan step smoke path |
| `epilogue_fusion` | activation, bias, norm, quant, store epilogue | one epilogue expression after anchor correctness |
| `shape_dispatch` | multi-shape buckets, constexpr layout, host dispatch | explicit host-side bucket with no-regression fallback |
| `wrapper_integration` | JIT/AOT/prebuilt modules, import wiring, fallback | launch/wiring evidence, not kernel-body tuning |
| `scheduler_launch` | persistent, work queue, swizzle, num_warps/stages | later-stage refinement after a safe anchor |

Use this lattice as a composition model:

```text
kernel family -> component signals -> primary_component -> L0 scope
secondary_components -> expected_failure_layers / blocked_by
```

Guidance:

- `patch_0` should have one primary component unless `bundle_allowed=true`.
- If the primary component cannot execute without converting unrelated
  components, shrink the task or emit a skeleton task that names those blockers.
- Unknown kernels should not default to `whole_jit_kernel`. Prefer Base/plain
  work, or the smallest `index_map`, `load_store`, or `layout_broadcast` smoke
  path with a note that classification is incomplete.
- Use family labels such as GEMM, attention, softmax, topk, or elementwise only
  to choose likely component groups. The component lattice decides the actual
  Gluon scope.

### Search policy: l0_scope_decision_before_emit

Before emitting a Gluon L0 task, connect the family/component route to exactly
one executable branch:

```text
kernel family -> candidate atomic components -> one primary_component
-> executable boundary -> Branch A or Branch B
```

Branch A: local single-component smoke/probe.

- Use when the primary component is low-coupling, or when the planner cannot
  prove that a whole helper is required.
- Keep `Target component` and `Allowed change` local, such as one index/mask,
  one load/store path, one parent layout, or one scale/dtype boundary.
- Use `minimum_executable_unit: inline_scoped_helper` and
  `allowed_execution_path: inline_scoped_helper`.
- Put matrix, reduction, state, wrapper, or integration dependencies in
  secondary/blocker notes or failure layers, not as `patch_0` targets.
- Do not mix a local target with `whole_jit_kernel`.

Branch B: whole-helper layout skeleton.

- Use only when source evidence shows the local target cannot execute and feed
  measured output without the whole helper/kernel.
- Retarget the task to the whole helper/stage. Do not keep wording such as
  "one load" or "one local path" as the target.
- If `minimum_executable_unit=whole_jit_kernel`, the task label,
  `Optimization direction`, `Allowed change`, and `Target component` should all
  say whole helper/stage/kernel anchor. Local phrases such as `index_anchor`,
  `mask_anchor`, or `load_store_anchor` may appear only as target subpath
  evidence, not as the patch scope name.
- Use `minimum_executable_unit: whole_jit_kernel` and
  `allowed_execution_path: whole_jit_kernel`.
- Include `whole_kernel_required_reason`, `expected_failure_layers`,
  `first_patch_compile_goal`, `do_not_optimize_before_compile: true`, and a
  `matrix_lowering_required: true|false` value consistent with the skeleton.
- Treat `patch_0` as compile/wiring/layout evidence, not performance tuning.
- For standalone micro-kernels or tiny whole-helper anchors, the default outcome
  is compile/execution evidence. Do not upgrade the task to a performance
  candidate unless the same boundary has a concrete removable overhead and a
  same-boundary plain competitor.

Unknown or ambiguous boundary:

- Default to Branch A, or spend the slot on Base/plain Triton.
- Do not default uncertainty to `whole_jit_kernel`.
- If Branch A later proves impossible, the worker records `Task correction` and
  shrink/report evidence; a later planner round may emit a Branch B task.

### Search policy: overlay_priority_routing

This policy decides whether a concrete Triton optimization direction should get
an AMD Gluon overlay task, and at what priority. It does not choose the
optimization direction itself.

Read order for planner priority:

1. `00_always_read.md` task routing table.
2. `10_search_policies.md`,
   `optimization_direction_dialect_overlay` and this section.
3. `20_component_traits.md` for the detected component traits.
4. `60_real_patterns.md` when the task is end-to-end, source-first, low-latency,
   architecture-guarded, or has benchmark/measurement-boundary risk.
5. `50_api_reference.md` only when assigning a concrete API-sensitive task such
   as buffer ops, MFMA/WMMA, descriptor, or failure triage.

Overlay priority buckets:

- Prefer: existing AMD/NV Gluon input; explicit layout is already part of the
  algorithm; one local sub-operation is layout-bound or matrix-lowering-bound;
  benchmark/profiling shows the target subpath is hot enough that layout
  control could pay for its overhead.
- Consider: memory/layout cleanup with a narrow load/store or index/mask
  subpath; a paired comparison can answer whether plain Triton or AMD Gluon wins;
  prior correctness-passing Gluon evidence isolates one overhead source to
  remove.
- Deprioritize: low-latency or tiny stages where launch/layout overhead is
  likely to dominate; full-operator rewrites; persistent scheduling,
  work-stealing, atomics, async/shared-memory, or descriptor work before a
  simpler Gluon path has passed correctness.
- Do not generate: no named plain Triton competitor, no concrete Gluon overlay
  reason, no same-direction optimization target, only compile-valid value, or a
  prior same-scope Gluon attempt already regressed without a named removable
  overhead.

Round 1 for plain Triton inputs may include at most one L0 overlay, and only in
the Prefer or high-confidence Consider buckets. Otherwise spend the task on the
plain Triton direction. Later rounds may create L1, `gluon_variant`, or
`hybrid_dispatch` only from verified Gluon execution evidence and safe-anchor
comparison.

Run-mode semantics:

- `auto`: keep the rule above. Ordinary `Consider`, high-coupling, or
  whole-kernel compile-risk L0 overlays may be dropped while Base/plain Triton
  tasks remain valid.
- `force_l0_anchor`: for coverage experiments only, allow at most one ordinary
  `Consider` L0 to run as a real AMD Gluon compile/execution anchor. Keep the
  same-direction plain Triton competitor and treat the result as evidence, not a
  guaranteed performance candidate.
- `require_viable_gluon`: do not replace a missing viable Gluon task with a
  Base-only plan. If no dispatchable Prefer/high-confidence Consider overlay
  exists, report no viable Gluon task for the round.

### Search policy: l0_scope_classification

Before emitting a Round-1 L0 task, or a later task that builds on L0 evidence,
classify whether the proposed target has a self-contained execution path:

```text
l0_scope_classification: low_coupling | high_coupling | infeasible
l0_coupling_reasons: <why this subpath is or is not self-contained>
minimum_executable_unit: inline_scoped_helper | separate_gluon_kernel | whole_jit_kernel | infeasible
allowed_execution_path: inline_scoped_helper | separate_gluon_kernel | whole_jit_kernel
scope_infeasible_policy: do_not_emit | shrink_or_report | separate_kernel_if_allowed
```

Use `inline_scoped_helper` only for low-coupling subpaths such as a scalar/1D
helper, one index/mask expression, or one load/store smoke path that can execute
without changing the surrounding algorithm. Do not use it for loop-carried
state, online reductions, dot/matrix paths, cross-stage ABI changes, wrapper
reroutes, multi-parent layout rewrites, or full helper/kernel rewrites.

If the target is high-coupling, shrink it to a lower-coupling subpath, mark it
`infeasible`, or use `whole_jit_kernel` only when the whole helper/kernel is
explicitly the minimum executable unit and `whole_kernel_required_reason` is
provided. A task that declares `inline_scoped_helper` but requires whole-kernel
conversion is invalid.

Treat a whole-kernel L0 candidate as `too_high_coupling_for_round1_l0` when
several complexity signals are present at once: loop-carried or cross-loop
state, multiple interdependent compute/memory subpaths, indirect or dynamic
shape-dependent memory routing, mixed mask/broadcast/layout conversion and
reduction state, or wrapper routes whose measured output path needs several
steps to prove. In `auto`, drop or diagnose these instead of dispatching a
required Gluon rewrite. In `force_l0_anchor`, prefer a diagnostic route map or
layout/failure-layer map unless the task clearly states a high-risk
compile/execution anchor.

High-coupling whole-kernel contract:

- `l0_scope_classification=high_coupling` plus
  `minimum_executable_unit=whole_jit_kernel` must retarget the task as a
  whole-kernel or whole-helper execution anchor.
- Do not keep local anchor labels such as `load_store_anchor`, `index_anchor`,
  or `mask_anchor` as the task scope when the patch must translate the whole
  helper to execute.
- The first patch proves execution path, launcher/ABI, and measured-output
  feeding. Matrix/MFMA, buffer, scheduler, or epilogue improvements are separate
  follow-up changes unless explicitly bundled.

Round-1 L0 overlays bind to the same component and same optimization direction
as `Plain competitor`, not merely to the same broad `Source Base family`. The
plain task may be a broad Base task, but it is only a valid local L0 anchor when
it exposes the same scoped `Target component` or `Allowed change` with a
credible plain Triton performance mechanism. If the only possible anchor would
be cleanup-only or broader than its own hypothesis justifies, omit the Gluon L0
overlay and spend the slot on Base/plain Triton work.

The execution-boundary fields above are required for Round-1 L0 AMD Gluon
overlays. Other routing fields such as `task_signals`,
`routed_doc_reasons`, `kernel_family_signal`, and `failure_layers` help route
worker docs, but they should be inferred or warned on when possible rather than
turning otherwise useful task batches into planner failures.

Required target path proof:

- `required_patch_target_symbols` must contain executable symbols: actual
  functions, helpers, kernel symbols, wrapper branches, or callable dispatch
  targets.
- Do not create required symbols by splitting prose from `Target component` or
  `Allowed change` into words such as `tl`, `load`, `of`, or pointer names that
  are not callable targets.
- Complex or wrapper-heavy tasks should include a `Required execution route`
  instead of relying on ambiguous target fragments:

```text
Required execution route:
- Benchmark case / shape:
- Public wrapper:
- Branch condition:
- Plain called kernel symbol:
- Expected Gluon symbol:
- Required target load/store or reduction path:
- Output feeding path:
- Proof after patch:
```

If this route cannot be stated for a high-coupling L0, use diagnostic/no-dispatch
or keep the slot for Base/plain Triton rather than asking a worker to guess the
target path.

Round progression:

- Prior Gluon compile failed, did not execute, or hit scope escalation: do not
  emit L1, `gluon_variant`, or `hybrid_dispatch`; shrink/retry L0 or spend the
  slot on Base/plain Triton.
- Prior Gluon passed correctness but was slower: refine only the recorded
  overhead source, such as layout conversion, buffer path, or one matrix operand
  layout.
- Prior Gluon executed and passed correctness but regressed on every shape:
  classify it as `Gluon-slower` / `not_viable_for_l1=true` unless the next task
  names one concrete removable overhead. Do not emit automatic MFMA, buffer,
  scheduler, or same-scope whole-kernel follow-up from that evidence alone.
- Prior Gluon wins for a shape or sub-operation: `gluon_variant` or
  `hybrid_dispatch` may be emitted, but must cite the safe anchor, the winning
  shape/sub-operation evidence, and `Comparison target: safe_anchor` or
  `Comparison target: anchor_patch`.

### Search policy: evidence_anchored_composition

This is a search policy, not a component trait. It says how to compose evidence
from previous rounds.

When prior-round evidence exists:

1. Identify the safe anchor:
   - prefer the best verified plain Triton/base-metadata patch;
   - if no Base patch is usable, use the best verified non-regressing patch;
   - if no verified patch exists, use the original baseline.
2. Identify portable components from Shared / Extension evidence:
   - `algorithm_decomposition`
   - `tiling_or_blocking`
   - `memory_access_policy`
   - `layout_or_indexing`
   - `matrix_lowering`
   - `mask_or_boundary_simplification`
   - `accumulator_representation`
   - `launch_or_dispatch_policy`
   - `scheduler_or_persistent_policy`
   - `dtype_or_precision_policy`
3. Generate composition tasks:
   - `base_refine`: continue optimizing the safe anchor.
   - `shared_transplant`: preserve safe-anchor semantics and transplant one
     portable component. Output may remain plain Triton.
   - `gluon_variant`: re-express the safe-anchor algorithm in AMD Gluon only
     when the useful component requires explicit layout, AMD memory path, or
     matrix lowering.
   - `hybrid_dispatch`: dispatch between Base and Gluon only when per-shape or
     sub-operation evidence shows different winners.

Composition tasks must include:

```text
Composition type: base_refine | shared_transplant | gluon_variant | hybrid_dispatch
Safe anchor: <task>/<patch or original_baseline>
Source component: <component_type> from <task>/<patch or none>
Comparison target: safe_anchor
Allowed change: <one component or one dispatch decision>
Reject if: <conditions that invalidate the patch>
```

Composition candidates must compare against the safe anchor, not only the
original baseline. A patch that is faster than the original baseline but slower
than the safe anchor is not a valid composition win.

L2 / composition constraints:

- Change at most one component unless the task explicitly says
  `bundle_allowed=true`.
- Preserve patch lineage for later rounds: composition evidence is strongest
  when each patch changes one component and records expected versus observed
  effect. A bundled patch that changes layout, memory, matrix, and launch tuning
  at once is weak evidence even if it passes correctness.
- `base_refine` may refine the safe plain/base anchor, but must not add a Gluon
  rewrite.
- `shared_transplant` may transplant one portable component into the safe
  anchor. It must preserve launcher ABI and safe-anchor algorithm unless the
  allowed change says otherwise.
- `gluon_variant` / `gluon_variant_from_anchor` must re-express the safe-anchor
  algorithm in AMD Gluon. It must preserve anchor semantics before adding
  memory, matrix, scheduler, or dispatch changes.
- `hybrid_dispatch` / `hybrid_dispatch_from_evidence` must add visible
  host-side shape/feature dispatch around verified Base and Gluon candidates.
  It must keep the plain Triton/base path for shapes or sub-operations where it wins.
- Correctness-passing but slower Gluon evidence is neutral/slower evidence. It
  can inform layouts or source routing, but it must not trigger
  `gluon_variant` or `hybrid_dispatch` unless a later result proves a shape or
  sub-operation where Gluon beats the safe anchor.
- Reject a composition if it changes multiple mutually exclusive components,
  changes launcher/constexpr contract without saying so, drops the safe anchor,
  or regresses any benchmark shape.

### Search policy: measurement_boundary_policy

For end-to-end or wrapper-heavy kernels, a task must state the benchmark
boundary before editing:

```text
Measurement boundary: kernel_only | fair_make_inputs_run_kernel | full_operator
Integration cost in scope: yes | no
Same ABI comparison: required
```

Rules:

- `kernel_only`: evaluates the hot kernel body with inputs already prepared.
  Kernel-only wins do not automatically justify fair/full-operator replacement.
- `fair_make_inputs_run_kernel`: includes input preparation plus kernel launch.
  Use this when pack/unpack, cache format conversion, or host dispatch is part of
  the product cost.
- `full_operator`: includes the entire operator path. Attribute wins carefully;
  wrapper, packing, and artifact lookup may dominate the measured result.
- Compare plain Triton and AMD Gluon under the same ABI. If an optimized path
  changes cache layout, packing, or input preparation, report that as an
  integration change rather than a pure kernel-body win.
- If Gluon wins `kernel_only` but loses `fair_make_inputs_run_kernel`, the next
  task should target integration cost or same-ABI pairing, not more Gluon kernel
  tuning.
- If `fair_make_inputs_run_kernel` wins while `kernel_only` loses, attribute the
  win to integration/ABI effects unless the kernel-only evidence is repaired.

## trait_policy_separation

Planner traits are internal signals. They are not user-facing requirements and
should not become generic task labels by themselves.

Use this separation:

- Component traits tell the worker what code direction to modify:
  `memory_access_policy`, `tiling_or_blocking`,
  `mask_or_boundary_simplification`, `accumulator_representation`,
  `layout_or_indexing`, `matrix_lowering`, `launch_or_dispatch_policy`,
  `scheduler_or_persistent_policy`, and `dtype_or_precision_policy`.
- Search policies tell the planner how to allocate and combine tasks:
  `optimization_direction_metadata_sets`,
  `optimization_direction_dialect_overlay`, `measurement_boundary_policy`,
  `evidence_anchored_composition`, and `dialect_contract_metadata`.

Do not treat `evidence_anchored_composition` as a component trait. It chooses a
safe anchor, identifies portable components, and enforces comparison target.
The implementation detail still comes from component traits.

### Search policy: dialect_contract_metadata

This metadata contract is Triton-family only. Do not apply it to HIP, CK, ASM,
FlyDSL, PyTorch-to-FlyDSL, or torch2hip tasks.

```yaml
required_output_dialect: plain_triton | amd_gluon | mixed | any
search_set: base | shared | extension  # optional compatibility bucket
```

`required_output_dialect`, `Implementation layer`, and `Extension layer` are
the output contract source of truth. `search_set` is optional compatibility
metadata; do not choose task ideas by `search_set` first.

Use `required_output_dialect=plain_triton` for plain competitors,
`required_output_dialect=any` for transplant/paired tasks that may remain plain
Triton, `required_output_dialect=amd_gluon` for required Gluon execution, and
`required_output_dialect=mixed` for explicit host-side dispatch. Required
Gluon tasks must attempt a real AMD Gluon patch and cannot silently succeed as
plain Triton fallback.

`mixed` is for explicit host-side dispatch between already validated plain
Triton and AMD Gluon candidates. It must keep a visible dispatch condition and
must compare every selected path against the safe anchor. Do not mark a generic
Extension L0 viability task as `mixed`; L0 is `required_output_dialect =
amd_gluon`.

## gluon_doc_gate_metadata

Planner-generated Gluon tasks should write documentation-gate metadata. This is
the source of truth for worker `save_and_test` gating; heuristic inference is
only for old or hand-written tasks.

```yaml
source_origin: generated_overlay | existing_amd_gluon_operator | nv_gluon_translation | unknown
gluon_tl_policy: strict_generated | preserve_existing_allowed | production_source_allowed
layout_construction_policy: host_preferred | constexpr_in_kernel_allowed | source_preserve
gluon_doc_profile: extension_l0_minimal | nv_to_amd_translation | memory_lowering | matrix_lowering | shape_bucketed_dispatch | jit_aot_sensitive | shared_transplant | gluon_variant_from_anchor | hybrid_dispatch | hybrid_dispatch_from_evidence | base_or_shared_gluon
required_gluon_docs:
  - gluon_skill_path
  - gluon_always_read_path
  - gluon_search_policies_path
```

Deterministic profile map:

| `gluon_doc_profile` | Required docs beyond mandatory skill/00/10 | Planner performance boundary |
| --- | --- | --- |
| `extension_l0_minimal` | `gluon_component_traits_path`, `gluon_api_reference_path` | Smallest executed Gluon anchor. Worth trying only with a concrete layout/API reason; stop after slower correctness on low-latency paths. |
| `nv_to_amd_translation` | `gluon_component_traits_path`, `gluon_architecture_notes_path`, `gluon_real_patterns_path`, `gluon_api_reference_path` | Translation, not renaming. Source-first for NVIDIA layout/async/TMA/WGMMA assumptions before AMD tuning. |
| `memory_lowering` | `gluon_component_traits_path`, `gluon_api_reference_path` | Use when a scoped load/store/cache path is hot. Start from generic `gl.load/store`; escalate to AMD buffer ops only when dtype/layout preconditions and performance mechanism are named. |
| `matrix_lowering` | `gluon_component_traits_path`, `gluon_architecture_notes_path`, `gluon_api_reference_path` | Requires real hot dot/scaled-dot path, result layout, operand layouts, target op, and conversion cost hypothesis. Downgrade to L0 if anchor/layout evidence is missing. |
| `shape_bucketed_dispatch` | `gluon_component_traits_path`, `gluon_real_patterns_path` | Use only when shapes have distinct regimes. Require visible host dispatch and per-shape no-regression. |
| `jit_aot_sensitive` | `gluon_architecture_notes_path`, `gluon_api_reference_path` | Use when JIT/AOT/prebuilt/signature/scratch/version details are part of execution. Preserve fallback gates unless benchmark contract proves otherwise. |
| `shared_transplant` | `gluon_component_traits_path`, `gluon_real_patterns_path` | Transplant one portable component into the safe anchor. Output may stay plain Triton; compare to safe anchor. |
| `gluon_variant_from_anchor` | `gluon_component_traits_path`, `gluon_api_reference_path`, `gluon_real_patterns_path` | Re-express the safe-anchor algorithm in AMD Gluon only when explicit layout, memory, or matrix mechanism justifies it. |
| `hybrid_dispatch` | `gluon_real_patterns_path` | Use visible host-side dispatch only after Base/Gluon candidates both exist and a dispatch condition is auditable. |
| `hybrid_dispatch_from_evidence` | `gluon_real_patterns_path` | Same as hybrid, but must name per-shape or sub-operation evidence for each branch. |
| `base_or_shared_gluon` | `gluon_component_traits_path` | Compatibility attribution profile. It must not by itself turn a plain Base task into a Gluon worker task. |

Documentation-gate merge order is additive: mandatory docs
(`gluon_skill_path`, `gluon_always_read_path`, `gluon_search_policies_path`),
profile docs, explicit `required_gluon_docs`, then heuristic docs for old or
hand-written tasks. Explicit docs augment the profile; they must not replace the
mandatory/profile set.

Profile and metadata consistency:

- `gluon_doc_profile` is a documentation routing profile. Valid values are the
  enum in the table above, such as `extension_l0_minimal`, `memory_lowering`,
  `matrix_lowering`, or `jit_aot_sensitive`.
- `mi3xx`, `raw`, or architecture/backend names belong in
  `gluon_baseline_profile`, `target_backend`, or benchmark metadata, not
  `gluon_doc_profile`.
- Ordinary L0 execution anchors should not become `jit_aot_sensitive` merely
  because they use `@gluon.jit` or have a compile goal. Use
  `jit_aot_sensitive` only when JIT/AOT, prebuilt artifacts, signatures,
  scratch, target triples, or version-sensitive integration is part of the task.
- Task frontmatter is the planner/dispatch/worker gate source of truth. Avoid
  repeating `gluon_doc_profile`, `required_gluon_docs`, and target fields in the
  task body; if repeated for human readability, the body must match frontmatter
  exactly.
- Task body prose must not override frontmatter values for
  `required_output_dialect`, `source_origin`, `gluon_doc_profile`,
  `required_gluon_docs`, `Target symbol`, `Target component`, or
  `required_patch_target_symbols`. If the template needs human-readable routing,
  derive it from frontmatter rather than hand-writing a second value.

Run-level Gluon feature metadata is a planner/search-space switch. Task-level
`required_output_dialect`, `implementation_layer`, and `extension_layer` are the
worker/selector contract. Legacy `search_set` values are audit metadata and
must not be the sole reason a plain Triton task receives a Gluon implementation
doc gate.

`source_origin` is fail-closed: missing or unknown origin follows
`generated_overlay` rules. Only an input that is already a measured production
AMD Gluon operator may use `source_origin=existing_amd_gluon_operator` to enter
the in-dialect refinement path. `input_dialect=amd_gluon` alone is not enough.

Keep output dialect separate from internal API policy:

- `required_output_dialect=amd_gluon` requires a real AMD Gluon execution path.
- `required_output_dialect=mixed` requires visible host-side dispatch between
  plain Triton and AMD Gluon paths.
- legal `tl.range` / `tl.constexpr` or source-preserved `tl.where` inside
  `@gluon.jit` is governed by `gluon_tl_policy` and does not make the output
  `mixed`.

Do not include `gluon_examples_doc_path` unless the task explicitly needs a
schematic example. Examples are not common context.

## round_progression

Round 1:

- choose optimization directions from the standard Triton priority order;
- fill mandatory plain Triton competitors for high-value directions;
- for `source_origin=existing_amd_gluon_operator`, treat those same direction
  names as taxonomy and emit 1-3 narrow in-dialect AMD Gluon refinements instead
  of mandatory Base coverage. Plain/shared tasks are optional comparison,
  fallback, portable-component, or plain-subkernel work;
- include at most one Extension L0 when a concrete Gluon overlay reason exists;
  that L0 must name the exact same-batch Base/plain `Plain competitor` it
  overlays. Shared or paired tasks are not valid plain competitors;
- for layout-heavy kernels, make Extension L0 a narrow compileable subpath
  rather than a full-kernel Gluon rewrite. The L0 task should explicitly say
  which index/mask/load/matrix skeleton is in scope and reject leftover plain
  Triton tensor APIs in that scoped `@gluon.jit` path;
- L0 execution path must be explicit: use `inline_scoped_helper` only when the
  language boundary permits the scoped change, `separate_gluon_kernel` only when
  second launch/temp buffer overhead is accepted as execution-anchor evidence,
  `whole_jit_kernel` only when the whole helper/kernel is the minimum executable
  unit, and `infeasible` when the scoped change would require widening into
  forbidden paths;
- if `minimum_executable_unit=infeasible`, do not emit a
  `required_output_dialect=amd_gluon` task. Keep the slot for Base/Shared or
  emit a non-dispatched infeasibility report;
- a local expression such as a scale epilogue is not an executable Gluon overlay
  unless the task can name an inline scoped helper or a separate kernel whose
  output feeds the measured correctness result. If the whole kernel is the
  minimum executable unit, say so with `whole_kernel_required_reason` and do not
  forbid whole-kernel rewrite in the same task;
- make the first Gluon patch prove the smallest real executed Gluon path. Later
  patches in the same task should be single-variable experiments so round 2 can
  attribute which component helped or hurt;
- for `whole_jit_kernel` L0, make the execution proof visible. Prefer a distinct
  `_gluon` kernel symbol launched from the measured wrapper; if using same-name
  in-place replacement, record the original launch, replacement decorator, and
  measured-output path explicitly in strategy notes;
- for low-latency kernels or tiny stages, do not spend L0 follow-up patches on
  repeated launch-constant sweeps after a slower correctness pass. Record the
  slower anchor as evidence and keep Base/Shared width;
- include paired probes for the same optimization direction when budget allows.

Round 2:

- refine the safe plain/base anchor;
- add shared transplants from useful Extension / Shared evidence;
- create Gluon variants only when traits or evidence justify the same
  optimization direction;
- make Extension L1 memory/buffer lowering refine an executed and
  performance-viable or locally-winning Gluon anchor. If no Gluon anchor passed,
  shrink the task to a layout or memory smoke path instead of restarting from
  plain Triton.
- every Extension L1 task must include `Anchor patch`, `Anchor speedup`,
  `Anchor execution: true`, `Comparison target: anchor_patch`, `Allowed change`,
  and `Reject if`. Do not emit L1 when the anchor is below `0.5x`, has material
  per-shape regression, or lacks executed AMD Gluon; emit anchor diagnosis or a
  smaller L0 repair instead.
- make Extension L1 matrix/MFMA lowering refine a verified Gluon layout anchor.
  If the planner cannot name the anchor and operand/result layouts, do not emit
  an MFMA task yet.
- treat correctness-passing but slower Gluon as neutral/slower evidence. Keep
  Base/Shared width and emit only one targeted Gluon refinement with a concrete
  performance hypothesis.

Round 3:

- converge around the safe anchor;
- keep only useful Gluon/Shared evidence;
- add hybrid dispatch only when per-shape or sub-operation evidence supports
  different winners; slower Gluon correctness anchors alone are not enough.

## result_attribution

- `Gluon-positive`: final best is `amd_gluon` or `mixed` and beats the Base
  anchor.
- `Gluon-informed`: final best is `plain_triton`, but includes a component
  first validated in Shared or Extension evidence.
- `Gluon-neutral`: Gluon candidates ran but final best uses only plain/base evidence.
- `Gluon-slower`: Gluon candidates executed and passed correctness but lost to
  the safe anchor or had material shape regression.
- `Blocked`: baseline correctness or benchmark contract fails before GEAK
  optimization.
