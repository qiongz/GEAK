# Triton-Gluon Search Policies

Planner-facing policy doc. Read this file when planning tasks, reviewing prior
rounds, or deciding how a Triton optimization direction should be tried as plain
Triton, AMD Gluon, paired comparison, shared transplant, or hybrid dispatch.
Implementation details live in worker-routed docs (`20_component_traits.md`,
`30_architecture_notes.md`, `50_api_reference.md`, and `60_real_patterns.md`).

## Internal Index

- `### Search policy: optimization_direction_metadata_sets`
- `### Search policy: optimization_direction_dialect_overlay`
- `### Search policy: overlay_priority_routing`
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

## l0_scope_classification

Before emitting a Round-1 L0 overlay, classify the proposed scope:

- Is this the smallest executable, attributable, low-coupling subpath?
- Does it contain layout-heavy risk such as complex element transforms, online
  reductions, multiple matrix paths, multiple 2D layout parents, nested
  `SliceLayout`, boundary-specific logic, or split-boundary control flow?
- Can the candidate execute and feed measured output without translating the
  whole algorithm?
- Is the expected outcome `execution_anchor` or `performance_candidate`?

L0 scope ladder:

1. scalar or 1D stage;
2. one load/store subpath;
3. one index/mask layout smoke path;
4. one source-first layout contract extraction;
5. one matrix subpath only after an executed anchor, or when that matrix subpath
   is explicitly the smallest viable component.

Round-1 L0 overlays must bind to the same component and same optimization
direction as `Plain competitor`, not merely the same broad `Source Base family`.
The referenced plain Triton competitor and the L0 overlay must use the exact
same `Optimization direction:` text and must both include an auditable
`Target component:` or `Allowed change:` with the same scoped symbol/stage. Use
`Target component:` / `Target symbol:` when the component is narrower than a
whole helper.
If the intended Gluon component is new to the batch, first create a plain Triton
Base competitor for that exact component and direction. For example, an overlay
for component B's memory/layout path needs a component-B memory/layout plain
competitor; it must not bind to a component-A cleanup/control-flow task just
because both share a broad Base family.

Optional metadata:

```yaml
extension_intent: execution_anchor | performance_candidate
expected_outcome: correctness_anchor_not_speedup | possible_speedup
not_viable_for_l1_if_slower_than_base: true
overhead_source_to_record: launch_layout_overhead | conversion_overhead | memory_path_overhead
minimum_executable_unit: inline_scoped_helper | separate_gluon_kernel | whole_jit_kernel | infeasible
allowed_execution_path: inline_scoped_helper | separate_gluon_kernel | whole_jit_kernel
scope_infeasible_policy: do_not_emit | shrink_or_report | separate_kernel_if_allowed
whole_kernel_required_reason: <required when a scoped component needs whole_jit_kernel>
target_symbol: <symbol or helper>
target_component: <scoped component description>
```

The execution-boundary fields are mandatory for Round-1 L0 AMD Gluon overlays.
`scope_infeasible_policy` describes what to do if the scoped component proves
infeasible; it is not itself an infeasibility declaration when
`minimum_executable_unit` names an executable path.
Do not add them to plain Triton tasks or to Gluon tasks where they would be
noise.

## round_progression

Round 1:

- choose optimization directions from the standard Triton priority order;
- fill mandatory plain Triton competitors for high-value directions;
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
