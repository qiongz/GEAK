# Triton-Gluon Always Read

Stable entrypoint for Triton-Gluon tasks. Planner reads this for product and
task-contract boundaries; workers read it for hard execution contracts and
split-doc routing. It is not an API cookbook.

## Internal Index

- `non_negotiables`
- `required_before_editing_or_save_and_test`
- `self_check`
- `product_contract`
- `stable_split_doc_index`
- `task_first_route_proof_protocol`
- `pre_edit_gluon_patch_contract`
- `semantic_contract`
- `dialect_contract`
- `imports_and_runtime`
- `output_and_fallback_contract`
- `measurement_boundary_contract`
- `multi_shape_contract`
- `detailed_reference_index`
- `anti_patterns`

## non_negotiables

- Keep `kernel_type = triton`.
- Valid optimized outputs are `plain_triton`, `amd_gluon`, or `mixed`.
- Never produce optimized `nv_gluon`.
- Required `amd_gluon` tasks must produce real AMD Gluon or a valid `mixed`
  path. Pure Triton fallback is not a success.
- `mixed` means explicit host-side dispatch between verified plain Triton and
  AMD Gluon paths; it is not an input dialect.
- Gluon is additive. Do not remove the plain Triton competitor for a high-value
  optimization direction.
- Compile-only success is not enough.

## required_before_editing_or_save_and_test

- Read this file first.
- Use task metadata (`gluon_doc_profile`, `required_gluon_docs`,
  `Target component`, `Allowed change`, and task signals) to choose the exact
  split-doc files and headings.
- Treat task frontmatter/metadata as the source of truth for doc routing and
  target contracts. If the task body repeats `gluon_doc_profile`,
  `required_gluon_docs`, or target fields and conflicts with frontmatter, record
  `Task correction` and follow the frontmatter contract.
- Do not rely on repeated task-body metadata for routing. If the body repeats
  frontmatter fields only for readability, it must be derived from and
  consistent with frontmatter.
- If task metadata provides `required_gluon_docs`, view every listed absolute
  path with `str_replace_editor command="view"` before editing or calling
  `save_and_test`.
- Do not guess Gluon API names from memory.
- If routed docs lack a required detail, read `70_backup_details.md`; use
  the missing-doc report path there if the detail is still absent.
- Before editing, write a Gluon knowledge lookup plan in strategy notes. It must
  map task signals to exact split-doc files/headings and record what has been
  viewed.
- Before editing, write a Gluon implementation plan in strategy notes. Do not
  edit first and repair layout errors later.

## self_check

Before reporting success:

- Required docs were viewed through `str_replace_editor view`.
- The Gluon knowledge lookup plan exists and maps each relevant signal to a
  viewed split-doc file/heading or a reported missing-doc route.
- The Gluon implementation plan exists and names every layout parent,
  `gl.arange(..., layout=...)`, tensor creation layout, broadcast/SliceLayout
  context, and allowed subpath/component.
- Source semantics, wrapper ABI, masks, dtype behavior, and benchmark intent are
  preserved.
- Required `amd_gluon` output is `amd_gluon` or valid `mixed`, not pure Triton.
- Required `mixed` output contains visible dispatch/no-regression semantics.
- Every benchmark shape passes correctness and avoids material regression.
- The patch does not modify harness, environment, or benchmark contract.

## product_contract

- Gluon is a Triton-family feature, not a new top-level kernel type.
- Keep `kernel_type = triton`.
- Use `input_dialect = plain_triton | nv_gluon | amd_gluon` only to classify
  the Triton-family source.
- Valid optimized outputs are:
  - `plain_triton`
  - `amd_gluon`
  - `mixed` when explicit host-side dispatch preserves no-regression.
- Never produce an optimized `nv_gluon` output path.
- Plain Triton winning the benchmark is a valid outcome.
- `Base`, `Shared`, and `Extension` are metadata for audit and attribution. The
  planner should choose an optimization direction first, then choose whether the
  implementation layer is plain Triton, AMD Gluon, paired comparison, or mixed.

## stable_split_doc_index

This file is the stable entrypoint, not a full routing table. Keep the route
short:

- Planner/search policy: `10_search_policies.md`, including
  `overlay_direction_vs_mechanism` for same-direction overlays and
  `atomic_component_lattice` / `l0_scope_decision_before_emit` for L0 scope
  choices.
- Worker axis 1, kernel-family writing model: `60_real_patterns.md` for
  source-first cases, wrapper/multi-stage behavior, and benchmark boundaries.
- Worker axis 2, atomic component implementation: `20_component_traits.md` for
  the one component changed by the current patch.
- Concrete API snippets, rewrites, and failure triage: `50_api_reference.md`.
- Target family, version, JIT/AOT, descriptor, MFMA/WMMA architecture details:
  `30_architecture_notes.md`.
- Examples: `40_examples.md`, at most one relevant section.
- Missing routed detail: `70_backup_details.md`, then report the missing route.

Worker gate: planner or dispatch metadata may provide `required_gluon_docs`.
Before editing or calling `save_and_test`, view every listed absolute path with
`str_replace_editor command="view"`. `save_and_test` rejects Gluon tasks until
the required docs have been viewed.

Use `gluon_doc_profile`, `required_gluon_docs`, and task signals to select the
exact heading in those docs. Do not replace this gate with the ROCm Gluon
knowledge base; that page is background/RAG, not an execution contract.

## task_first_route_proof_protocol

Worker reading order is hard:

1. Task file / clean task packet.
2. `COMMANDMENT`.
3. Required docs and only the task-relevant headings.
4. Operator-local source.
5. Harness / manifest / benchmark cases.
6. `Required execution route proof`.
7. Edit.

Before editing or calling `save_and_test`, persist a strategy artifact such as
`strategy_notes.md` containing:

- `Task objective extraction`: objective, target symbols, allowed change,
  recommended route, and reject conditions.
- `Required execution route proof`: current wrapper/kernel path, target
  wrapper/kernel path, guard conditions, output feeding, reduce/temporary path
  skip-or-preserve behavior, same ABI proof, and measurement boundary
  reconciliation.
- `Patch evolution ledger`: `Changed component`, `Expected effect`,
  `Observed effect`, `Keep/Revert`, and `Next patch allowed scope` for each
  patch.

Failure-layer lock:

- `helper_not_executed` or `target_not_touched` -> next patch may only fix
  wrapper wiring, launch, target association, or measured output feeding.
- `scope_violation` -> revert the forbidden scope before doing anything else.
- `slow_correct` -> change one named removable overhead only.
- `compile/layout` -> fix only that compile/layout layer.

For `full_operator` wrapper/reduction or one-shot/output-feeding tasks, the
first patch is wrapper dispatch/output feeding only unless the task explicitly
permits kernel-body, MFMA, or reduce-algorithm rewriting.

## pre_edit_knowledge_lookup_contract

Before editing any AMD Gluon candidate, write this lookup plan in strategy
notes:

```text
Gluon knowledge lookup plan:
- Task signals: <keywords/API/layout/error risks seen in task and source>.
- Required docs/headings:
  - <signal> -> <split doc path> :: <heading> :: viewed=yes/no
- Source sections viewed: <operator-local files/functions read for layout or
  module wiring>.
- Missing details: <none, or the exact route/detail still unresolved>.
```

Signal routing examples:

- `tl.arange`, masks, broadcasts, `[:, None]` -> `20_component_traits.md` /
  `layout_basic` and `layout_slice_broadcast`, plus `50_api_reference.md` /
  `common_rewrite_table` or `slice_broadcast_recipe`.
- `tl.cdiv`, `tl.minimum`, `tl.maximum`, `tl.max`, `tl.sum`, `tl.sigmoid`,
  `tl.exp`, `tl.where` inside `@gluon.jit` -> `20_component_traits.md` /
  `layout_basic` and
  `50_api_reference.md` / `common_language_api_surface`; device math should use
  `gl.*`.
- MFMA, `tl.dot`, `instr_shape` -> `20_component_traits.md` / `matrix_dot`,
  `30_architecture_notes.md` / version or target section, and
  `50_api_reference.md` / `matrix_lowering_ladders_by_arch` and
  `amd_quick_patterns`.
- `buffer_load` / `buffer_store` -> `20_component_traits.md` /
  `memory_amd_buffer` and `60_real_patterns.md` /
  `extension_l1_memory_lowering_anchor`.
- `_..._gluon` helper wiring, JIT/AOT, prebuilt modules ->
  `50_api_reference.md` / `jit_entry_and_host_launcher` and
  `30_architecture_notes.md` / `execution_jit_aot_sensitive`.
- `BlockedLayout` verifier, `size_per_thread`, parent-layout mismatch ->
  `20_component_traits.md` / `layout_basic`, `layout_slice_broadcast`, and
  `layout_derivation_and_cost_model`.
- compile-risk whole-kernel or composite path -> `10_search_policies.md` /
  `l0_scope_classification`, `20_component_traits.md` /
  `whole_kernel_layout_map_recipe`, and `50_api_reference.md` /
  `broadcast_failure_debug_recipe` / `dot_lowering_minimal_recipe` as needed.
- `gfx950`, CDNA4, scaled MFMA, FP8/FP4, or backend-only support evidence ->
  `60_real_patterns.md` / `evidence_inventory_for_guide_authoring`,
  `30_architecture_notes.md` / `amd_arch_family_quick_directions`, and
  `50_api_reference.md` / `matrix_lowering_ladders_by_arch`.

Do not write code while any required row is still `viewed=no`, unless the route
is genuinely missing and has been recorded as a missing-doc detail.

## pre_edit_gluon_patch_contract

Planner task prompts and worker strategy notes have different contracts:

- Planner-audited fields are defined in `10_search_policies.md`
  (`optimization_direction_metadata_sets`, `dialect_contract_metadata`, and
  `gluon_doc_gate_metadata`). Workers should treat task metadata as the contract,
  but should not copy or extend planner schemas while editing.
- Any task that enters Gluon worker docs/context must include `Gluon knowledge
  lookup plan`, `Gluon implementation plan`, `Performance hypothesis`, `Same ABI
  comparison`, and `Patch evolution` before the first edit. This includes
  generated Gluon overlays, L1/Hybrid/mixed tasks that edit a Gluon path,
  required `amd_gluon` tasks, and existing AMD Gluon in-dialect refinements.
- Base/plain Triton tasks are not forced into this Gluon state machine. A
  `plain_subkernel_refine` task with `required_output_dialect=plain_triton`
  should keep ordinary strategy notes and no-regression benchmark comparison,
  not Gluon lookup/layout/execution-route notes.
- Strategy notes must be useful before and after each patch, not only as a final
  summary. Preserve the fields needed to choose the next patch: `Required
  execution route`, `Same ABI comparison`, `Layout plan`, `Overhead
  attribution`, and `Next patch decision`.

The worker implementation plan is intentionally concise in this entrypoint.
It must name the scoped subpath/component, same-ABI comparison, freeze contract,
hot-path evidence, parent layouts or "not applicable", matrix path or "none",
buffer path or "generic load/store", module wiring, target symbol/component,
and patch hygiene. Use profile-routed implementation docs for the detailed API,
layout, MFMA, buffer, JIT/AOT, or real-operator checklist.

For complex or wrapper-heavy targets, add this route before editing:

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

For any task where the declared measurement boundary may differ from the actual
`save_and_test` or task-runner boundary, add this reconciliation before editing:

```text
Measurement boundary reconciliation:
- Declared boundary: kernel_only | fair_make_inputs_run_kernel | full_operator
- Actual save_and_test / task-runner path:
- Host work included in measured timing: yes|no|unknown
- Host layout / dispatch overhead risk:
- Consequence for next patch:
```

If the actual runner goes through a Python wrapper or fair/full boundary, host
layout factories, wrapper dispatch, and cache construction can be measured even
when the task says `kernel_only`.

For slow correctness-passing L0 anchors, add this post-patch decision block:

```text
Next patch decision:
- measurement boundary:
- same ABI / public wrapper preserved: yes|no
- fair-boundary host overhead risk:
- Layout plan: gl.arange / gl.zeros / masked load-store / store layout / dtype fallback
- Overhead attribution: <one or more named sources>
- Decision: try_one_removable_overhead:<name> | stop_not_viable_for_l1
```

`required_patch_target_symbols` must be callable symbols or explicit route
entries, not words split from prose. If the route cannot be stated, report
`Task correction: target route ambiguous` instead of guessing.

Hard rules:

- Do not start a required AMD Gluon task by wrapping the full plain Triton body
  in `@gluon.jit`.
- Required AMD Gluon output must execute real Gluon code. Definition-only helpers,
  import-only changes, empty patches, and pure plain Triton fallback are invalid.
- Do not add broad `try/except Exception` fallback that hides Gluon failure behind
  a plain Triton success path.
- Keep one subpath/component per patch unless `bundle_allowed=true`.
- If the task names `Target symbol`, `Target component`, scoped names in
  `Allowed change`, or `required_patch_target_symbols`, success requires touching
  and executing that exact target path.
- Preserve public API/import contracts. Do not rename, delete, or replace the
  exported wrapper that the harness imports. Add an internal Gluon helper or
  guarded dispatch inside the existing public function instead.
- Do not create backup or temporary files (`*.bak`, `*.backup`, `*.orig`,
  `*.tmp`, editor swap files).
- L0 is the smallest executed Gluon anchor and may be slower than Base. L1
  memory/MFMA/Hybrid work requires executed anchor evidence and a concrete
  performance hypothesis.
- A Gluon L0 `Plain competitor` is not a synthetic placeholder. It must remain a
  real plain Triton no-regression candidate with its own scoped performance
  reason; do not widen or rewrite Base work merely to make the Gluon task pass
  anchor audit.
- For `extension_intent=execution_anchor`, slower correctness-passing L0 is
  overhead evidence. Record the overhead source and do not expand the same scope
  into L1 unless a later task names a removable overhead.
- `tiny_stage_overhead` is an umbrella label, not a stop condition. Before
  writing `removable_by_next_task: none` or `stop_not_viable_for_l1`, check
  whether a concrete single-variable overhead applies: host layout construction,
  layout padding, mask path, typed fallback, loop invariant, small launch params,
  or memory path overhead.
- Use this compact overhead template for slower L0 anchors:

```text
L0 overhead attribution:
- observed_speedup:
- not_viable_for_l1: true|false
- overhead_source: layout_padding | layout_conversion | host_layout_construction | mask_path_overhead | typed_fallback_overhead | loop_invariant_overhead | small_stage_launch_params | tiny_stage_overhead | memory_path_overhead | unknown
- evidence:
- Next patch decision: try_one_removable_overhead:<name> | stop_not_viable_for_l1
```

- If a scoped L0 cannot be implemented without touching forbidden paths, do not
  widen the patch. Follow `allowed_execution_path` / `scope_infeasible_policy`:
  inline only when legal, split to a separate Gluon kernel only when allowed, or
  use a `whole_jit_kernel` anchor only when the task declares it as the
  `minimum_executable_unit`. If the task says the scope is `infeasible`, report
  or shrink instead of saving a required Gluon patch.
- A local L0 smoke/probe must not be upgraded by the worker into a whole-kernel
  rewrite. A whole-helper skeleton must prove compile/wiring/layout ancestry
  before MFMA, buffer ops, scheduler, epilogue, or performance tuning.
- Patch evolution after failure is constrained: helper-not-executed fixes only
  wiring/launch/output feeding; forbidden-scope failures must revert forbidden
  changes; slow correctness passes may only change one named overhead source.
- If `save_and_test` reports helper-only / not-executed Gluon, the next patch is
  wiring-only: launch the existing helper from the target path or feed its output
  into measured correctness. Do not continue editing layout, matrix, memory, or
  wrapper ABI until that execution contract is satisfied.
- If one wiring-only patch still cannot prove the required target path executes,
  stop and report `Task correction: target route ambiguous`; do not keep
  changing layout factories, kernel bodies, or wrapper branches.
- For `whole_jit_kernel`, prefer a visible `_gluon` kernel symbol launched from
  the measured wrapper. Same-name in-place replacement must record the wrapper,
  original launch line, replaced decorator, same-name launch, and output-feeding
  path in strategy notes.
- Before `save_and_test` on `strict_generated` tasks, scan edited `@gluon.jit`
  bodies for forbidden device `tl.*` APIs, including scalar `tl.load`,
  `tl.cdiv`, `tl.dot`, and `tl.store`. Convert generated-overlay device
  dataflow to `gl.*` before saving.
- Before `save_and_test`, also scan edited `@gluon.jit` bodies for runtime
  layout-object construction such as `layout=gl.SliceLayout(...)`,
  `gl.DotOperandLayout(...)`, or target-specific layout constructors. Layouts
  should be host-created and passed as `gl.constexpr`.

## semantic_contract

### Trait: semantics_contract

Preserve these before changing algorithms:

- launcher shape and wrapper ABI;
- indexing and pointer arithmetic semantics;
- masks and boundary behavior;
- dtype / precision behavior;
- correctness oracle;
- benchmark intent.

Compile-only success is not enough. A candidate must pass correctness and be
compared against the same benchmark contract.

## dialect_contract

### Trait: dialect_plain_triton

- First preserve the plain Triton launcher, indexing, masks, correctness oracle,
  and benchmark intent.
- Recover the implicit `tl.arange` / tile / `num_warps` layout before changing
  APIs.
- Convert the planned subset completely. A partial Gluon patch that leaves
  `tl.arange` or layout-less tensor creation in the edited `@gluon.jit` path is
  not a valid L0 result.
- Generate an early minimal `amd_gluon` viability task only after the plain
  Triton competitor coverage for high-value directions is preserved.
- Keep plain Triton as a benchmarked competitor unless AMD Gluon is required.

### Trait: dialect_nv_gluon

- Treat this as translation, not API renaming.
- Preserve common Gluon control flow, launcher shape, indexing, masks, and
  correctness scaffolding.
- Re-evaluate wave32 layouts, NVIDIA async-copy paths, TMA / descriptor
  assumptions, WGMMA / tensor-memory features, and cluster behavior before
  carrying anything to AMD.
- The output goal is AMD-facing Gluon, never an optimized `nv_gluon` path.

### Trait: dialect_amd_gluon

- Preserve the existing AMD-facing structure first.
- Optimize in dialect before drifting back to plain Triton.
- Read operator-local architecture guards before assuming support from a global
  Gluon-available helper.
- Keep JIT/AOT fallback behavior unless the benchmark contract proves it is not
  part of this path.

## imports_and_runtime

Use the supported Gluon import path:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

Do not use this as an availability check:

```python
from triton import gluon
```

That import path is not supported and must not justify fallback. If a required
AMD Gluon task fails, record the actual failure from a real Gluon patch using
`triton.experimental.gluon`, `@gluon.jit`, or `gl.*`.

## output_and_fallback_contract

`10_search_policies.md` is the canonical source for planner metadata values,
including `required_output_dialect`, `gluon_doc_profile`, `required_gluon_docs`,
`source_origin`, `gluon_tl_policy`, and `layout_construction_policy`. This
section records the worker-facing execution meaning of those fields.

Rules:

- `required_output_dialect`, `Implementation layer`, and `Extension layer` are
  the output contract source of truth.
- `search_set` is optional compatibility metadata, not the optimization
  direction and not sufficient by itself to trigger a worker Gluon doc gate.
- Run-level Gluon feature metadata decides whether the planner may widen the
  Triton search space with Gluon guidance; task-level `required_output_dialect`,
  `Implementation layer`, and `Extension layer` decide whether a dispatched
  worker must read implementation docs and produce a real Gluon path.
- `base_or_shared_gluon` is a compatibility documentation profile for Base or
  Shared tasks that carry Gluon metadata but do not map to a narrower
  implementation profile. It must not by itself turn a plain Triton Base task
  into a Gluon worker task.
- `required_output_dialect=amd_gluon` means a real AMD Gluon patch is required.
- Plain Triton fallback is not a valid success for required AMD Gluon tasks.
- Fallback is only evidence after a real Gluon patch was saved/tested and
  failed with a recorded compile/runtime error.
- `required_output_dialect=mixed` means explicit host-side dispatch between
  verified `plain_triton` and `amd_gluon` paths. It is valid only when the
  dispatch condition is visible to the audit and every benchmark shape preserves
  no-regression against the safe anchor.
- `mixed` is an output contract, not an input dialect and not a way to satisfy a
  required pure AMD Gluon Extension task.
- Shared tasks may output plain Triton; those results are
  `Gluon-informed` / shared transplant evidence, not Gluon-positive evidence.
- Required AMD Gluon tasks are checked by patch dialect classification.
- Output dialect is an execution-path contract. Do not mark an executed AMD Gluon
  path as `mixed` merely because legal or source-preserved `tl.*` occurs inside
  `@gluon.jit`.
- `gluon_tl_policy` is the internal API contract:
  - `strict_generated`: generated overlays may keep only compile-time-safe
    `tl.*` such as `tl.constexpr` and `tl.range`; new tensor/dataflow `tl.*`
    is rejected.
  - `preserve_existing_allowed`: translation tasks may preserve source-proven
    compatible `tl.*`, but new plain Triton tensor/dataflow remains suspect.
  - `production_source_allowed`: existing production AMD Gluon operators may
    preserve audited mixed `tl.*` idioms; new `tl.where` / `tl.cdiv` still needs
    explicit task evidence.
- `layout_construction_policy` is separate from output dialect. Generated L0
  defaults to host-created layouts; existing production source may preserve
  in-kernel `gl.constexpr` layout declarations. Runtime layout objects remain
  invalid.

Existing AMD Gluon refinement notes:

- Only tasks with `source_origin=existing_amd_gluon_operator` may use the
  in-dialect refinement path.
- Before editing, write `source contract preserved:` with the public wrapper,
  target/JIT/AOT guard, fallback or artifact route, layout declarations, and
  measured output feeding path.
- Write `single primary component:` and keep `patch_0` to that component.
- Write `rollback condition:` before changing wrapper dispatch, shape dispatch,
  matrix layout, reduction state, or partition/scheduler policy.
- After each patch, record `keep/revert/compose-later:`. Use
  `compose_later:<components>` when multiple components are needed; do not
  silently widen the current task.
- If the target is a plain `@triton.jit` subkernel, report task correction to
  `plain_subkernel_refine` instead of forcing AMD Gluon output.

## measurement_boundary_contract

For end-to-end or wrapper-heavy tasks, state the measurement boundary before
editing:

- `kernel_only`: only the hot kernel body is in scope.
- `fair_make_inputs_run_kernel`: input preparation, packing, cache conversion,
  and kernel launch are in scope.
- `full_operator`: the full operator path is in scope.

Compare plain Triton and AMD Gluon under the same ABI. If a patch changes input
packing, cache layout, wrapper dispatch, prebuilt artifact lookup, or JIT/AOT
selection, report that as integration evidence rather than a pure kernel-body
Gluon win. A kernel-only Gluon win cannot replace a fair/full-operator path
unless the fair/full boundary also preserves no-regression against the safe
anchor.

## multi_shape_contract

When the harness exposes multiple cases:

- correctness and performance use the same ordered case stream;
- every shape must pass correctness;
- compare like with like: baseline per-shape total against candidate per-shape
  total, or baseline geomean against candidate geomean;
- never compare a baseline total across shapes with a single candidate
  `GEAK_RESULT_LATENCY_MS` / fastest-shape latency;
- when `GEAK_BENCHMARK_RESULTS_MS` is available, use that per-shape map as the
  source of truth for no-regression and summary reporting;
- do not accept a patch with material per-shape regression;
- do not hardcode shape literals to win one case;
- prefer explicit host-side dispatch for bucketed shape regimes;
- do not hide bucket selection inside `@triton.heuristics`; the audit cannot
  verify per-shape coverage from heuristic mutation alone.

## detailed_reference_index

Read only the file needed for the current task:

- Concrete API or code skeleton needed: read `50_api_reference.md`.
- Runtime, layout, architecture, real operator pattern, or benchmark nuance needed:
  read `60_real_patterns.md`.
- One schematic example is enough: read at most one relevant section from
  `40_examples.md`.
- If the primary split docs still do not answer the question, read
  `70_backup_details.md` and report the missing route if still unresolved.

## anti_patterns

- Introducing `kernel_type=gluon`.
- Producing optimized `nv_gluon`.
- Renaming NVIDIA APIs into guessed AMD names.
- Replacing `tl.dot` with MFMA / WMMA without result and operand layouts.
- Wrapping a plain Triton body in `@gluon.jit` without first replacing tensor
  creation, indexing, loads/stores, masks, reductions, and matrix ops with
  layout-aware Gluon equivalents for the edited path.
- Reusing one `SliceLayout`-derived index across different parent layouts.
- Importing or dispatching to guessed `_..._gluon` helper names that are not
  defined in the patch/module.
- Adding MFMA when the source has no matrix trait.
- Starting with descriptor, async, persistent, scheduler, atomics, or work
  stealing before a simpler AMD Gluon candidate passes correctness.
- Claiming a Gluon task succeeded when the patch is actually pure Triton
  fallback.
