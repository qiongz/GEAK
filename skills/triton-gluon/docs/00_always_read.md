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
- Use `stable_split_doc_index` and `Task routing` to choose exact split-doc
  files and headings.
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

Use this stable index instead of reading broad background references. Do not
implement from memory or guess missing Gluon APIs. A planner or worker must map
the task to the exact file and heading below before generating or editing a
Gluon candidate.

Worker gate: the planner or dispatch metadata may provide `required_gluon_docs`.
Before editing or calling `save_and_test`, view every listed absolute path with
`str_replace_editor command="view"`. `save_and_test` rejects Gluon tasks until
the required docs have been viewed.

Component traits:

- `semantics_contract` -> this file, `### Trait: semantics_contract`
- `dialect_plain_triton` -> this file, `### Trait: dialect_plain_triton`
- `dialect_nv_gluon` -> this file, `### Trait: dialect_nv_gluon`
- `dialect_amd_gluon` -> this file, `### Trait: dialect_amd_gluon`
- `layout_basic` -> `20_component_traits.md`, `### Trait: layout_basic`
- `layout_slice_broadcast` -> `20_component_traits.md`,
  `### Trait: layout_slice_broadcast`
- `layout_source_first_required` -> `20_component_traits.md`,
  `### Trait: layout_source_first_required`
- `layout_derivation_and_cost_model` -> `20_component_traits.md`,
  `## layout_derivation_and_cost_model`
- `memory_generic` -> `20_component_traits.md`, `### Trait: memory_generic`
- `memory_amd_buffer` -> `20_component_traits.md`,
  `### Trait: memory_amd_buffer`
- `memory_shared_async_descriptor` -> `20_component_traits.md`,
  `### Trait: memory_shared_async_descriptor`
- `matrix_none` -> `20_component_traits.md`, `### Trait: matrix_none`
- `matrix_dot` -> `20_component_traits.md`, `### Trait: matrix_dot`
- `matrix_scaled_dot` -> `20_component_traits.md`,
  `### Trait: matrix_scaled_dot`
- `matrix_wmma_descriptor` -> `20_component_traits.md`,
  `### Trait: matrix_wmma_descriptor`
- `execution_jit_aot_sensitive` -> `30_architecture_notes.md`,
  `### Trait: execution_jit_aot_sensitive`
- `version_sensitive` -> `30_architecture_notes.md`,
  `### Trait: version_sensitive`
- `operator_support_sensitive` -> `30_architecture_notes.md`,
  `### Trait: operator_support_sensitive`
- `amd_arch_family_quick_directions` -> `30_architecture_notes.md`,
  `## amd_arch_family_quick_directions`
- `shape_coverage_unknown` -> `20_component_traits.md`,
  `### Trait: shape_coverage_unknown`
- `shape_coverage_single` -> `20_component_traits.md`,
  `### Trait: shape_coverage_single`
- `shape_coverage_multi` -> `20_component_traits.md`,
  `### Trait: shape_coverage_multi`
- `shape_coverage_bucketed` -> `20_component_traits.md`,
  `### Trait: shape_coverage_bucketed`
- `shape_layout_constexpr_risk` -> `20_component_traits.md`,
  `### Trait: shape_layout_constexpr_risk`
- `shape_dispatch_required` -> `20_component_traits.md`,
  `### Trait: shape_dispatch_required`

Search policies:

- `optimization_direction_metadata_sets` -> `10_search_policies.md`,
  `### Search policy: optimization_direction_metadata_sets`
- `optimization_direction_dialect_overlay` -> `10_search_policies.md`,
  `### Search policy: optimization_direction_dialect_overlay`
- `overlay_priority_routing` -> `10_search_policies.md`,
  `### Search policy: overlay_priority_routing`
- `measurement_boundary_policy` -> `10_search_policies.md`,
  `### Search policy: measurement_boundary_policy`
- `evidence_anchored_composition` -> `10_search_policies.md`,
  `### Search policy: evidence_anchored_composition`
- `dialect_contract_metadata` -> `10_search_policies.md`,
  `### Search policy: dialect_contract_metadata`

Detailed references:

- API surface, concrete snippets, quick patterns, compatibility checklist, and
  debug order -> `50_api_reference.md`
- Matrix lowering ladder by architecture family ->
  `50_api_reference.md`, `### matrix_lowering_ladders_by_arch`
- Evidence inventory and generalized real-code patterns ->
  `60_real_patterns.md`, `evidence_inventory_for_guide_authoring` and
  `generalized_real_code_patterns`
- product/runtime context, writing model, layout/sync/descriptor mental model,
  NVIDIA/AMD family differences, real operator patterns, kernel-family
  optimization paths, source-first triggers, repo-local notes, benchmark rules,
  and anti-patterns -> `60_real_patterns.md`
- Representative schematic examples -> `40_examples.md`
- Residual long-form details not covered by the routed primary docs ->
  `70_backup_details.md`

Task routing:

| Task need or visible signal | Must read |
| --- | --- |
| Any Triton-Gluon task | `00_always_read.md`, `product_contract`, `semantic_contract`, `output_and_fallback_contract` |
| Planning optimization-direction overlays or compatibility metadata | `10_search_policies.md`, `### Search policy: optimization_direction_metadata_sets`, `### Search policy: optimization_direction_dialect_overlay`, `### Search policy: overlay_priority_routing`, `### Search policy: dialect_contract_metadata` |
| L0 scope classification, micro-anchor, or compile-risk whole-kernel task | `10_search_policies.md`, `### Search policy: l0_scope_classification`; `20_component_traits.md`, `whole_kernel_layout_map_recipe` |
| Later-round composition from prior results | `10_search_policies.md`, `### Search policy: evidence_anchored_composition` |
| End-to-end, wrapper-heavy, or multi-stage operator pipeline | `10_search_policies.md`, `### Search policy: measurement_boundary_policy`; `60_real_patterns.md`, `benchmark_boundary_and_integration_costs` |
| `plain_triton -> amd_gluon` | `00_always_read.md`, `### Trait: dialect_plain_triton`; then relevant traits in `20_component_traits.md` |
| `nv_gluon -> amd_gluon` | `00_always_read.md`, `### Trait: dialect_nv_gluon`; `60_real_patterns.md`, `nvidia_amd_family_differences`; optionally `40_examples.md`, `nv_gluon_to_amd_gluon_translation` |
| existing `amd_gluon` input | `00_always_read.md`, `### Trait: dialect_amd_gluon`; `30_architecture_notes.md`, `### Trait: operator_support_sensitive` |
| `tl.arange`, block sizes, layout, masks, broadcasts, `[:, None]`, `expand_dims`, `BlockedLayout`, `SliceLayout`, `convert_layout` | `20_component_traits.md`, `### Trait: layout_basic`, `### Trait: layout_slice_broadcast`, and `layout_derivation_and_cost_model` |
| broadcast or layout compile error such as `expected expand_dims input layout`, rank mismatch, or parent-layout mismatch | `20_component_traits.md`, `### Trait: layout_slice_broadcast` and `whole_kernel_layout_map_recipe`; `50_api_reference.md`, `broadcast_failure_debug_recipe` |
| `tl.load` / `tl.store`, buffer ops, memory-bound path | `20_component_traits.md`, `### Trait: memory_generic` and/or `### Trait: memory_amd_buffer` |
| shared memory, swizzle, async, descriptor, `tdm`, cluster, scheduler hints | `20_component_traits.md`, `### Trait: memory_shared_async_descriptor`; `50_api_reference.md`, `shared_memory_synchronization_cluster` and `descriptor_and_tensor_memory_surface` |
| `tl.dot`, `tl.dot_scaled`, MFMA, WMMA, FP8/FP4/scales, `AMDMFMALayout`, `AMDWMMALayout`, `DotOperandLayout` | `20_component_traits.md`, `### Trait: matrix_dot`, `### Trait: matrix_scaled_dot`, and/or `### Trait: matrix_wmma_descriptor`; `50_api_reference.md`, `matrix_lowering_ladders_by_arch` and `amd_quick_patterns`; `30_architecture_notes.md`, `amd_arch_family_quick_directions` |
| dot or matrix compile error such as `DotOperandEncodingAttr`, `tt.dot failed to infer`, or missing `gl.dot` | `20_component_traits.md`, `### Trait: matrix_dot`; `50_api_reference.md`, `dot_lowering_minimal_recipe` and `matrix_lowering_ladders_by_arch` |
| reduction or accumulator layout mismatch | `20_component_traits.md`, `layout_derivation_and_cost_model`; `50_api_reference.md`, `reduction_accumulator_layout_recipe` |
| target backend, `gfx942`, `gfx950`, `gfx1250`, arch guards, wave32/wave64 assumptions | `30_architecture_notes.md`, `amd_arch_family_quick_directions`, target section, and `runtime_target_resolution`; `60_real_patterns.md`, `nvidia_amd_family_differences` |
| Triton version, `instr_shape`, JIT/AOT, prebuilt kernels | `30_architecture_notes.md`, `### Trait: version_sensitive` and `### Trait: execution_jit_aot_sensitive`; `50_api_reference.md`, `version_and_compatibility_checklist` |
| multi-shape or bucketed benchmark | `20_component_traits.md`, relevant `shape_coverage_*` trait plus `### Trait: shape_layout_constexpr_risk` / `### Trait: shape_dispatch_required` |
| concrete code skeleton or exact API call needed | `50_api_reference.md`; do not invent API names from memory |
| source-derived guide claim, limited Gluon sample, backend-only support, or `gfx950` capability question | `60_real_patterns.md`, `evidence_inventory_for_guide_authoring`; then the relevant API/architecture/source section |
| real attention-style, GEMM, preshuffled, descriptor, or RDNA WMMA pattern | `60_real_patterns.md`, `generalized_real_code_patterns`, `real_operator_patterns`, and `optimization_paths_by_kernel_family` |
| source shows `DistributedLinearLayout`, `PartitionedSharedLayout`, host `TensorDescriptor`, nested 3D/5D layouts, unshuffle transforms, JIT/AOT package gates | `60_real_patterns.md`, `source_first_triggers`; then read operator-local source before editing |
| one schematic example would help | at most one relevant section in `40_examples.md` after reading the trait/policy file above |
| routed split docs still lack a needed nuance or rationale | `70_backup_details.md`; report the missing route if still unresolved |

If a task does not match any route above, stop and read `00_always_read.md`
again plus the closest trait/policy file. If the routed primary docs lack the
required detail, read `70_backup_details.md` and report the missing split-doc
route if still unresolved.

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

- Planner-audited fields: `Extension layer`, `Optimization direction`,
  `Source Base family`, `Plain competitor`, `Gluon overlay reason`,
  `Overlay priority`, `Implementation layer`, `Performance hypothesis`,
  `Measurement boundary`, `Comparison target`, `Allowed change`, and `Reject if`.
- Worker strategy notes must include `Gluon knowledge lookup plan`,
  `Gluon implementation plan`, `Performance hypothesis`, `Same ABI comparison`,
  and `Patch evolution` before the first edit.

The worker implementation plan is intentionally concise in this entrypoint.
It must name the scoped subpath/component, same-ABI comparison, freeze contract,
hot-path evidence, parent layouts or "not applicable", matrix path or "none",
buffer path or "generic load/store", module wiring, target symbol/component,
and patch hygiene. Use profile-routed implementation docs for the detailed API,
layout, MFMA, buffer, JIT/AOT, or real-operator checklist.

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
- Do not create backup or temporary files (`*.bak`, `*.backup`, `*.orig`,
  `*.tmp`, editor swap files).
- L0 is the smallest executed Gluon anchor and may be slower than Base. L1
  memory/MFMA/Hybrid work requires executed anchor evidence and a concrete
  performance hypothesis.
- For `extension_intent=execution_anchor`, slower correctness-passing L0 is
  overhead evidence. Record the overhead source and do not expand the same scope
  into L1 unless a later task names a removable overhead.
- If a scoped L0 cannot be implemented without touching forbidden paths, do not
  widen the patch. Follow `allowed_execution_path` / `scope_infeasible_policy`:
  inline only when legal, split to a separate Gluon kernel only when allowed, or
  use a `whole_jit_kernel` anchor only when the task declares it as the
  `minimum_executable_unit`. If the task says the scope is `infeasible`, report
  or shrink instead of saving a required Gluon patch.
- Patch evolution after failure is constrained: helper-not-executed fixes only
  wiring/launch/output feeding; forbidden-scope failures must revert forbidden
  changes; slow correctness passes may only change one named overhead source.

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

Triton-family task frontmatter may include:

```yaml
required_output_dialect: plain_triton | amd_gluon | mixed | any
gluon_doc_profile: extension_l0_minimal | nv_to_amd_translation | memory_lowering | matrix_lowering | shape_bucketed_dispatch | jit_aot_sensitive | shared_transplant | gluon_variant_from_anchor | hybrid_dispatch | hybrid_dispatch_from_evidence | base_or_shared_gluon
search_set: base | shared | extension  # optional compatibility bucket
source_origin: generated_overlay | existing_amd_gluon_operator | nv_gluon_translation | unknown
gluon_tl_policy: strict_generated | preserve_existing_allowed | production_source_allowed
layout_construction_policy: host_preferred | constexpr_in_kernel_allowed | source_preserve
execution_mode: jit | aot | mixed_jit_aot | prebuilt
```

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
- compare aggregate latency and per-shape speedups;
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
