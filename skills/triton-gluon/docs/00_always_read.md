# Triton-Gluon Always Read

Every planner or worker touching Triton-Gluon must read this file first. It is
the stable contract shared by Base, Shared, Extension, and Hybrid tasks.

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
| Later-round composition from prior results | `10_search_policies.md`, `### Search policy: evidence_anchored_composition` |
| End-to-end, wrapper-heavy, or multi-stage operator pipeline | `10_search_policies.md`, `### Search policy: measurement_boundary_policy`; `60_real_patterns.md`, `benchmark_boundary_and_integration_costs` |
| `plain_triton -> amd_gluon` | `00_always_read.md`, `### Trait: dialect_plain_triton`; then relevant traits in `20_component_traits.md` |
| `nv_gluon -> amd_gluon` | `00_always_read.md`, `### Trait: dialect_nv_gluon`; `60_real_patterns.md`, `nvidia_amd_family_differences`; optionally `40_examples.md`, `nv_gluon_to_amd_gluon_translation` |
| existing `amd_gluon` input | `00_always_read.md`, `### Trait: dialect_amd_gluon`; `30_architecture_notes.md`, `### Trait: operator_support_sensitive` |
| `tl.arange`, block sizes, layout, masks, broadcasts, `[:, None]`, `expand_dims`, `BlockedLayout`, `SliceLayout`, `convert_layout` | `20_component_traits.md`, `### Trait: layout_basic`, `### Trait: layout_slice_broadcast`, and `layout_derivation_and_cost_model` |
| `tl.load` / `tl.store`, buffer ops, memory-bound path | `20_component_traits.md`, `### Trait: memory_generic` and/or `### Trait: memory_amd_buffer` |
| shared memory, swizzle, async, descriptor, `tdm`, cluster, scheduler hints | `20_component_traits.md`, `### Trait: memory_shared_async_descriptor`; `50_api_reference.md`, `shared_memory_synchronization_cluster` and `descriptor_and_tensor_memory_surface` |
| `tl.dot`, `tl.dot_scaled`, MFMA, WMMA, FP8/FP4/scales, `AMDMFMALayout`, `AMDWMMALayout`, `DotOperandLayout` | `20_component_traits.md`, `### Trait: matrix_dot`, `### Trait: matrix_scaled_dot`, and/or `### Trait: matrix_wmma_descriptor`; `50_api_reference.md`, `matrix_lowering_ladders_by_arch` and `amd_quick_patterns`; `30_architecture_notes.md`, `amd_arch_family_quick_directions` |
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
- `gfx950`, CDNA4, scaled MFMA, FP8/FP4, or backend-only support evidence ->
  `60_real_patterns.md` / `evidence_inventory_for_guide_authoring`,
  `30_architecture_notes.md` / `amd_arch_family_quick_directions`, and
  `50_api_reference.md` / `matrix_lowering_ladders_by_arch`.

Do not write code while any required row is still `viewed=no`, unless the route
is genuinely missing and has been recorded as a missing-doc detail.

## pre_edit_gluon_patch_contract

Planner task prompts and worker strategy notes have different contracts:

- Planner-audited Gluon prompt fields: `Extension layer`, `Optimization
  direction`, `Source Base family`, `Plain competitor`, `Gluon overlay reason`,
  `Overlay priority`, `Implementation layer`, `Performance hypothesis`,
  `Measurement boundary`, `Comparison target`, `Allowed change`, and
  `Reject if`.
- Worker pre-edit plan fields: the full `Gluon knowledge lookup plan` and
  `Gluon implementation plan` below. These are written in strategy notes before
  code edits; they do not all need to appear as top-level planner prompt fields.

For any task that will edit a real AMD Gluon candidate, write the following plan
before changing code:

```text
Gluon implementation plan:
- Scope: L0 | L1 | Hybrid, and the single subpath/component allowed by this task.
- Direction binding: optimization direction, Source Base family, Plain
  competitor, Gluon overlay reason, Overlay priority, and Implementation layer.
- Measurement boundary: kernel_only | fair_make_inputs_run_kernel |
  full_operator; include whether integration cost is in scope.
- Same ABI comparison: how the Gluon candidate is compared to the plain Triton
  competitor without changing input packing, cache layout, or wrapper semantics.
- Freeze contract: launcher signature, grid, constexprs, wrapper ABI, and
  non-target modules that must stay unchanged.
- Hot path evidence: profiler/source evidence that the scoped component is on
  the measured path.
- Regression ladder: compare against true baseline, safe plain Triton anchor,
  and any prior Gluon anchor named by the task.
- Parent layouts: one line per logical expression, e.g. [H,C], [H,R], [H,N],
  [Q,K], [P,V].
- Index tensors: every original `tl.arange(...)` and its replacement
  `gl.arange(..., layout=<layout>)`.
- Tensor creation: every accumulator / mask / temporary and its explicit layout.
- Broadcasts: every `[:, None]` / `[None, :]` pair and the shared parent layout
  used to derive both `SliceLayout` inputs.
- 2D offset expressions: every pair of 1D slice tensors that will be combined,
  and the explicit expansion (`x[:, None]`, `y[None, :]`) that makes them shape
  compatible before addition or masking.
- Matrix path: result layout, operand layouts, `convert_layout`, and target op,
  or `none` if this patch is not doing matrix lowering.
- Pure-kernel API audit: every `tl.*` still present inside the edited
  `@gluon.jit` body and its Gluon lowering. Include `tl.sigmoid`, `tl.max`,
  `tl.sum`, and all `tl.dot` calls explicitly.
- Buffer path: loaded dtype, `other` dtype/layout, stored value dtype, and
  whether generic `gl.load` / `gl.store` is safer for this patch.
- Performance hypothesis: why this scoped Gluon path might improve the safe
  Base/plain path, what extra overhead it may add (layout conversion, launch or
  dispatch branch, scalar loops, memory path cost), and what result should make
  it neutral/slower evidence instead of a win.
- Patch evolution: what `patch_0` will prove, what single component the next
  patch may change, and which observation would cause the component to be kept,
  reverted, or saved only as evidence for later composition.
- Module wiring: helper functions being defined and how the host dispatch calls
  them.
  If the task names a stage or helper, include `Target symbol: <symbol>` and
  only report success if the patch touches that exact symbol and executes the
  intended Gluon path.
- Patch hygiene: files that will be modified. Do not create backup or temporary
  source copies such as `.bak`, `.backup`, `.orig`, `.tmp`, or editor-swap
  files.
```

Rules:

- A required AMD Gluon task must not begin as a plain Triton body wrapped in
  `@gluon.jit`.
- Inside a `@gluon.jit` body, do not leave plain Triton tensor APIs such as
  `tl.arange`, `tl.load`, `tl.store`, `tl.dot`, `tl.where`, `tl.zeros`, or
  `tl.full` in the edited Gluon path. Reductions such as `tl.max` and `tl.sum`
  must also become `gl.max` and `gl.sum`.
- Inside the edited Gluon path, also prefer Gluon scalar/math APIs such as
  `gl.cdiv`, `gl.minimum`, `gl.maximum`, `gl.max`, `gl.sum`, and `gl.exp` over `tl.*` equivalents.
  Host-side launch math outside `@gluon.jit` may still use `triton.cdiv`.
- Do not leave `tl.sigmoid` inside `@gluon.jit`. If the local Gluon API docs do
  not prove a `gl.sigmoid` exists, lower sigmoid with documented Gluon math
  such as `1 / (1 + gl.exp(-x))`.
- Do not leave `tl.dot` inside `@gluon.jit`. A dot path is only valid after the
  plan names result layout, operand `DotOperandLayout`s, `convert_layout`, and a
  target matrix op such as `gl.amd.cdna3.mfma`; otherwise reduce scope to a
  non-dot L0 path. Do not substitute a generic `gl.dot` for a task whose
  hypothesis requires MFMA or operand-layout lowering.
- If the plan cannot name the layout for an index, mask, temporary, or matrix
  operand, reduce the task scope before editing.
- If the plan cannot name the optimization direction or Gluon overlay reason,
  do not write a Gluon patch; keep the work in the plain Triton/HIP path.
- A Gluon patch must not be the only implementation of a high-value direction
  unless the task explicitly requires AMD Gluon. Keep or compare against the
  plain Triton competitor.
- Do not combine differently sized 1D `SliceLayout` tensors directly. If an
  offset uses `[X] + [Y]`, `[X] + [Z]`, or similar, first broadcast them into the
  intended 2D parent expression with `[:, None]` / `[None, :]`.
- If a stage-specific L1 task cannot name the exact function/helper it must
  touch, reduce or rewrite the task before editing. Changing a different stage
  is not a valid success.
- Defining `target_gluon` while the host/wrapper still dispatches the plain
  `target` path is not a valid Gluon execution result. The plan must name the
  call/dispatch line that executes the Gluon helper, or directly convert the
  original target function that existing dispatch already calls.
- Required pure AMD Gluon tasks must not add broad `try/except Exception`
  fallback to a plain Triton launcher. Compile/runtime failure should be
  reported by `save_and_test`, not hidden behind a plain fallback success path.
- Do not create or commit backup files (`*.bak`, `*.backup`, `*.orig`, `*.tmp`,
  editor swap files). Backup source copies can pollute dialect detection and are
  invalid patch content.
- If L1 has no correctness-passing Gluon/mixed anchor, do not attempt MFMA or
  buffer lowering over the full original kernel; shrink to an L0-style smoke
  path and record that no anchor exists.
- L0 compile/correctness success is an executed Gluon anchor, not a promised
  speedup. If it is slower than Base, record it as anchor or negative performance
  evidence and do not keep tuning launch constants without a concrete overhead
  hypothesis.
- For low-latency kernels or tiny stages (roughly sub-100us benchmark cases),
  L0 must be especially small: one executed subpath, no repeated launch tuning,
  no `num_warps` / block-size sweep after a slower correctness pass. Record the
  overhead source and stop escalation unless a later task names a verified anchor
  and concrete removed overhead.
- L1 memory/MFMA work must state why the change is performance-plausible before
  editing. If the plan cannot identify the hot path being reduced, the layout
  conversions avoided, and the benchmark path that will execute it, do not
  escalate beyond a narrow layout/memory candidate.
- Patch sequence rule: `patch_0` should establish the smallest real executed
  Gluon path that can compile and pass correctness. Later patches should make
  one change at a time (`buffer_load`, `buffer_store`, one MFMA subpath, one
  layout fix, one launch constant, or one dispatch condition). Before the next
  patch, record `Changed component`, `Expected effect`, `Observed effect`, and
  `Keep / revert / compose later`.

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
gluon_doc_profile: extension_l0_minimal | nv_to_amd_translation | memory_lowering | matrix_lowering | shape_bucketed_dispatch | jit_aot_sensitive | shared_transplant | gluon_variant_from_anchor | hybrid_dispatch | hybrid_dispatch_from_evidence
search_set: base | shared | extension  # optional compatibility bucket
```

Rules:

- `required_output_dialect`, `Implementation layer`, and `Extension layer` are
  the output contract source of truth.
- `search_set` is optional compatibility metadata, not the optimization
  direction.
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
