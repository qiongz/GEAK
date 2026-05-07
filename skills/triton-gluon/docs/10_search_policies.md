# Triton-Gluon Search Policies

Read this file when planning tasks, reviewing prior rounds, or deciding how
Base Triton, Shared transplants, and AMD Gluon candidates should be allocated.
Implementation details live in `20_component_traits.md`.

## Internal Index

- `### Search policy: base_shared_extension`
- `### Search policy: evidence_anchored_composition`
- `### Search policy: dialect_contract_metadata`
- `gluon_doc_gate_metadata`
- `trait_policy_separation`
- `round_progression`
- `result_attribution`

### Search policy: base_shared_extension

GEAK plans Triton-family optimization as three sets:

- **Base Set**: plain Triton search that must preserve no-regression coverage.
- **Shared Set**: a strategy valid in both plain Triton and AMD Gluon, or a
  transplant of an insight from one side to the other.
- **Extension Set**: AMD-Gluon-only exploration, including L0 viability and L1
  trait-specific lowering.

Base Set is mandatory. Extension and Shared tasks are additive and must not
replace Base Triton coverage.

Base Set task prompts should include:

```text
Base family: <family_id>
```

Shared Set task prompts should include:

```text
Shared source family: <base_family_id>
```

Extension Set task prompts should include:

```text
Extension layer: L0 | L1 | Hybrid
```

### Search policy: evidence_anchored_composition

This is a search policy, not a component trait. It says how to compose evidence
from previous rounds.

When prior-round evidence exists:

1. Identify the safe anchor:
   - prefer the best verified Base Set patch;
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
```

Composition candidates must compare against the safe anchor, not only the
original baseline. A patch that is faster than the original baseline but slower
than the safe anchor is not a valid composition win.

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
  `base_shared_extension`, `evidence_anchored_composition`, and
  `dialect_contract_metadata`.

Do not treat `evidence_anchored_composition` as a component trait. It chooses a
safe anchor, identifies portable components, and enforces comparison target.
The implementation detail still comes from component traits.

### Search policy: dialect_contract_metadata

This metadata contract is Triton-family only. Do not apply it to HIP, CK, ASM,
FlyDSL, PyTorch-to-FlyDSL, or torch2hip tasks.

```yaml
search_set: base | shared | extension
required_output_dialect: plain_triton | amd_gluon | mixed | any
```

Mapping:

- Base Set:
  - `search_set = base`
  - `required_output_dialect = plain_triton`
- Shared transplant:
  - `search_set = shared`
  - `required_output_dialect = any`
- True AMD Gluon Extension:
  - `search_set = extension`
  - `required_output_dialect = amd_gluon`
- Hybrid dispatch:
  - `search_set = shared` or `extension`
  - `required_output_dialect = mixed`

Required AMD Gluon Extension tasks must run before staged dispatch can stop.
They must attempt a real AMD Gluon patch and cannot silently succeed as plain
Triton fallback.

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
gluon_doc_profile: extension_l0_minimal | nv_to_amd_translation | matrix_lowering | shape_bucketed_dispatch | jit_aot_sensitive | shared_transplant | hybrid_dispatch
required_gluon_docs:
  - gluon_skill_path
  - gluon_always_read_path
  - gluon_search_policies_path
```

Profile guidance:

- `extension_l0_minimal`: add `gluon_component_traits_path` and
  `gluon_api_reference_path`.
- `nv_to_amd_translation`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_real_patterns_path`.
- `matrix_lowering`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_api_reference_path`.
- `shape_bucketed_dispatch`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_real_patterns_path`.
- `jit_aot_sensitive`: add `gluon_architecture_notes_path` and
  `gluon_api_reference_path`.
- `shared_transplant`: add `gluon_component_traits_path` and
  `gluon_real_patterns_path` when the source component comes from real Gluon or
  aiter evidence.
- `hybrid_dispatch`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_real_patterns_path`.

Do not include `gluon_examples_doc_path` unless the task explicitly needs a
schematic example. Examples are not common context.

## round_progression

Round 1:

- fill mandatory Base families;
- include one Extension L0 if AMD Gluon is allowed;
- include small Shared probes when budget allows.

Round 2:

- refine the safe Base anchor;
- add shared transplants from useful Extension / Shared evidence;
- create Gluon variants only when traits or evidence justify them.

Round 3:

- converge around the safe anchor;
- keep only useful Gluon/Shared evidence;
- add hybrid dispatch only when per-shape evidence supports different winners.

## result_attribution

- `Gluon-positive`: final best is `amd_gluon` or `mixed` and beats the Base
  anchor.
- `Gluon-informed`: final best is `plain_triton`, but includes a component
  first validated in Shared or Extension evidence.
- `Gluon-neutral`: Gluon candidates ran but final best uses only Base evidence.
- `Blocked`: baseline correctness or benchmark contract fails before GEAK
  optimization.
