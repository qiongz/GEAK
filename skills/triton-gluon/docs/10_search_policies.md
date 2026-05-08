# Triton-Gluon Search Policies

Read this file when planning tasks, reviewing prior rounds, or deciding how
Base Triton, Shared transplants, and AMD Gluon candidates should be allocated.
Implementation details live in `20_component_traits.md`.

## Internal Index

- `### Search policy: base_shared_extension`
- `### Search policy: optimization_direction_dialect_overlay`
- `### Search policy: evidence_anchored_composition`
- `### Search policy: measurement_boundary_policy`
- `### Search policy: dialect_contract_metadata`
- `gluon_doc_gate_metadata`
- `trait_policy_separation`
- `round_progression`
- `result_attribution`

### Search policy: base_shared_extension

GEAK stores Triton-family optimization metadata in three sets:

- **Base Set**: a plain Triton implementation of an optimization direction.
- **Shared Set**: a paired comparison or transplant for the same optimization
  direction across dialects.
- **Extension Set**: an AMD Gluon implementation of an optimization direction,
  including L0 viability and L1 trait-specific lowering.

These sets are not mutually exclusive optimization directions. They are
metadata used for audit, result attribution, and selector contracts. The primary
planning axis is the optimization direction: split/decomposition, fusion,
memory/layout cleanup, shape specialization, persistent/launch amortization, or
another HIP/Triton family from the main planner.

Plain Triton coverage is mandatory for every high-value direction. Gluon is an
additional implementation option, not a reason to remove the plain Triton
competitor.

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
Optimization direction: <main HIP/Triton strategy>
Source Base family: <family_id>
Gluon overlay reason: explicit_layout | buffer_path | matrix_lowering | shape_bucket | dialect_specific_memory | local_subpath_win
Extension layer: L0 | L1 | Hybrid
```

### Search policy: optimization_direction_dialect_overlay

Use the standard HIP/Triton priority order to choose what to optimize first.
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
- `base_refine` may refine the safe Base anchor, but must not add a Gluon
  rewrite.
- `shared_transplant` may transplant one portable component into the safe
  anchor. It must preserve launcher ABI and safe-anchor algorithm unless the
  allowed change says otherwise.
- `gluon_variant` / `gluon_variant_from_anchor` must re-express the safe-anchor
  algorithm in AMD Gluon. It must preserve anchor semantics before adding
  memory, matrix, scheduler, or dispatch changes.
- `hybrid_dispatch` / `hybrid_dispatch_from_evidence` must add visible
  host-side shape/feature dispatch around verified Base and Gluon candidates.
  It must keep the Base path for shapes or sub-operations where Base wins.
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
  `base_shared_extension`, `optimization_direction_dialect_overlay`,
  `measurement_boundary_policy`, `evidence_anchored_composition`, and
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
gluon_doc_profile: extension_l0_minimal | nv_to_amd_translation | memory_lowering | matrix_lowering | shape_bucketed_dispatch | jit_aot_sensitive | shared_transplant | gluon_variant_from_anchor | hybrid_dispatch | hybrid_dispatch_from_evidence
required_gluon_docs:
  - gluon_skill_path
  - gluon_always_read_path
  - gluon_search_policies_path
```

Profile guidance:

- `extension_l0_minimal`: add `gluon_component_traits_path` and
  `gluon_api_reference_path`. The task must ask the worker for a pre-edit
  layout/API mapping and must scope layout-heavy kernels to one Gluon subpath.
  Treat L0 as an executed correctness anchor; do not describe it as a
  performance win unless benchmark evidence beats the safe Base/plain path. For
  low-latency kernels or tiny stages, L0 should not ask for launch tuning or
  block-size sweeps after a slower correctness pass; it should record overhead
  evidence and stop as an anchor.
- `nv_to_amd_translation`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_real_patterns_path`.
- `memory_lowering`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, `gluon_api_reference_path`, and
  `gluon_real_patterns_path`. This profile must refine a verified Gluon layout
  anchor; do not wrap the original plain Triton body in `@gluon.jit` just to add
  `buffer_load` / `buffer_store`. A buffer/load/store task should keep
  `Matrix path: none` unless it explicitly declares matrix lowering or
  `bundle_allowed=true`.
- `matrix_lowering`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_api_reference_path`. The task must
  name the verified layout anchor, result layout, operand layouts, and target
  matrix op; otherwise it should be downgraded to L0 layout viability. It must
  also name the expected performance mechanism, such as replacing a real hot dot
  path without adding dominant layout-conversion or dispatch overhead.
- `shape_bucketed_dispatch`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_real_patterns_path`.
- `jit_aot_sensitive`: add `gluon_architecture_notes_path` and
  `gluon_api_reference_path`.
- `shared_transplant`: add `gluon_component_traits_path` and
  `gluon_real_patterns_path` when the source component comes from real Gluon or
  aiter evidence.
- `gluon_variant_from_anchor`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, `gluon_api_reference_path`, and
  `gluon_real_patterns_path`. The task must name the safe anchor and preserve
  its algorithm before changing performance components.
- `hybrid_dispatch`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_real_patterns_path`.
- `hybrid_dispatch_from_evidence`: add `gluon_component_traits_path`,
  `gluon_architecture_notes_path`, and `gluon_real_patterns_path`. The task must
  name per-shape or sub-operation evidence for each dispatch branch.

Do not include `gluon_examples_doc_path` unless the task explicitly needs a
schematic example. Examples are not common context.

## round_progression

Round 1:

- choose optimization directions from the standard HIP/Triton priority order;
- fill mandatory plain Triton competitors for high-value directions;
- include at most one Extension L0 when a concrete Gluon overlay reason exists;
- for layout-heavy kernels, make Extension L0 a narrow compileable subpath
  rather than a full-kernel Gluon rewrite. The L0 task should explicitly say
  which index/mask/load/matrix skeleton is in scope and reject leftover plain
  Triton tensor APIs in that scoped `@gluon.jit` path;
- make the first Gluon patch prove the smallest real executed Gluon path. Later
  patches in the same task should be single-variable experiments so round 2 can
  attribute which component helped or hurt;
- for low-latency kernels or tiny stages, do not spend L0 follow-up patches on
  repeated launch-constant sweeps after a slower correctness pass. Record the
  slower anchor as evidence and keep Base/Shared width;
- include paired probes for the same optimization direction when budget allows.

Round 2:

- refine the safe Base anchor;
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
- `Gluon-neutral`: Gluon candidates ran but final best uses only Base evidence.
- `Gluon-slower`: Gluon candidates executed and passed correctness but lost to
  the safe anchor or had material shape regression.
- `Blocked`: baseline correctness or benchmark contract fails before GEAK
  optimization.
