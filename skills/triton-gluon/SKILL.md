---
name: triton-gluon
description: Rewrite or optimize Triton-family kernels (`plain_triton`, `nv_gluon`, or `amd_gluon`) toward valid AMD-facing Gluon candidates. Use when the task involves Gluon, `input_dialect`, AMD Triton layouts, MFMA or WMMA paths, or translating NVIDIA-facing Gluon assumptions to AMD.
tier: general
---

# Triton-Gluon

This skill is the workflow entrypoint. Detailed API, architecture, examples, and
real-pattern guidance live in `skills/triton-gluon/docs/`.

## When To Use

- The task stays on the Triton route: `kernel_type = triton`.
- The input dialect is `plain_triton`, `nv_gluon`, or `amd_gluon`.
- The task is to rewrite, optimize, benchmark, or review a Triton-family kernel
  where AMD Gluon may be a valid candidate.

Do not apply this skill to HIP, CK, ASM, FlyDSL, PyTorch-to-FlyDSL, or
torch2hip tasks.

## Non-Negotiables

- Keep `kernel_type = triton`.
- Valid optimized outputs are `plain_triton`, `amd_gluon`, or `mixed`.
- Never produce an optimized `nv_gluon` output path.
- `mixed` is explicit host-side dispatch between verified `plain_triton` and
  `amd_gluon` paths; it is not an input dialect.
- Gluon is an additive implementation overlay for a named optimization
  direction. Do not replace the plain Triton/Base competitor for that direction.
- Required AMD Gluon tasks must attempt a real AMD Gluon patch. Pure Triton
  fallback is not a valid success.
- Compile-only success is not enough; benchmark against the same correctness
  and performance contract.

## Audience And Routing

- Planner: use `00_always_read.md` and `10_search_policies.md` to decide whether
  a Triton optimization direction deserves an AMD Gluon implementation layer and
  which `gluon_doc_profile` / `required_gluon_docs` to emit.
- Worker: use the task's doc gate and profile to read only the implementation
  package needed for the scoped patch. Do not read API/examples/backup docs by
  default.

## Required Before Editing

1. Read `skills/triton-gluon/docs/00_always_read.md`.
2. View every doc path listed in task metadata or in the injected
   `REQUIRED BEFORE EDITING OR SAVE_AND_TEST` block with
   `str_replace_editor command="view"` before editing or calling
   `save_and_test`.
3. Write these strategy-note blocks before the first edit:
   `Gluon knowledge lookup plan`, `Gluon implementation plan`,
   `Performance hypothesis`, `Same ABI comparison`, and `Patch evolution`.
4. Use only supported imports:

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

Do not use `from triton import gluon` as an availability probe.

Hard contracts:

- Required AMD Gluon tasks must execute real Gluon code; import-only,
  helper-only, empty, or pure plain Triton fallback patches are not success.
- Do not add broad `try/except Exception` fallback that silently succeeds through
  the plain Triton path.
- Change one subpath/component unless the task explicitly says
  `bundle_allowed=true`.
- If the task names `Target symbol`, `Target component`,
  `required_patch_target_symbols`, or scoped names in `Allowed change`, only
  report success when that target path is modified and executes the intended
  Gluon code.
- Do not create backup or temporary files such as `.bak`, `.backup`, `.orig`,
  `.tmp`, or editor-swap copies.

`save_and_test` enforces the required-doc gate for Gluon tasks before patch
contract checks.

## Workflow

1. Classify input dialect:
   - `plain_triton`
   - `nv_gluon`
   - `amd_gluon`
2. Preserve source semantics:
   - launcher shape and wrapper ABI
   - indexing and pointer arithmetic
   - masks and boundary behavior
   - dtype / precision behavior
   - correctness oracle
   - benchmark intent
3. Check runtime and integration contract:
   - installed Triton version
   - `triton.experimental.gluon` JIT availability
   - JIT/AOT/prebuilt package gates
   - `AMDMFMALayout.instr_shape` form
   - target backend, architecture, `num_warps`, `num_ctas`
   - operator-local support matrix
4. Choose the smallest valid action path for the named optimization direction:
   - plain Triton/Base competitor
   - AMD Gluon L0 overlay viability
   - trait-specific AMD Gluon L1 overlay
   - Shared transplant of one portable component
   - Hybrid/mixed dispatch when per-shape or sub-operation evidence justifies it
5. Verify correctness and benchmark. Reject per-shape regressions.
6. If the first Gluon candidate is correct, keep following the existing
   `patch_0` / `patch_1+` evolution rule: change one component at a time,
   compare against the safe anchor, and record keep/revert/compose-later
   evidence. Do not invent a separate tuning workflow from a quick guide.

## Required AMD Gluon Implementation Contract

For `required_output_dialect=amd_gluon`, do not start by wrapping the original
plain Triton kernel in `@gluon.jit`. First produce the lookup plan and
implementation plan, then edit only the scoped path.

- L0 proves the smallest real executed Gluon anchor. It may be slower than Base;
  record that as evidence instead of launch-tuning blindly.
- For `extension_intent=execution_anchor`, a slower correctness-passing L0 is
  overhead evidence, not permission to expand the same scope into L1/MFMA unless
  the next task names a removable overhead.
- L1 memory/MFMA work must refine an executed Gluon/mixed anchor or shrink back
  to L0. It must name anchor evidence and one allowed change.
- Matrix/MFMA work requires result layout, operand layouts, `convert_layout`,
  target op, epilogue/store plan, and a performance reason before editing.
- Buffer work starts with generic `gl.load` / `gl.store`; use AMD buffer ops only
  after dtype/layout preconditions are known.
- Profile-routed implementation details live in `20_component_traits.md`,
  `30_architecture_notes.md`, `50_api_reference.md`, and `60_real_patterns.md`.

## Planner Metadata Contract

Planner-generated Triton-family tasks should include these metadata fields for
audit and result attribution. They are not the primary planning axis; the
primary axis is the named optimization direction.

```yaml
required_output_dialect: plain_triton | amd_gluon | mixed | any
search_set: base | shared | extension  # optional legacy compatibility bucket
gluon_doc_profile: extension_l0_minimal | nv_to_amd_translation | memory_lowering | matrix_lowering | shape_bucketed_dispatch | jit_aot_sensitive | shared_transplant | gluon_variant_from_anchor | hybrid_dispatch | hybrid_dispatch_from_evidence | base_or_shared_gluon
required_gluon_docs:
  - gluon_skill_path
  - gluon_always_read_path
  - gluon_search_policies_path
source_base_family: <family_id for the overlaid Triton direction>
plain_competitor: <same-batch plain Triton task label for Round 1 L0 overlays>
```

Add `gluon_component_traits_path`, `gluon_architecture_notes_path`,
`gluon_api_reference_path`, or `gluon_real_patterns_path` when the task route
requires them. `save_and_test` gates on these paths.

Use `required_output_dialect=amd_gluon` for required AMD Gluon L0/L1 tasks and
`required_output_dialect=mixed` only for explicit host-side dispatch between
verified plain Triton and AMD Gluon paths. `search_set` is optional compatibility
metadata, not a planning axis.
Round 1 L0 overlays must name a same-batch `plain_competitor`; that task's
`Base family` must match the overlay's `source_base_family`.

## Read Next

- Always start with `skills/triton-gluon/docs/00_always_read.md`.
- Planner/task allocation: `skills/triton-gluon/docs/10_search_policies.md`.
- Component traits: `skills/triton-gluon/docs/20_component_traits.md`.
- Architecture, JIT/AOT, version, operator support:
  `skills/triton-gluon/docs/30_architecture_notes.md`.
- API syntax, snippets, compatibility checks, failure-fix order:
  `skills/triton-gluon/docs/50_api_reference.md`.
- Real operator patterns, benchmark rules, source-first triggers:
  `skills/triton-gluon/docs/60_real_patterns.md`.
- Schematic examples, at most one relevant section:
  `skills/triton-gluon/docs/40_examples.md`.
- Residual backup routing:
  `skills/triton-gluon/docs/70_backup_details.md`.

## Self-Check Before Reporting Success

- Did every required doc path get viewed with `str_replace_editor view`?
- Did the pre-edit knowledge lookup plan list task signals, routed docs/headings,
  viewed status, and missing details?
- Did the pre-edit Gluon implementation plan list all parent layouts, aranges,
  broadcasts, tensor creations, and the single allowed subpath/component?
- Does a required `amd_gluon` result contain real Gluon markers or a valid
  `mixed` path?
- Does every `@gluon.jit` tensor creation use `gl.*` with explicit layouts, not
  leftover `tl.arange`, `tl.zeros`, `tl.full`, `tl.load`, `tl.dot`, or `tl.where`?
- Does every `@gluon.jit` body avoid leftover `tl.*` math/reductions such as
  `tl.sigmoid`, `tl.max`, `tl.sum`, and direct `tl.dot`?
- Does the patch avoid backup/temp files such as `.bak`, `.backup`, `.orig`,
  `.tmp`, or editor-swap copies?
- Does a required `mixed` result contain both Base/dispatch and Gluon paths?
- Did correctness pass for every benchmark shape?
- Did the patch avoid modifying harness, environment, or benchmark contract?
- Did the result compare against the safe anchor when prior evidence exists?
