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
- Gluon is an additive Extension Set. Do not replace Base Triton coverage.
- Required AMD Gluon tasks must attempt a real AMD Gluon patch. Pure Triton
  fallback is not a valid success.
- Compile-only success is not enough; benchmark against the same correctness
  and performance contract.

## Required Before Editing

1. Read `skills/triton-gluon/docs/00_always_read.md`.
2. Use its `stable_split_doc_index` and `Task routing` table to map the task to
   exact split-doc files and headings.
3. If task metadata provides `required_gluon_docs`, view every listed absolute
   path with `str_replace_editor command="view"` before editing or calling
   `save_and_test`.
4. Do not implement from memory or guess Gluon API names. If routed primary docs
   lack detail, read `skills/triton-gluon/docs/70_backup_details.md`; if the
   detail is still missing, report the missing split-doc route so it can be
   added.
5. Before the first edit, write a short Gluon knowledge lookup plan in your
   strategy notes. It must list:
   - task signals found in the kernel or task prompt, such as `tl.arange`,
     `tl.dot`, MFMA, buffer ops, `[:, None]`, `BlockedLayout`, AOT/JIT, or
     module wiring;
   - the exact split-doc file and heading for each signal;
   - whether that heading has been viewed;
   - any missing detail that must be resolved before editing.
6. After the lookup plan is complete, write a short Gluon implementation plan in your
   strategy notes. It must name:
   - the parent layout for each logical 1D/2D expression;
   - every `tl.arange` / tensor creation that will become
     `gl.arange(..., layout=...)`, `gl.zeros(..., layout=...)`, or
     `gl.full(..., layout=...)`;
   - every broadcast or `[:, None]` / `[None, :]` expression and its matching
     `SliceLayout(axis, parent)`;
   - every `tl.*` device scalar/math use in the edited Gluon path and its
     `gl.*` equivalent (`gl.cdiv`, `gl.minimum`, `gl.maximum`, `gl.exp`,
     `gl.where`, etc.);
   - for buffer ops, the loaded element dtype, typed `other` value/layout, and
     `stored_value` dtype before using `buffer_load` / `buffer_store`;
   - the performance hypothesis before editing: why this Gluon change might help
     versus the safe Base/plain path, what overhead it may add, and what evidence
     would make it neutral or slower instead of a win;
   - the patch evolution plan: `patch_0` should be the smallest real executed
     Gluon path that can pass correctness; each later patch should change one
     component or one dispatch decision and record the expected and observed
     effect before moving on;
   - whether the task is L0, L1, or Hybrid, and the single subpath/component it
     is allowed to change. If the task names a stage/helper, write
     `Target symbol: <symbol>` and do not modify a different stage as the
     successful patch. If you define a new `_..._gluon` helper, wire the host or
     caller so that helper is actually executed; a definition-only helper while
     the dispatch still uses the plain Triton path is not a valid result.

`save_and_test` enforces the required-doc gate for Gluon tasks.

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
4. Choose the smallest valid action path:
   - Base Triton competitor
   - Extension L0 AMD Gluon viability
   - trait-specific Extension L1
   - Shared transplant
   - Hybrid/mixed dispatch when evidence justifies it
5. Verify correctness and benchmark. Reject per-shape regressions.

## Required AMD Gluon Implementation Contract

For `required_output_dialect=amd_gluon`, do not start by wrapping the original
plain Triton kernel in `@gluon.jit`. First produce the lookup plan and
implementation plan above, then edit only the scoped path.

- Extension L0: make one small Gluon subpath compile and preserve correctness
  semantics. For layout-heavy kernels, this is usually one index/mask
  expression, one load/store path, or one matrix-layout skeleton, not the full
  attention/decode/GEMM body. L0 success is a real executed Gluon anchor; it may
  be slower than Base. Do not keep tuning `num_warps` / block sizes as if L0 is a
  performance win unless the hypothesis explains which overhead was removed.
- Extension L1: refine a correctness-passing Gluon or mixed anchor. If no anchor
  exists, shrink the task to an L0-style layout/memory smoke path.
- Matrix/MFMA work: do not introduce MFMA until result layout, operand layouts,
  `convert_layout`, valid `AMDMFMALayout.instr_shape`, and a plausible
  performance reason are known. If the plan cannot say why MFMA reduces a real
  hot path rather than adding layout conversion, extra loops, or dispatch
  overhead, keep the task at L0 layout/memory viability.
- Module wiring: define the Gluon helper in the edited module before importing
  or dispatching to it. Do not reference guessed `_..._gluon` symbols.
- Stage execution: for stage-specific L1 tasks, touching a target symbol means
  exact identifier use plus an executed Gluon path. A `_target_gluon` helper does
  not satisfy `target` unless the patch also dispatches/calls that helper.
- Buffer ops: prefer generic `gl.load` / `gl.store` first. When using AMD
  `buffer_load`, create `other` as a typed Gluon tensor compatible with
  `ptr.dtype.element_ty`; when using `buffer_store`, cast `stored_value` to the
  destination pointer element dtype if needed.
- Patch evolution: keep patch history useful for later rounds. Do not bundle
  buffer ops, MFMA, layout rewrites, scheduler changes, and launch tuning in one
  patch unless the task explicitly says `bundle_allowed=true`. After each
  `save_and_test`, record whether the changed component should be kept, reverted,
  or composed later.

## Planner Metadata Contract

Planner-generated Gluon tasks should include:

```yaml
search_set: base | shared | extension
required_output_dialect: plain_triton | amd_gluon | mixed | any
gluon_doc_profile: extension_l0_minimal | nv_to_amd_translation | matrix_lowering | shape_bucketed_dispatch | jit_aot_sensitive | shared_transplant | hybrid_dispatch
required_gluon_docs:
  - gluon_skill_path
  - gluon_always_read_path
  - gluon_search_policies_path
```

Add `gluon_component_traits_path`, `gluon_architecture_notes_path`,
`gluon_api_reference_path`, or `gluon_real_patterns_path` when the task route
requires them. `save_and_test` gates on these paths.

## Read Next

- Always start with `skills/triton-gluon/docs/00_always_read.md`.
- Planner/task allocation: `skills/triton-gluon/docs/10_search_policies.md`.
- Component traits: `skills/triton-gluon/docs/20_component_traits.md`.
- Architecture, JIT/AOT, version, operator support:
  `skills/triton-gluon/docs/30_architecture_notes.md`.
- API syntax, snippets, compatibility checks, failure-fix order:
  `skills/triton-gluon/docs/50_api_reference.md`.
- Real aiter patterns, benchmark rules, source-first triggers:
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
- Does a required `mixed` result contain both Base/dispatch and Gluon paths?
- Did correctness pass for every benchmark shape?
- Did the patch avoid modifying harness, environment, or benchmark contract?
- Did the result compare against the safe anchor when prior evidence exists?
