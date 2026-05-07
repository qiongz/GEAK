# Triton-Gluon Always Read

Every planner or worker touching Triton-Gluon must read this file first. It is
the stable contract shared by Base, Shared, Extension, and Hybrid tasks.

## Internal Index

- `non_negotiables`
- `required_before_editing_or_save_and_test`
- `self_check`
- `product_contract`
- `stable_split_doc_index`
- `semantic_contract`
- `dialect_contract`
- `imports_and_runtime`
- `output_and_fallback_contract`
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
- Gluon is additive. Do not remove Base Set coverage.
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

## self_check

Before reporting success:

- Required docs were viewed through `str_replace_editor view`.
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

- `base_shared_extension` -> `10_search_policies.md`,
  `### Search policy: base_shared_extension`
- `evidence_anchored_composition` -> `10_search_policies.md`,
  `### Search policy: evidence_anchored_composition`
- `dialect_contract_metadata` -> `10_search_policies.md`,
  `### Search policy: dialect_contract_metadata`

Detailed references:

- API surface, concrete snippets, quick patterns, compatibility checklist, and
  debug order -> `50_api_reference.md`
- product/runtime context, writing model, layout/sync/descriptor mental model,
  NVIDIA/AMD family differences, real aiter patterns, kernel-family
  optimization paths, source-first triggers, repo-local notes, benchmark rules,
  and anti-patterns -> `60_real_patterns.md`
- Representative schematic examples -> `40_examples.md`
- Residual long-form details not covered by the routed primary docs ->
  `70_backup_details.md`

Task routing:

| Task need or visible signal | Must read |
| --- | --- |
| Any Triton-Gluon task | `00_always_read.md`, `product_contract`, `semantic_contract`, `output_and_fallback_contract` |
| Planning Base/Shared/Extension/Hybrid tasks | `10_search_policies.md`, `### Search policy: base_shared_extension`, `### Search policy: dialect_contract_metadata` |
| Later-round composition from prior results | `10_search_policies.md`, `### Search policy: evidence_anchored_composition` |
| `plain_triton -> amd_gluon` | `00_always_read.md`, `### Trait: dialect_plain_triton`; then relevant traits in `20_component_traits.md` |
| `nv_gluon -> amd_gluon` | `00_always_read.md`, `### Trait: dialect_nv_gluon`; `60_real_patterns.md`, `nvidia_amd_family_differences`; optionally `40_examples.md`, `nv_gluon_to_amd_gluon_translation` |
| existing `amd_gluon` input | `00_always_read.md`, `### Trait: dialect_amd_gluon`; `30_architecture_notes.md`, `### Trait: operator_support_sensitive` |
| `tl.arange`, block sizes, layout, masks, broadcasts, `[:, None]`, `expand_dims` | `20_component_traits.md`, `### Trait: layout_basic` and `### Trait: layout_slice_broadcast` |
| `tl.load` / `tl.store`, buffer ops, memory-bound path | `20_component_traits.md`, `### Trait: memory_generic` and/or `### Trait: memory_amd_buffer` |
| shared memory, swizzle, async, descriptor, `tdm`, cluster, scheduler hints | `20_component_traits.md`, `### Trait: memory_shared_async_descriptor`; `50_api_reference.md`, `shared_memory_synchronization_cluster` and `descriptor_and_tensor_memory_surface` |
| `tl.dot`, `tl.dot_scaled`, MFMA, WMMA, FP8/FP4/scales | `20_component_traits.md`, `### Trait: matrix_dot`, `### Trait: matrix_scaled_dot`, and/or `### Trait: matrix_wmma_descriptor`; `50_api_reference.md`, `amd_quick_patterns` |
| target backend, `gfx942`, `gfx950`, `gfx1250`, arch guards | `30_architecture_notes.md`, target section plus `runtime_target_resolution`; `60_real_patterns.md`, `nvidia_amd_family_differences` |
| Triton version, `instr_shape`, JIT/AOT, prebuilt kernels | `30_architecture_notes.md`, `### Trait: version_sensitive` and `### Trait: execution_jit_aot_sensitive`; `50_api_reference.md`, `version_and_compatibility_checklist` |
| multi-shape or bucketed benchmark | `20_component_traits.md`, relevant `shape_coverage_*` trait plus `### Trait: shape_layout_constexpr_risk` / `### Trait: shape_dispatch_required` |
| concrete code skeleton or exact API call needed | `50_api_reference.md`; do not invent API names from memory |
| real attention/decode/GEMM/preshuffled/gfx1250 pattern | `60_real_patterns.md`, `real_patterns_from_aiter` and `optimization_paths_by_kernel_family` |
| source shows `DistributedLinearLayout`, `PartitionedSharedLayout`, host `TensorDescriptor`, nested 3D/5D layouts, unshuffle transforms, JIT/AOT package gates | `60_real_patterns.md`, `source_first_triggers`; then read operator-local source before editing |
| one schematic example would help | at most one relevant section in `40_examples.md` after reading the trait/policy file above |
| routed split docs still lack a needed nuance or rationale | `70_backup_details.md`; report the missing route if still unresolved |

If a task does not match any route above, stop and read `00_always_read.md`
again plus the closest trait/policy file. If the routed primary docs lack the
required detail, read `70_backup_details.md` and report the missing split-doc
route if still unresolved.

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
- Generate an early minimal `amd_gluon` viability task only after the Base Set
  quota is preserved.
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
search_set: base | shared | extension
required_output_dialect: plain_triton | amd_gluon | mixed | any
```

Rules:

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
- Runtime, layout, architecture, real aiter pattern, or benchmark nuance needed:
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
- Adding MFMA when the source has no matrix trait.
- Starting with descriptor, async, persistent, scheduler, atomics, or work
  stealing before a simpler AMD Gluon candidate passes correctness.
- Claiming a Gluon task succeeded when the patch is actually pure Triton
  fallback.
