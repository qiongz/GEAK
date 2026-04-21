# Triton Gluon MI3xx Benchmark-Safe Knowledge

## When to use
- The task metadata enables the Gluon feature (`gluon_feature_mode != off`).
- The baseline profile is `mi3xx`.
- The target backend is an AMD MI3xx-class path such as `hip/gfx942`.
- You need bounded Gluon guidance without relying on sample-specific truth data.

## Output contract
- Keep `kernel_type = triton`.
- Treat the input as one of:
  - `plain_triton`
  - `nv_gluon`
  - `amd_gluon`
- Only produce these optimized outputs:
  - `plain_triton`
  - `amd_gluon`
- Never introduce a new `nv_gluon` output path.

## Safe MI3xx Gluon guidance
- Prefer explicit layout-driven design when an AMD Gluon path is competitive.
- On MI3xx targets, favor wave64-valid decomposition and layout choices.
- Keep layout decisions coherent across load, convert, dot operand, and store stages:
  - `BlockedLayout`
  - `DotOperandLayout`
  - `convert_layout`
- Preserve source-kernel semantics first:
  - indexing
  - masks
  - boundary checks
  - correctness behavior
- Use AMD CDNA3-style primitives only when the kernel dataflow matches them:
  - `buffer_load`
  - `buffer_store`
  - `mfma`
- If the kernel is already best expressed in plain Triton, keep plain Triton as a real competitor.

## Hard constraints
- Do not treat this file as example truth or a fixed recipe.
- Do not copy from layer summaries, run manifests, translation playbooks, or optimization logs.
- Do not assume NVIDIA Gluon APIs map 1:1 onto AMD APIs.
- Do not mix AMD and NVIDIA layout families in one kernel path.
- Do not force Gluon when the task can be solved cleanly in plain Triton.

## Quick review checklist
1. Is the candidate output still within `plain_triton` or `amd_gluon`?
2. Are layout choices explicit and wave64-valid for MI3xx?
3. Did the change preserve masks, indexing, and correctness semantics?
4. Are AMD-specific primitives justified by the kernel dataflow rather than copied from a tutorial?
