---
name: triton-gluon-mi3xx-benchmark-safe
description: Use when a Triton-family task explicitly enables the gluon feature with the mi3xx baseline profile and needs benchmark-safe AMD Gluon guidance for hip/gfx942 without any authoring-only truth data.
tier: benchmark_safe
---

# Triton Gluon MI3xx Benchmark-Safe Notes

## When to use
- The task metadata says `gluon_feature_mode != off`.
- The baseline profile is `mi3xx`.
- The target backend is `hip/gfx942` or another AMD MI3xx-class backend.
- You need bounded guidance for AMD Gluon output without relying on example-specific manifests, summaries, or translation playbooks.

## Output contract
- Keep `kernel_type = triton`.
- Treat the input as one of:
  - `plain_triton`
  - `nv_gluon`
  - `amd_gluon`
- Only target these optimized outputs:
  - `plain_triton`
  - `amd_gluon`
- Never introduce a new `nv_gluon` output path.

## Safe AMD Gluon guidance
- Use `@gluon.jit` with explicit layouts when the task allows an AMD Gluon result.
- On MI3xx-class targets, prefer wave64-valid decomposition and layout choices.
- Favor explicit AMD CDNA3-style memory and compute primitives only when they match the kernel's dataflow:
  - `buffer_load`
  - `buffer_store`
  - `mfma`
- Keep layout decisions explicit and coherent:
  - `BlockedLayout`
  - `DotOperandLayout`
  - `convert_layout`
- Preserve masks, indexing semantics, and correctness behavior from the source kernel before changing low-level execution details.

## Hard constraints
- Do not copy benchmark truths, fixed sample manifests, or authoring summaries into the solution.
- Do not assume NVIDIA Gluon APIs map 1:1 onto AMD APIs.
- Do not mix AMD and NVIDIA architecture-specific layout families in one kernel path.
- If the task can be solved well in plain Triton, keep plain Triton as a valid competitor instead of forcing Gluon.
