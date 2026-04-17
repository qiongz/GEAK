# Triton-Gluon MI3xx Baseline

This document records the GEAK-side integration points for the `triton-gluon` MI3xx/gfx942 baseline workflow.

## Repo assets
- Skill: `skills/triton-gluon-mi3xx/SKILL.md`
- Layer 1 scope: `docs/triton_gluon_layer1_scope.md`
- API quick reference: `docs/triton_gluon_api_quick_reference.md`
- Writing guide: `docs/triton_gluon_writing_guide.md`
- Examples: `examples/triton_gluon_mi3xx/`
- Manifest template: `examples/triton_gluon_mi3xx/run_manifest_template.md`

## Document split
- `skills/triton-gluon-mi3xx/SKILL.md` now focuses on reusable Gluon syntax, architecture mapping, and anti-patterns.
- This document keeps the current-repo defaults for the `/apps/qiongzhu/triton` MI3xx baseline flow.

## GEAK integration points
- Harness prompt guidance lives in `src/minisweagent/run/preprocess/config/mini_unit_test_agent.yaml`.
- Triton task generation enables skills in `src/minisweagent/agents/heterogeneous/task_generator.py`.
- Generated heterogeneous task files carry `kernel_type` and `use_skills` metadata.
- Task execution rehydrates `use_skills` from task metadata in `src/minisweagent/run/dispatch.py`.
- Current policy is intentionally broad: the heterogeneous Triton path may enable skills for Triton tasks in general. The actual experiment boundary is then constrained by task input and test selection, e.g. plain Triton, Triton + NVIDIA Gluon, or Triton + AMD Gluon.

## P0 defaults
- GEAK root: `/apps/qiongzhu/GEAK`
- Triton root: `/apps/qiongzhu/triton`
- Branch naming: `feature/triton-gluon-mi3xx-baseline`
- Runtime preference: one fixed container for the whole P0 cycle
- Target preference: `gfx942`

## Canary order
1. `test_inline_with_amdgpu_dialect`
2. `test_buffer_load_store`
3. `test_amd_mfma`

## Gate order
1. Runtime/path preflight
2. Canary smoke
3. `GEAK_HARNESS_ONLY=1`
4. `--correctness`
5. small `--benchmark`
6. `--full-benchmark`

Do not move to the next gate when the current one is red.
