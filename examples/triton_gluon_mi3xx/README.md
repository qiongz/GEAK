# Triton-Gluon MI3xx Baseline

This directory holds the repo-local assets for the `feature/triton-gluon-mi3xx-baseline` workflow. It is meant to keep the first execution reproducible without relying on chat history.

## Fixed paths
- `GEAK` root: `/apps/qiongzhu/GEAK`
- `triton` root: `/apps/qiongzhu/triton`
- default branch/workstream: `feature/triton-gluon-mi3xx-baseline`
- default target: `gfx942`

## Runtime default
- Prefer one fixed container for the whole P0 cycle.
- Recommended image:
  - `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0`
- Normalize the branch name for the container name:

```bash
BRANCH_NAME="feature/triton-gluon-mi3xx-baseline"
CONTAINER_NAME="${BRANCH_NAME//\//-}"
```

## Create the fixed container

```bash
BRANCH_NAME="feature/triton-gluon-mi3xx-baseline"
CONTAINER_NAME="${BRANCH_NAME//\//-}"
ROCM_PYTORCH_IMAGE="${ROCM_PYTORCH_IMAGE:-rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  -v /apps/qiongzhu:/apps/qiongzhu \
  -v "${HOME}/.triton:${HOME}/.triton" \
  -v "${HOME}/.ccache:${HOME}/.ccache" \
  -e GEAK_ROOT=/apps/qiongzhu/GEAK \
  -e AMD_LLM_API_KEY="${AMD_LLM_API_KEY}" \
  -e TRITON_HOME="${TRITON_HOME:-$HOME/.triton}" \
  -e TRITON_BUILD_WITH_CLANG_LLD=true \
  -e TRITON_BUILD_WITH_CCACHE=true \
  -e MAX_JOBS="${MAX_JOBS:-8}" \
  -w /apps/qiongzhu/triton \
  "${ROCM_PYTORCH_IMAGE}" \
  sleep infinity
```

## Preflight

```bash
docker exec "${CONTAINER_NAME}" bash -lc '
set -euo pipefail
python3 --version
test -d /apps/qiongzhu/GEAK
test -d /apps/qiongzhu/triton
cd /apps/qiongzhu/triton
git rev-parse --short HEAD
python3 - <<'"'"'PY'"'"'
import os
print("GEAK_ROOT", os.environ.get("GEAK_ROOT"))
print("AMD_LLM_API_KEY_SET", bool(os.environ.get("AMD_LLM_API_KEY")))
print("TRITON_HOME", os.environ.get("TRITON_HOME"))
PY
'
```

## One-time bootstrap

```bash
docker exec "${CONTAINER_NAME}" bash -lc '
set -euo pipefail
export GEAK_ROOT=/apps/qiongzhu/GEAK
export TRITON_HOME="${TRITON_HOME:-$HOME/.triton}"
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_WITH_CCACHE=true
export MAX_JOBS="${MAX_JOBS:-8}"
cd /apps/qiongzhu/triton
python3 -m pip install -r python/requirements.txt
python3 -m pip install -r python/test-requirements.txt
python3 -m pip install -i https://test.pypi.org/simple/ hip-python
make dev-install
'
```

## Canary order
1. `test_inline_with_amdgpu_dialect`
2. `test_buffer_load_store`
3. `test_amd_mfma`

Run only the first one until it passes.

```bash
docker exec "${CONTAINER_NAME}" bash -lc '
set -euo pipefail
cd /apps/qiongzhu/triton
python3 -m pytest -x -s python/test/gluon/test_core.py -k "test_inline_with_amdgpu_dialect"
'
```

## Gate sequence

### 1. Harness-only

```bash
cd /apps/qiongzhu/GEAK && GEAK_HARNESS_ONLY=1 python3 -m minisweagent.run.preprocess.preprocessor "<kernel_or_task_path>" --repo /apps/qiongzhu/triton --output_dir "<phase0_output_dir>"
```

### 2. Correctness

```bash
cd /apps/qiongzhu/triton && python3 <resolved_harness_path> --correctness
```

### 3. Small benchmark

```bash
cd /apps/qiongzhu/triton && GEAK_BENCHMARK_ITERATIONS=5 python3 <resolved_harness_path> --benchmark
```

### 4. Full benchmark

```bash
cd /apps/qiongzhu/triton && GEAK_BENCHMARK_ITERATIONS="${GEAK_EVAL_BENCHMARK_ITERATIONS:-30}" python3 <resolved_harness_path> --full-benchmark
```

## Required artefacts
- `run_manifest.md` derived from `run_manifest_template.md`
- `harness_shapes_source.txt`
- `benchmark_baseline.txt`
- a benchmark output sample containing `GEAK_RESULT_LATENCY_MS`
- a `Known Bad / Do Not Repeat` section that records failed environment or harness attempts
