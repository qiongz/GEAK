# Triton-Gluon Layer2 Run Manifest

## Identity
- run_id: layer2-elementwise-add-gfx942-20260420
- date: 2026-04-20T03:06:22+00:00
- operator_or_agent: Cursor GPT-5.4
- purpose: first fixed Layer2 sample for `03-async-copy.py::elementwise_add*`

## Repositories
- geak_root: /apps/qiongzhu/GEAK
- geak_branch: `feature/triton-gluon-mi3xx-baseline`
- geak_commit: `f8bed180402d39b0a033c8f1ddaa74102211268e`
- triton_root: /apps/qiongzhu/triton
- triton_branch: `feature/triton-gluon-mi3xx-baseline`
- triton_commit: `3be1a23e2683067376a8b66fcef6de866bd93704`

## Runtime
- runtime_type: fixed_container
- runtime_name: `feature-triton-gluon-mi3xx-baseline`
- runtime_branch_name: `feature/triton-gluon-mi3xx-baseline`
- runtime_container_name: `feature-triton-gluon-mi3xx-baseline`
- runtime_image: unknown
- python_executable: `/opt/venv/bin/python3`
- python_version: `3.10.12`
- target_detected: `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
- gpu_name: `AMD Instinct MI308X`
- gpu_detect_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc '/opt/venv/bin/python3 -c "import torch, triton; ..."'`
- target_expected: `gfx942`

## Paths
- GEAK_ROOT: `/apps/qiongzhu/GEAK`
- TRITON_HOME: `/apps/qiongzhu/.triton`
- GEAK_OUTPUT_DIR: unset
- GEAK_TESTCASE_CACHE_DIR: unset
- pip_cache_dir: `/apps/qiongzhu/.pip-cache`
- ccache_dir: `/apps/qiongzhu/.ccache`
- mount_root: `/apps/qiongzhu`

## Build Flags
- TRITON_BUILD_WITH_CLANG_LLD: unknown
- TRITON_BUILD_WITH_CCACHE: unknown
- MAX_JOBS: unset
- GEAK_EVAL_BENCHMARK_ITERATIONS: unset

## Canary
- canary_task_name: `layer2_elementwise_add_gfx942`
- canary_source_file: `/apps/qiongzhu/triton/python/tutorials/gluon/03-async-copy.py`
- canary_harness_path: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_layer2.py`
- shape_source_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/harness_shapes_source.txt`
- commandment_path: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_full/COMMANDMENT.md`

## Stage Gate Status
- phase0_preflight: passed
- harness_only: passed
- correctness: passed
- profile: passed
- benchmark_small: passed
- benchmark_full: passed
- preprocess_full: passed

## GEAK Preprocess Outputs
- harness_shapes_source_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/harness_shapes_source.txt`
- harness_only_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_harness_only`
- full_preprocess_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_full`
- harness_only_artifacts:
  - `resolved.json`
  - `CODEBASE_CONTEXT.md`
  - `discovery.json`
  - `harness_results.json`
  - `testcase_selection.json`
- full_preprocess_artifacts:
  - `resolved.json`
  - `CODEBASE_CONTEXT.md`
  - `discovery.json`
  - `harness_results.json`
  - `testcase_selection.json`
  - `benchmark_baseline.txt`
  - `full_benchmark_baseline.txt`
  - `profile.json`
  - `baseline_metrics.json`
  - `COMMANDMENT.md`

## Validation Results
- compile_shape: `[64, 128]`
- correctness_shapes:
  - `[65, 129]`
  - `[1000, 2000]`
- profile_shape: `[128, 256]`
- benchmark_shape: `[2048, 4096]`
- benchmark_latency_ms: `0.1357`
- benchmark_gbps: `741.9088`
- full_benchmark_shapes:
  - `[512, 1024]`
  - `[1024, 2048]`
  - `[2048, 4096]`
- preprocess_canonical_baseline_latency_ms: `0.0829`
- preprocess_canonical_baseline_gbps: `414.4287`
- profiler_kernel_name: `elementwise_add_kernel`
- profiler_kernel_duration_us: `12.378`
- profiler_bottleneck: `latency`

## Known Good
- in_place_geak_install: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && /opt/venv/bin/pip install -e "." "fastmcp>=0.2.0" "mcp[cli]>=1.2.0"'`
- correctness_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK/examples/triton_gluon_layer2 && /opt/venv/bin/python3 test_elementwise_add_layer2.py --correctness'`
- profile_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK/examples/triton_gluon_layer2 && /opt/venv/bin/python3 test_elementwise_add_layer2.py --profile'`
- benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK/examples/triton_gluon_layer2 && /opt/venv/bin/python3 test_elementwise_add_layer2.py --benchmark --iterations 5'`
- full_benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK/examples/triton_gluon_layer2 && /opt/venv/bin/python3 test_elementwise_add_layer2.py --full-benchmark --iterations 5'`
- run_harness_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && /opt/venv/bin/python3 -m minisweagent.run.preprocess.run_harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_layer2.py --mode all --repo-root /apps/qiongzhu/GEAK'`
- harness_only_preprocess_command: `sudo docker exec --env LLM_GATEWAY_KEY=<runtime> --env AMD_LLM_API_KEY=<runtime> "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export GEAK_ROOT=/apps/qiongzhu/GEAK && export TRITON_HOME=/apps/qiongzhu/.triton && export PYTHONPATH=/apps/qiongzhu/GEAK/src && GEAK_HARNESS_ONLY=1 /opt/venv/bin/python3 -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/03_elementwise_add_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_layer2.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_harness_only'`
- full_preprocess_command: `sudo docker exec --env LLM_GATEWAY_KEY=<runtime> --env AMD_LLM_API_KEY=<runtime> "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export GEAK_ROOT=/apps/qiongzhu/GEAK && export TRITON_HOME=/apps/qiongzhu/.triton && export PYTHONPATH=/apps/qiongzhu/GEAK/src && /opt/venv/bin/python3 -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/03_elementwise_add_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_layer2.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_full'`

## Known Bad / Do Not Repeat
- The fixed container's installed `triton.experimental.gluon._runtime` still assumes CUDA-style `maxnreg`.
- The same runtime hardcodes `ttg.threads-per-warp = 32`, which is wrong for `gfx942` wave64 validation.
- The workaround is localized in `03_elementwise_add_amd.py`; do not widen it into GEAK main routing as part of this sample.
- The current GEAK preprocessor canonicalizes `benchmark_baseline.txt` to the `full-benchmark` output whenever a full-benchmark baseline is present.
- The generated `COMMANDMENT.md` currently uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections for this harness-driven path.

## Stop Reasons
- repeated_env_failure: false
- repeated_harness_failure: false
- target_mismatch: false
