# Triton-Gluon Run Manifest

## Identity
- run_id:
- date:
- operator_or_agent:
- purpose: triton-gluon mi3xx baseline

## Repositories
- geak_root: /apps/qiongzhu/GEAK
- geak_branch:
- geak_commit:
- triton_root: /apps/qiongzhu/triton
- triton_branch:
- triton_commit:

## Runtime
- runtime_type: local_rocm | fixed_container
- runtime_name:
- runtime_branch_name:
- runtime_container_name:
- runtime_image:
- python_executable:
- python_version:
- target_detected:
- gpu_detect_command:
- target_expected: gfx942

## Paths
- GEAK_ROOT:
- TRITON_HOME:
- GEAK_OUTPUT_DIR:
- GEAK_TESTCASE_CACHE_DIR:
- pip_cache_dir:
- ccache_dir:
- mount_root: /apps/qiongzhu

## Build Flags
- TRITON_BUILD_WITH_CLANG_LLD: true
- TRITON_BUILD_WITH_CCACHE: true
- MAX_JOBS:
- GEAK_EVAL_BENCHMARK_ITERATIONS:

## Canary
- canary_task_name:
- canary_source_file:
- canary_harness_path:
- shape_source_file:
- commandment_path:

## Stage Gate Status
- phase0_preflight: pending
- harness_only: pending
- correctness: pending
- benchmark_small: pending
- benchmark_full: pending

## Known Good
- install_command:
- smoke_test_command:
- benchmark_command:

## Known Bad / Do Not Repeat
- 

## Stop Reasons
- repeated_env_failure:
- repeated_harness_failure:
- target_mismatch:
