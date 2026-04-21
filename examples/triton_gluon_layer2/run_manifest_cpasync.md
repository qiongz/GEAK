# Triton-Gluon Layer2 Run Manifest (cpasync sample)

## Identity
- run_id: layer2-elementwise-add-cpasync-gfx942-20260420
- date: 2026-04-20T08:39:00+00:00
- operator_or_agent: Cursor GPT-5.4
- purpose: second fixed Layer2 sample for `03-async-copy.py::elementwise_add_cpasync*`

## Repositories
- geak_root: `/apps/qiongzhu/GEAK`
- geak_branch: `feature/triton-gluon-mi3xx-baseline`
- geak_commit: `f8bed180402d39b0a033c8f1ddaa74102211268e`
- triton_root: `/apps/qiongzhu/triton`
- triton_branch: `feature/triton-gluon-mi3xx-baseline`
- triton_commit: `3be1a23e2683067376a8b66fcef6de866bd93704`

## Runtime
- runtime_type: fixed_container
- runtime_name: `feature-triton-gluon-mi3xx-baseline`
- runtime_container_name: `feature-triton-gluon-mi3xx-baseline`
- runtime_container_id: `bb037b8f0bf34aca4d75ed769b8077417d91247253d007514bfb31a44e3d5a61`
- python_command_used: `python`
- python_executable_resolved: `/opt/venv/bin/python`
- python_version: `3.10.12`
- target_detected: `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
- target_expected: `gfx942`
- gpu_name: `AMD Instinct MI308X`
- source_tree_pythonpath: `/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python`

## Scope
- source_file: `/apps/qiongzhu/triton/python/tutorials/gluon/03-async-copy.py`
- allowed_symbols:
  - `elementwise_add_cpasync_kernel`
  - `elementwise_add_cpasync`
  - `test_elementwise_add_cpasync`
- nv_reference_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/03_elementwise_add_cpasync_nv.py`
- amd_translation_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/03_elementwise_add_cpasync_amd.py`
- harness_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py`

## Stage Gate Status
- phase0_preflight: passed
- harness_only: passed
- correctness: passed
- profile: passed
- benchmark_small: passed
- benchmark_full: passed
- preprocess_full: passed

## GEAK Preprocess Outputs
- harness_only_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only`
- full_preprocess_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full`
- commandment_path: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full/COMMANDMENT.md`
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
  - `harness_path.txt`

## Validation Results
- compile_shape: `[64, 128]`
- correctness_shapes:
  - `[65, 129]`
  - `[1000, 2000]`
- profile_shape: `[128, 256]`
- benchmark_shape: `[2048, 4096]`
- benchmark_latency_ms: `0.2555`
- benchmark_gbps: `393.9238`
- full_benchmark_shapes:
  - `[512, 1024]`
  - `[1024, 2048]`
  - `[2048, 4096]`
- full_benchmark_aggregate_latency_ms: `0.1516`
- full_benchmark_aggregate_gbps: `226.7729`
- preprocess_canonical_baseline_latency_ms: `0.1529`
- preprocess_canonical_baseline_gbps: `224.7703`
- profiler_kernel_name: `elementwise_add_cpasync_kernel`
- profiler_kernel_duration_us: `20.951`
- profiler_bottleneck: `latency`

## Translation Strategy
- source_path_kind: explicit NVIDIA `cp.async` tutorial fragment
- target_strategy: semantic downgrade to synchronous `gfx942` implementation
- kept_host_interface:
  - `elementwise_add_cpasync(...)`
  - `smem_layout` launcher parameter retained for source compatibility
- dropped_nv_only_mechanisms:
  - `cp.async_copy_global_to_shared`
  - `commit_group`
  - `wait_group`
  - async shared-memory pipeline behavior
- geak_closure_note:
  - this sample now has the same harness-only and full preprocess closure depth as the first fixed sample, but the AMD execution path remains intentionally synchronous

## Known Good
- compile_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py --mode compile'`
- correctness_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py --correctness'`
- profile_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py --profile'`
- benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py --benchmark'`
- full_benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py --full-benchmark'`
- harness_only_preprocess_command: `sudo docker exec -e HOME=/apps/qiongzhu -e LLM_GATEWAY_KEY=<runtime> -e AMD_LLM_API_KEY=<runtime> -e GEAK_ROOT=/apps/qiongzhu/GEAK -e TRITON_HOME=/apps/qiongzhu/.triton -e CCACHE_DIR=/apps/qiongzhu/.ccache -e PIP_CACHE_DIR=/apps/qiongzhu/.pip-cache -e PYTHONPATH=/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && exec setpriv --reuid=100352 --regid=100352 --groups=100352,44,110 env GEAK_HARNESS_ONLY=1 python -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/03_elementwise_add_cpasync_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only'`
- full_preprocess_command: `sudo docker exec -e HOME=/apps/qiongzhu -e LLM_GATEWAY_KEY=<runtime> -e AMD_LLM_API_KEY=<runtime> -e GEAK_ROOT=/apps/qiongzhu/GEAK -e TRITON_HOME=/apps/qiongzhu/.triton -e CCACHE_DIR=/apps/qiongzhu/.ccache -e PIP_CACHE_DIR=/apps/qiongzhu/.pip-cache -e PYTHONPATH=/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && exec setpriv --reuid=100352 --regid=100352 --groups=100352,44,110 python -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/03_elementwise_add_cpasync_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_cpasync_layer2.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full'`

## Known Bad / Do Not Repeat
- Do not invent an AMD `cp.async` rename on `gfx942`.
- Do not widen this sample into `04+` tutorials, `tma`, `TensorDescriptor`, `warpgroup_mma`, `tcgen05_*`, or `clc`.
- Do not claim async overlap parity from this sample; the current AMD path is intentionally synchronous.
- The fixed-container HIP Gluon compatibility workaround is still required and remains localized inside `03_elementwise_add_cpasync_amd.py`.
- The current GEAK preprocessor canonicalizes `benchmark_baseline.txt` to the `full-benchmark` output when a full-benchmark baseline is present.
- The generated `COMMANDMENT.md` currently uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections for this harness-driven path.

## Stop Reasons
- repeated_env_failure: false
- repeated_compile_failure: false
- repeated_correctness_failure: false
- sample_switched: false
