# Triton-Gluon Layer3 Run Manifest (matmul sample)

## Identity
- run_id: layer3-matmul-gfx942-20260421
- date: 2026-04-21T03:06:13+00:00
- operator_or_agent: Cursor GPT-5.4
- purpose: third fixed Layer3 direct-lift sample for `03-matrix-multiplication.py`

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
- target_detected: `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
- target_expected: `gfx942`
- runtime_pythonpath: `/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python`
- cache_paths:
  - `/apps/qiongzhu/.triton`
  - `/apps/qiongzhu/.ccache`
  - `/apps/qiongzhu/.pip-cache`
- model_access_note: full preprocess succeeded with `LLM_GATEWAY_KEY` only; `AMD_LLM_API_KEY` was not required on this runtime path

## Scope
- source_file: `/apps/qiongzhu/triton/python/tutorials/03-matrix-multiplication.py`
- allowed_symbols:
  - `matmul_kernel`
  - `matmul`
  - tutorial benchmark intent
- frozen_plain_triton_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/03_matmul_triton.py`
- amd_lift_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/03_matmul_amd.py`
- harness_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/test_matmul_layer3.py`
- bounded_first_pass_exclusions:
  - `activation="leaky_relu"`
  - FP8 inputs
  - full autotune config parity
  - persistent grouped variants beyond one fixed config

## Stage Gate Status
- compile: passed
- correctness: passed
- profile: passed
- benchmark_small: passed
- benchmark_full: passed
- preprocess_harness_only: passed
- preprocess_full: passed

## GEAK Preprocess Outputs
- harness_only_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_harness_only`
- full_preprocess_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full`
- commandment_path: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full/COMMANDMENT.md`
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
- compile_case: `[256, 320, 128]`
- correctness_cases:
  - `[129, 197, 80]`
  - `[192, 128, 96]`
  - `[384, 512, 160]`
  - `[512, 512, 256]`
- profile_case: `[256, 320, 128]`
- benchmark_cases:
  - `[129, 197, 80]`
  - `[256, 320, 128]`
  - `[512, 512, 256]`
- benchmark_aggregate_latency_ms: `0.0180`
- benchmark_aggregate_tflops: `2.7450`
- full_benchmark_cases:
  - `[129, 197, 80]`
  - `[192, 128, 96]`
  - `[256, 320, 128]`
  - `[384, 512, 160]`
  - `[512, 512, 256]`
- full_benchmark_aggregate_latency_ms: `0.0175`
- full_benchmark_aggregate_tflops: `2.4175`
- preprocess_canonical_baseline_latency_ms: `0.0161`
- preprocess_canonical_baseline_tflops: `2.5210`
- profiler_primary_kernel_name: `matmul_kernel`
- profiler_primary_kernel_duration_us: `10.843`
- profiler_primary_kernel_duration_us_min: `10.816`
- profiler_primary_kernel_duration_us_max: `10.857`
- profiler_bottleneck: `latency`
- profiler_gpu_info_detected: `false`

## Lift Strategy
- source_path_kind: plain Triton tutorial fragment
- target_strategy: preserve 2D pointer arithmetic, grouped launch, and masked K-loop semantics while making the `tl.dot` path explicit as CDNA3 MFMA with a bounded `64x64x32` config on `gfx942`
- kept_triton_semantics:
  - grouped block scheduling
  - `offs_am` / `offs_bn` / `offs_k` pointer arithmetic
  - masked K-loop semantics
  - fp32 accumulation then fp16 output
  - PyTorch reference validation
- explicit_gluon_layout_decisions:
  - fixed `BlockedLayout(size_per_thread=[4, 4], threads_per_warp=[4, 16], warps_per_cta=[4, 1], order=[1, 0])`
  - fixed `AMDMFMALayout(version=3, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[4, 1])`
  - bounded host config `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=32`, `GROUP_SIZE_M=4`
- amd_specific_parts_added_or_changed:
  - `ttgl.amd.cdna3.buffer_load`
  - `ttgl.amd.cdna3.buffer_store`
  - `ttgl.amd.cdna3.mfma`
  - explicit `DotOperandLayout` lowering for A/B tiles
  - HIP Gluon runtime compatibility shim
- plain_triton_parts_not_carried_verbatim:
  - full autotune surface
  - `leaky_relu` path
  - FP8 path
  - cuBLAS / rocBLAS parity claims beyond the bounded config stream
- known_non_parity:
  - no generic matmul compiler claim
  - no full autotune parity claim
  - no persistent grouped or FP8 coverage claim

## Known Good
- compile_harness: `python examples/triton_gluon_layer3/test_matmul_layer3.py --mode compile`
- correctness_harness: `python examples/triton_gluon_layer3/test_matmul_layer3.py --correctness`
- profile_harness: `python examples/triton_gluon_layer3/test_matmul_layer3.py --profile`
- benchmark_harness: `python examples/triton_gluon_layer3/test_matmul_layer3.py --benchmark`
- full_benchmark_harness: `python examples/triton_gluon_layer3/test_matmul_layer3.py --full-benchmark`
- harness_only_preprocess: `GEAK_HARNESS_ONLY=1 python -m minisweagent.run.preprocess.preprocessor .../03_matmul_amd.py --repo /apps/qiongzhu/GEAK --harness .../test_matmul_layer3.py -o .../geak_preprocess_matmul_harness_only`
- full_preprocess: `python -m minisweagent.run.preprocess.preprocessor .../03_matmul_amd.py --repo /apps/qiongzhu/GEAK --harness .../test_matmul_layer3.py -o .../geak_preprocess_matmul_full`

## Stop Reasons
- repeated_env_failure: false
- repeated_compile_failure: false
- repeated_correctness_failure: false
- sample_switched: false
- needed_second_container: false
- needed_venv_rebuild: false
- needed_preprocessor_main_logic_change: false
