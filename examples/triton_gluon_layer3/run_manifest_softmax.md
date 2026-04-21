# Triton-Gluon Layer3 Run Manifest (fused_softmax sample)

## Identity
- run_id: layer3-softmax-gfx942-20260421
- date: 2026-04-21T03:06:13+00:00
- operator_or_agent: Cursor GPT-5.4
- purpose: second fixed Layer3 direct-lift sample for `02-fused-softmax.py`

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
- source_file: `/apps/qiongzhu/triton/python/tutorials/02-fused-softmax.py`
- allowed_symbols:
  - `softmax_kernel`
  - `softmax`
  - tutorial correctness / benchmark intent
- frozen_plain_triton_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/02_fused_softmax_triton.py`
- amd_lift_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/02_fused_softmax_amd.py`
- harness_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/test_fused_softmax_layer3.py`
- excluded_first_round_features:
  - full occupancy parity with the original tutorial host heuristic
  - shared-memory async movement
  - multi-CTA reduction strategies
  - generic reduction compiler claims

## Stage Gate Status
- compile: passed
- correctness: passed
- profile: passed
- benchmark_small: passed
- benchmark_full: passed
- preprocess_harness_only: passed
- preprocess_full: passed

## GEAK Preprocess Outputs
- harness_only_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_harness_only`
- full_preprocess_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full`
- commandment_path: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full/COMMANDMENT.md`
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
- compile_case: `[1024, 781]`
- correctness_cases:
  - `[256, 97]`
  - `[512, 257]`
  - `[2048, 1024]`
  - `[4096, 1536]`
- profile_case: `[1024, 781]`
- benchmark_cases:
  - `[256, 97]`
  - `[1024, 781]`
  - `[4096, 1536]`
- benchmark_aggregate_latency_ms: `0.0237`
- benchmark_aggregate_gbps: `511.3628`
- full_benchmark_cases:
  - `[256, 97]`
  - `[512, 257]`
  - `[1024, 781]`
  - `[2048, 1024]`
  - `[4096, 1536]`
- full_benchmark_aggregate_latency_ms: `0.0302`
- full_benchmark_aggregate_gbps: `315.7986`
- preprocess_canonical_baseline_latency_ms: `0.0207`
- preprocess_canonical_baseline_gbps: `478.3987`
- profiler_primary_kernel_name: `softmax_kernel`
- profiler_primary_kernel_duration_us: `13.621`
- profiler_primary_kernel_duration_us_min: `13.14`
- profiler_primary_kernel_duration_us_max: `13.901`
- profiler_bottleneck: `latency`
- profiler_gpu_info_detected: `false`

## Lift Strategy
- source_path_kind: plain Triton tutorial fragment
- target_strategy: preserve row-wise softmax semantics, masking, and numerical-stability shift while lowering row memory traffic to AMD CDNA3 buffer load/store with an explicit wave64 layout on `gfx942`
- kept_triton_semantics:
  - row-wise normalization intent
  - `tl.max` / `tl.sum` reduction semantics
  - irregular-column mask / padding semantics
  - numerically stable `exp(x - max(x)) / sum(exp(...))`
  - PyTorch reference validation
- explicit_gluon_layout_decisions:
  - bounded one-row-per-program launcher instead of reproducing the full tutorial occupancy heuristic
  - `BlockedLayout(size_per_thread=[block_size / (64 * num_warps)], threads_per_warp=[64], warps_per_cta=[num_warps], order=[0])`
  - explicit `num_warps` selection from the next power-of-two row width
- amd_specific_parts_added_or_changed:
  - `ttgl.amd.cdna3.buffer_load`
  - `ttgl.amd.cdna3.buffer_store`
  - HIP Gluon runtime compatibility shim
  - CPU-side input construction in `--profile`
- plain_triton_parts_not_carried_verbatim:
  - full tutorial occupancy warmup and register-occupancy heuristic
  - top-level demo execution and plotting
  - `naive_softmax` as a benchmark baseline requirement
- known_non_parity:
  - no generic reduction compiler claim
  - no shared-memory async-copy claim
  - no multi-CTA softmax coverage claim

## Known Good
- compile_harness: `python examples/triton_gluon_layer3/test_fused_softmax_layer3.py --mode compile`
- correctness_harness: `python examples/triton_gluon_layer3/test_fused_softmax_layer3.py --correctness`
- profile_harness: `python examples/triton_gluon_layer3/test_fused_softmax_layer3.py --profile`
- benchmark_harness: `python examples/triton_gluon_layer3/test_fused_softmax_layer3.py --benchmark`
- full_benchmark_harness: `python examples/triton_gluon_layer3/test_fused_softmax_layer3.py --full-benchmark`
- harness_only_preprocess: `GEAK_HARNESS_ONLY=1 python -m minisweagent.run.preprocess.preprocessor .../02_fused_softmax_amd.py --repo /apps/qiongzhu/GEAK --harness .../test_fused_softmax_layer3.py -o .../geak_preprocess_softmax_harness_only`
- full_preprocess: `python -m minisweagent.run.preprocess.preprocessor .../02_fused_softmax_amd.py --repo /apps/qiongzhu/GEAK --harness .../test_fused_softmax_layer3.py -o .../geak_preprocess_softmax_full`

## Stop Reasons
- repeated_env_failure: false
- repeated_compile_failure: false
- repeated_correctness_failure: false
- sample_switched: false
- needed_second_container: false
- needed_venv_rebuild: false
- needed_preprocessor_main_logic_change: false
