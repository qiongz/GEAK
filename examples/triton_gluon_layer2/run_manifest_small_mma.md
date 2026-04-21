# Triton-Gluon Layer2 Run Manifest (small_mma sample)

## Identity
- run_id: layer2-small-mma-gfx942-20260420
- date: 2026-04-20T08:39:00+00:00
- operator_or_agent: Cursor GPT-5.4
- purpose: third fixed Layer2 sample for `05-wgmma.py::small_mma*`

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
- source_file: `/apps/qiongzhu/triton/python/tutorials/gluon/05-wgmma.py`
- allowed_symbols:
  - `small_mma_kernel`
  - `small_mma`
  - `test_small_mma`
- nv_reference_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/05_small_mma_nv.py`
- amd_translation_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/05_small_mma_amd.py`
- harness_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_small_mma_layer2.py`

## Stage Gate Status
- phase0_preflight: passed
- harness_only: passed
- correctness: passed
- profile: passed
- benchmark_small: passed
- benchmark_full: passed
- preprocess_full: passed

## GEAK Preprocess Outputs
- harness_only_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only`
- full_preprocess_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full`
- commandment_path: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full/COMMANDMENT.md`
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
- compile_case: `[64, 32, 32, false, 16, 4]`
- correctness_cases:
  - `[64, 32, 32, false, 16, 4]`
  - `[64, 256, 128, true, 64, 8]`
- profile_case: `[64, 32, 32, false, 16, 4]`
- benchmark_case: `[64, 256, 128, false, 64, 4]`
- benchmark_latency_ms: `0.0181`
- benchmark_tflops: `0.2328`
- full_benchmark_cases:
  - `[64, 32, 32, false, 16, 4]`
  - `[64, 256, 128, false, 64, 4]`
- full_benchmark_aggregate_latency_ms: `0.0126`
- full_benchmark_aggregate_tflops: `0.1436`
- preprocess_canonical_baseline_latency_ms: `0.0132`
- preprocess_canonical_baseline_tflops: `0.1458`
- profiler_primary_kernel_name: `small_mma_kernel`
- profiler_primary_kernel_duration_us: `4.714`
- profiler_aux_kernel_name: `__amd_rocclr_copyBuffer`
- profiler_aux_kernel_duration_us: `2.995`
- profiler_aggregate_duration_us: `7.624`
- profiler_bottleneck: `latency`

## Translation Strategy
- source_path_kind: NV WGMMA/TMA-style tutorial fragment
- target_strategy: rewrite to AMD CDNA3 MFMA plus direct buffer load/store on `gfx942`
- kept_host_interface:
  - `small_mma(A, B, C, D, INSTR_SHAPE_N, LHS_IN_REG=False, num_warps=4)`
- explicit_compatibility_downgrades:
  - `LHS_IN_REG` is retained for source compatibility but lowered to a validated no-op on the AMD path
  - `INSTR_SHAPE_N` is retained for source compatibility but lowered to a validated compatibility knob; execution uses fixed CDNA3 MFMA `instr_shape=[32, 32, 8]`
  - `num_warps` remains active and is wired into both `BlockedLayout` and `AMDMFMALayout`
- dropped_nv_only_mechanisms:
  - `TensorDescriptor`
  - `tma`
  - `mbarrier`
  - `warpgroup_mma`
  - `warpgroup_mma_wait`
  - `fence_async_shared`
- geak_closure_note:
  - this sample now has the same harness-only and full preprocess closure depth as the first fixed sample, while keeping the AMD rewrite centered on MFMA semantics rather than NV feature parity

## Known Good
- compile_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_small_mma_layer2.py --mode compile'`
- correctness_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_small_mma_layer2.py --correctness'`
- profile_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_small_mma_layer2.py --profile'`
- benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_small_mma_layer2.py --benchmark'`
- full_benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer2/test_small_mma_layer2.py --full-benchmark'`
- harness_only_preprocess_command: `sudo docker exec -e HOME=/apps/qiongzhu -e LLM_GATEWAY_KEY=<runtime> -e AMD_LLM_API_KEY=<runtime> -e GEAK_ROOT=/apps/qiongzhu/GEAK -e TRITON_HOME=/apps/qiongzhu/.triton -e CCACHE_DIR=/apps/qiongzhu/.ccache -e PIP_CACHE_DIR=/apps/qiongzhu/.pip-cache -e PYTHONPATH=/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && exec setpriv --reuid=100352 --regid=100352 --groups=100352,44,110 env GEAK_HARNESS_ONLY=1 python -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/05_small_mma_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_small_mma_layer2.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only'`
- full_preprocess_command: `sudo docker exec -e HOME=/apps/qiongzhu -e LLM_GATEWAY_KEY=<runtime> -e AMD_LLM_API_KEY=<runtime> -e GEAK_ROOT=/apps/qiongzhu/GEAK -e TRITON_HOME=/apps/qiongzhu/.triton -e CCACHE_DIR=/apps/qiongzhu/.ccache -e PIP_CACHE_DIR=/apps/qiongzhu/.pip-cache -e PYTHONPATH=/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && exec setpriv --reuid=100352 --regid=100352 --groups=100352,44,110 python -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/05_small_mma_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_small_mma_layer2.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full'`

## Known Bad / Do Not Repeat
- Do not use Python `del` inside a `@gluon.jit` kernel; the current Gluon frontend rejects `Delete` AST nodes.
- Do not rely on the container's installed Triton package alone for this sample; keep the source-tree Triton/Gluon modules first in `PYTHONPATH`.
- Do not use wave32-style blocked layouts on `gfx942`; module metadata is validated against wave64 `threads-per-warp = 64`.
- Do not claim parity for `TensorDescriptor`, `tma`, `mbarrier`, `warpgroup_mma`, `tcgen05_*`, `clc`, `gfx1250`, or `cdna4`.
- The current GEAK preprocessor canonicalizes `benchmark_baseline.txt` to the `full-benchmark` output when a full-benchmark baseline is present.
- The generated `COMMANDMENT.md` currently uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections for this harness-driven path.

## Stop Reasons
- repeated_env_failure: false
- repeated_compile_failure: false
- repeated_correctness_failure: false
- sample_switched: false
