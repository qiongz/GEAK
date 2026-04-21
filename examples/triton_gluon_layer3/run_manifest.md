# Triton-Gluon Layer3 Run Manifest (vector_add sample)

## Identity
- run_id: layer3-vector-add-gfx942-20260420
- date: 2026-04-20T09:40:02+00:00
- operator_or_agent: Cursor GPT-5.4
- purpose: first fixed Layer3 direct-lift sample for `01-vector-add.py`

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
- model_access_note: full preprocess succeeded with `LLM_GATEWAY_KEY` only; `AMD_LLM_API_KEY` was not required on this runtime path

## Scope
- source_file: `/apps/qiongzhu/triton/python/tutorials/01-vector-add.py`
- allowed_symbols:
  - `add_kernel`
  - `add`
  - tutorial correctness / benchmark intent
- frozen_plain_triton_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/01_vector_add_triton.py`
- amd_lift_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/01_vector_add_amd.py`
- harness_file: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/test_vector_add_layer3.py`
- excluded_first_round_samples:
  - `/apps/qiongzhu/triton/python/tutorials/02-fused-softmax.py`
  - `/apps/qiongzhu/triton/python/tutorials/03-matrix-multiplication.py`

## Stage Gate Status
- phase0_preflight: passed
- compile: passed
- correctness: passed
- profile: passed
- benchmark_small: passed
- benchmark_full: passed
- preprocess_harness_only: passed
- preprocess_full: passed

## GEAK Preprocess Outputs
- harness_only_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_harness_only`
- full_preprocess_output_dir: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full`
- commandment_path: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full/COMMANDMENT.md`
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
- compile_case: `[98432]`
- correctness_cases:
  - `[4096]`
  - `[98432]`
  - `[1048576]`
  - `[4194304]`
- profile_case: `[262144]`
- benchmark_cases:
  - `[4096]`
  - `[262144]`
  - `[4194304]`
- benchmark_aggregate_latency_ms: `0.0157`
- benchmark_aggregate_gbps: `688.9354`
- full_benchmark_cases:
  - `[4096]`
  - `[98432]`
  - `[262144]`
  - `[1048576]`
  - `[4194304]`
- full_benchmark_aggregate_latency_ms: `0.0133`
- full_benchmark_aggregate_gbps: `685.6220`
- preprocess_canonical_baseline_latency_ms: `0.0136`
- preprocess_canonical_baseline_gbps: `658.3114`
- profiler_primary_kernel_name: `add_kernel`
- profiler_primary_kernel_duration_us: `5.435`
- profiler_primary_kernel_duration_us_min: `4.887`
- profiler_primary_kernel_duration_us_max: `6.369`
- profiler_bottleneck: `latency`
- profiler_gpu_info_detected: `false`

## Lift Strategy
- source_path_kind: plain Triton tutorial fragment
- target_strategy: preserve the one-dimensional launcher, offsets, and mask semantics while rewriting the memory path to AMD CDNA3 buffer load/store with an explicit wave64 layout on `gfx942`
- kept_triton_semantics:
  - host-side one-dimensional `grid`
  - `block_start + offsets` indexing pattern
  - bounds mask semantics
  - elementwise `x + y` compute semantics
  - PyTorch reference validation
- explicit_gluon_layout_decisions:
  - `BlockedLayout(size_per_thread=[4], threads_per_warp=[64], warps_per_cta=[4], order=[0])` for the default `BLOCK_SIZE=1024`
  - expose `num_warps` as an explicit host-side knob rather than leaving warp topology implicit
- amd_specific_parts_added_or_changed:
  - `ttgl.amd.cdna3.buffer_load`
  - `ttgl.amd.cdna3.buffer_store`
  - HIP Gluon runtime compatibility shim for the current container runtime
  - deterministic profile path that builds inputs on CPU before `.to("cuda")`
- plain_triton_parts_not_carried_verbatim:
  - top-level demo prints
  - `perf_report` decorator wiring
  - plot / notebook-style presentation flow
- known_non_parity:
  - no claim of line-by-line Triton-to-Gluon equivalence
  - no reduction, softmax, matmul, shared-memory, or async-copy coverage
  - no claim of `mfma`, `wmma`, `tma`, or `tdm` parity

## Known Good
- compile_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer3/test_vector_add_layer3.py --mode compile'`
- correctness_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer3/test_vector_add_layer3.py --correctness'`
- profile_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer3/test_vector_add_layer3.py --profile'`
- benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer3/test_vector_add_layer3.py --benchmark'`
- full_benchmark_command: `sudo docker exec "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && export PYTHONPATH=/apps/qiongzhu/triton/python:/apps/qiongzhu/GEAK${PYTHONPATH:+:$PYTHONPATH} && python examples/triton_gluon_layer3/test_vector_add_layer3.py --full-benchmark'`
- harness_only_preprocess_command: `sudo docker exec -e HOME=/apps/qiongzhu -e LLM_GATEWAY_KEY=<runtime> -e GEAK_ROOT=/apps/qiongzhu/GEAK -e TRITON_HOME=/apps/qiongzhu/.triton -e CCACHE_DIR=/apps/qiongzhu/.ccache -e PIP_CACHE_DIR=/apps/qiongzhu/.pip-cache -e PYTHONPATH=/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && exec setpriv --reuid=100352 --regid=100352 --groups=100352,44,110 env GEAK_HARNESS_ONLY=1 python -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer3/01_vector_add_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer3/test_vector_add_layer3.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_harness_only'`
- full_preprocess_command: `sudo docker exec -e HOME=/apps/qiongzhu -e LLM_GATEWAY_KEY=<runtime> -e GEAK_ROOT=/apps/qiongzhu/GEAK -e TRITON_HOME=/apps/qiongzhu/.triton -e CCACHE_DIR=/apps/qiongzhu/.ccache -e PIP_CACHE_DIR=/apps/qiongzhu/.pip-cache -e PYTHONPATH=/apps/qiongzhu/GEAK/src:/apps/qiongzhu/triton/python "feature-triton-gluon-mi3xx-baseline" bash --noprofile --norc -lc 'cd /apps/qiongzhu/GEAK && exec setpriv --reuid=100352 --regid=100352 --groups=100352,44,110 python -m minisweagent.run.preprocess.preprocessor /apps/qiongzhu/GEAK/examples/triton_gluon_layer3/01_vector_add_amd.py --repo /apps/qiongzhu/GEAK --harness /apps/qiongzhu/GEAK/examples/triton_gluon_layer3/test_vector_add_layer3.py -o /apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full'`

## Known Bad / Do Not Repeat
- Do not allocate or randomize tensors directly on GPU inside `--profile`; deterministic harness validation rejects that path because it pollutes the profiler trace.
- Do not rely on plain Triton's implicit layout decisions when lifting to AMD Gluon; make the wave64 layout explicit on `gfx942`.
- Do not assume the in-container Python comes from `/apps/qiongzhu/.venvs/triton-gluon-mi3xx/bin`; the validated runtime here uses `/opt/venv/bin/python`.
- The current GEAK preprocessor canonicalizes `benchmark_baseline.txt` to the `full-benchmark` output when a full-benchmark baseline is present.
- The generated `COMMANDMENT.md` currently uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections for this harness-driven path.

## Stop Reasons
- repeated_env_failure: false
- repeated_compile_failure: false
- repeated_correctness_failure: false
- sample_switched: false
- needed_second_container: false
- needed_venv_rebuild: false
- needed_preprocessor_main_logic_change: false
