# Layer3 Summary (vector_add sample)

## Outcome
- The first fixed Layer3 sample succeeded on `gfx942`.
- The required gates all pass in the fixed docker runtime: compile, correctness, benchmark, `run_manifest.md`, and `layer3_summary.md`.
- The recommended closure also passes: deterministic four-mode harness, `GEAK_HARNESS_ONLY=1`, full preprocess, `COMMANDMENT.md`, `benchmark_baseline.txt`, `full_benchmark_baseline.txt`, `profile.json`, and `baseline_metrics.json`.
- This sample stays within the locked scope of `01-vector-add.py` and only covers:
  - `add_kernel`
  - `add`
  - the tutorial's correctness / benchmark intent

## Lift notes
- source_tutorial: `/apps/qiongzhu/triton/python/tutorials/01-vector-add.py`
- source_symbols:
  - `add_kernel`
  - `add`
  - tutorial benchmark intent
- target_arch: `gfx942`
- kept_triton_semantics:
  - host-side one-dimensional grid launch
  - `block_start + offsets` indexing
  - bounds mask semantics
  - elementwise `x + y` compute semantics
  - PyTorch reference validation
- explicit_gluon_layout_decisions:
  - chose an explicit wave64-compatible `BlockedLayout(size_per_thread=[4], threads_per_warp=[64], warps_per_cta=[4], order=[0])` for the default `BLOCK_SIZE=1024`
  - made `num_warps` explicit in the AMD launcher instead of leaving warp topology implicit
- amd_specific_parts_added_or_changed:
  - lowered memory movement to `ttgl.amd.cdna3.buffer_load`
  - lowered stores to `ttgl.amd.cdna3.buffer_store`
  - added a small HIP Gluon runtime compatibility shim for the current container
  - adjusted `--profile` to build inputs on CPU first so Metrix captures the target kernel cleanly
- plain_triton_parts_not_carried_verbatim:
  - top-level tutorial prints
  - `perf_report` decorator wiring
  - plot / notebook-style presentation flow
- known_non_parity:
  - this sample proves a bounded direct lift, not a generic Triton-to-Gluon compiler
  - no reduction, shared-memory, async-copy, or matrix-op coverage
  - no claim of `mfma`, `wmma`, `tma`, `tdm`, or descriptor-path parity

## Validation
- compile:
  - passed on `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
  - compile case: `[98432]`
- correctness:
  - passed on `[4096]`
  - passed on `[98432]`
  - passed on `[1048576]`
  - passed on `[4194304]`
- profile:
  - passed on `[262144]`
  - Metrix now captures only `add_kernel`
  - primary kernel duration: `5.435 us`
  - primary kernel duration floor: `4.887 us`
  - bottleneck classification: `latency`
- benchmark:
  - passed on `[4096]`
  - passed on `[262144]`
  - passed on `[4194304]`
  - aggregate latency: `0.0157 ms`
  - aggregate throughput: `688.9354 GB/s`
- full benchmark:
  - passed on `[4096]`
  - passed on `[98432]`
  - passed on `[262144]`
  - passed on `[1048576]`
  - passed on `[4194304]`
  - aggregate latency: `0.0133 ms`
  - aggregate throughput: `685.6220 GB/s`
- preprocess baseline:
  - `benchmark_baseline.txt` aggregate latency: `0.0136 ms`
  - `benchmark_baseline.txt` aggregate throughput: `658.3114 GB/s`

## GEAK-native closure
- harness-only preprocess passed and wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_harness_only/resolved.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_harness_only/discovery.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_harness_only/harness_results.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_harness_only/testcase_selection.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_harness_only/CODEBASE_CONTEXT.md`
- full preprocess passed and additionally wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full/COMMANDMENT.md`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full/benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full/full_benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full/profile.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full/baseline_metrics.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_vector_add_full/harness_path.txt`
- the current runtime path only needed `LLM_GATEWAY_KEY`; `AMD_LLM_API_KEY` was not required for this sample
- the current GEAK harness-driven path still canonicalizes both `benchmark_baseline.txt` and `full_benchmark_baseline.txt` to the full-benchmark output
- the generated `COMMANDMENT.md` also uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections

## Interpretation
- This run proves that GEAK can lift a plain Triton tutorial into an AMD Gluon sample on the fixed `gfx942` runtime, then close the same deterministic harness and preprocess loop already used by Layers 1 and 2.
- The successful path depends on three concrete constraints:
  - explicit wave64 layout on `gfx942`
  - source-tree Triton/Gluon on `PYTHONPATH`
  - CPU-side input creation in `--profile` so the profiler stays focused on the lifted kernel
- The first round completed entirely in `examples/`, `docs/`, and the Layer3 harness surface. No changes were required in `preprocess.py`, `dispatch.py`, or `task_generator.py` main logic.
- This sample still does not prove a generic compiler, reduction/matmul coverage, Arena readiness, CI readiness, or any target-specific parity beyond this bounded vector-add direct lift.
