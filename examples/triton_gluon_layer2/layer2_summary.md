# Layer2 Summary

## Outcome
- The first fixed Layer2 sample succeeded on `gfx942`.
- Minimum success criteria are met: compile passed, correctness passed, and one benchmark path produced stable latency.
- The sample is now wired into a GEAK-native four-mode harness and preprocess closure.
- This sample stays within the locked scope of `03-async-copy.py` and only covers:
  - `elementwise_add_kernel`
  - `elementwise_add`
  - `test_elementwise_add`

## Translation notes
- source_tutorial: `/apps/qiongzhu/triton/python/tutorials/gluon/03-async-copy.py`
- source_symbols:
  - `elementwise_add_kernel`
  - `elementwise_add`
  - `test_elementwise_add`
- target_arch: `gfx942`
- kept_common_gluon_parts:
  - `@gluon.jit`
  - `program_id(0)` launcher structure
  - row-by-row program decomposition over `xnumel`
  - `BlockedLayout` plus `SliceLayout` indexing style
  - masked `load` / `store`
  - host-side `grid = (triton.cdiv(xnumel, XBLOCK),)`
  - PyTorch reference check `a + b`
- changed_layout_parts:
  - retuned the layout from NVIDIA wave32 assumptions to AMD wave64:
    - `threads_per_warp`: `[1, 32] -> [1, 64]`
    - `warps_per_cta`: `[1, 4] -> [1, 2]`
  - changed the AMD sample defaults to `XBLOCK=32`, `YBLOCK=128`, `num_warps=2`
- changed_runtime_or_device_assumptions:
  - added a localized HIP/Gluon compatibility shim in `03_elementwise_add_amd.py`
  - the shim only compensates for the fixed container's older Gluon runtime, which:
    - accesses CUDA-style `maxnreg` on HIP
    - hardcodes `threads-per-warp = 32`
- nv_only_parts_removed:
  - no `cp.async` imports
  - no `async_copy_global_to_shared`
  - no `commit_group` / `wait_group`
  - no shared-memory async pipeline
- amd_specific_parts_added:
  - `gfx942` wave64 layout selection
  - localized HIP runtime compatibility patch for compile-time metadata
- known_non_parity:
  - NVIDIA `cp.async` path is explicitly not translated in this sample
  - `memcpy_1d_cpasync_kernel` and `elementwise_add_cpasync_kernel` remain out of scope
  - no claim of parity for `tma`, `tdm`, `wgmma`, `tcgen05_*`, or `clc`

## Validation
- compile:
  - passed on `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
  - compile shape: `[64, 128]`
- correctness:
  - passed on `[65, 129]`
  - passed on `[1000, 2000]`
- profile:
  - passed on `[128, 256]`
- benchmark:
  - passed on `[2048, 4096]`
  - latency: `0.1357 ms`
  - throughput: `741.9088 GB/s`
- full benchmark:
  - passed on `[[512, 1024], [1024, 2048], [2048, 4096]]`
  - aggregate latency: `0.0829 ms`
  - aggregate throughput: `414.4287 GB/s`

## GEAK-native closure
- Deterministic harness path:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/test_elementwise_add_layer2.py`
- Shape source path:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/harness_shapes_source.txt`
- Harness-only preprocess passed and wrote:
  - `resolved.json`
  - `CODEBASE_CONTEXT.md`
  - `discovery.json`
  - `harness_results.json`
  - `testcase_selection.json`
- Full preprocess passed and additionally wrote:
  - `benchmark_baseline.txt`
  - `full_benchmark_baseline.txt`
  - `profile.json`
  - `baseline_metrics.json`
  - `COMMANDMENT.md`
- Output directories:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_harness_only`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_full`

## Profiling and baseline notes
- Metrix-backed profiling completed successfully for `elementwise_add_kernel`.
- The profiled kernel duration was `12.378 us`.
- The baseline classifier labeled the kernel as `latency` bottleneck with very low HBM utilization.
- The current GEAK preprocessor canonicalizes `benchmark_baseline.txt` to the `full-benchmark` output when full-benchmark data exists, so in this run:
  - `benchmark_baseline.txt`
  - `full_benchmark_baseline.txt`
  contain the same canonical text.
- The generated `COMMANDMENT.md` also uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections for this harness-driven path. That is current GEAK behavior, not a new Layer2 sample change.

## Boundaries
- This sample proves that a bounded NVIDIA Gluon tutorial fragment can be rewritten into a runnable AMD Gluon fragment on `gfx942`.
- It does not prove NVIDIA async-copy parity on AMD.
- It now proves GEAK-native deterministic harness execution and full preprocess closure for this fixed sample.
- It still does not prove NVIDIA async-copy parity, Arena readiness, task-validator readiness, or CI readiness.
