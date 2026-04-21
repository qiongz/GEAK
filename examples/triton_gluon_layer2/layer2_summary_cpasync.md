# Layer2 Summary (cpasync sample)

## Outcome
- The second fixed Layer2 sample succeeded on `gfx942`.
- The required gates now all pass in the fixed docker runtime: compile, correctness, one direct benchmark path, full benchmark, harness-only preprocess, and full preprocess.
- This sample stays within the locked scope of `03-async-copy.py` and only covers:
  - `elementwise_add_cpasync_kernel`
  - `elementwise_add_cpasync`
  - `test_elementwise_add_cpasync`

## Translation notes
- source_tutorial: `/apps/qiongzhu/triton/python/tutorials/gluon/03-async-copy.py`
- source_symbols:
  - `elementwise_add_cpasync_kernel`
  - `elementwise_add_cpasync`
  - `test_elementwise_add_cpasync`
- target_arch: `gfx942`
- kept_common_gluon_parts:
  - `@gluon.jit`
  - `program_id(0)` launcher structure
  - row-by-row program decomposition over `xnumel`
  - column blocking over `YBLOCK`
  - `BlockedLayout` plus `SliceLayout` indexing style
  - masked `load` / `store`
  - host-side `grid = (triton.cdiv(xnumel, XBLOCK),)`
  - PyTorch reference check `a + b`
- changed_layout_parts:
  - retuned the register layout from NVIDIA wave32 assumptions to AMD wave64:
    - `threads_per_warp`: `[1, 32] -> [1, 64]`
    - `warps_per_cta`: `[1, 4] -> [1, 2]`
  - kept the source-style `XBLOCK=32`, `YBLOCK=64` defaults while moving to a wave64 launch
- changed_runtime_or_device_assumptions:
  - retained the source-level `smem_layout` launcher parameter for translation continuity
  - added the same localized HIP/Gluon compatibility shim used by the first sample
  - replaced the explicit NVIDIA async-copy path with a synchronous `gfx942` fallback
- nv_only_parts_removed:
  - `cp.async_copy_global_to_shared`
  - `commit_group`
  - `wait_group`
  - any claim of copy/compute overlap
  - the shared-memory async pipeline itself
- amd_specific_parts_added:
  - `gfx942` wave64 layout selection
  - synchronous common-Gluon load/store fallback under the original `elementwise_add_cpasync*` symbol names
- known_non_parity:
  - there is no same-name AMD `cp.async` path in this sample
  - the AMD translation preserves output semantics, not async-copy overlap semantics
  - no claim of parity for `tma`, `TensorDescriptor`, `warpgroup_mma`, `tcgen05_*`, or `clc`

## Validation
- compile:
  - passed on `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
  - compile shape: `[64, 128]`
- correctness:
  - passed on `[65, 129]`
  - passed on `[1000, 2000]`
- profile:
  - passed on `[128, 256]`
  - Metrix top kernel: `elementwise_add_cpasync_kernel`
  - profiler duration floor: `20.951 us`
  - bottleneck classification: `latency`
- benchmark:
  - passed on `[2048, 4096]`
  - latency: `0.2555 ms`
  - throughput: `393.9238 GB/s`
- full benchmark:
  - passed on `[512, 1024]`
  - passed on `[1024, 2048]`
  - passed on `[2048, 4096]`
  - aggregate latency: `0.1516 ms`
  - aggregate throughput: `226.7729 GB/s`
- preprocess baseline:
  - `benchmark_baseline.txt` aggregate latency: `0.1529 ms`
  - `benchmark_baseline.txt` aggregate throughput: `224.7703 GB/s`

## GEAK-native closure
- harness-only preprocess passed and wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only/resolved.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only/discovery.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only/harness_results.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only/testcase_selection.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only/CODEBASE_CONTEXT.md`
- full preprocess passed and additionally wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full/COMMANDMENT.md`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full/benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full/full_benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full/profile.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full/baseline_metrics.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full/harness_path.txt`
- the current GEAK harness-driven path canonicalizes both `benchmark_baseline.txt` and `full_benchmark_baseline.txt` to the full-benchmark output for this sample
- the generated `COMMANDMENT.md` also uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections

## Interpretation
- This sample proves that an explicit NVIDIA `cp.async` tutorial fragment can be downgraded into a runnable, correctness-preserving AMD `gfx942` path and can be wrapped into the same GEAK-native preprocess closure used by the first sample.
- It does not prove NVIDIA async-copy parity on AMD.
- It does not prove shared-memory overlap, copy/compute pipelining, or any stronger AMD async movement feature.
- It also does not prove Arena readiness, task-validator readiness, CI readiness, or `04+` tutorial feature support.
