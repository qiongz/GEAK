# Layer2 Summary (small_mma sample)

## Outcome
- The third fixed Layer2 sample succeeded on `gfx942`.
- The required gates now all pass in the fixed docker runtime: compile, correctness, one direct benchmark path, full benchmark, harness-only preprocess, and full preprocess.
- This sample stays within the locked scope of `05-wgmma.py` and only covers:
  - `small_mma_kernel`
  - `small_mma`
  - `test_small_mma`

## Translation notes
- source_tutorial: `/apps/qiongzhu/triton/python/tutorials/gluon/05-wgmma.py`
- source_symbols:
  - `small_mma_kernel`
  - `small_mma`
  - `test_small_mma`
- target_arch: `gfx942`
- kept_common_gluon_parts:
  - `@gluon.jit`
  - host-side `small_mma(...)` entry shape and correctness interface
  - blocked tile indexing with `BlockedLayout` and `SliceLayout`
  - `DotOperandLayout`-driven operand lowering
  - matrix multiply plus add semantics `A @ B + C -> D`
  - PyTorch reference validation
- amd_specific_parts_added_or_changed:
  - rewrote the kernel around `ttgl.amd.AMDMFMALayout` and `ttgl.amd.cdna3.mfma`
  - replaced descriptor/TMA movement with direct `ttgl.amd.cdna3.buffer_load` and `buffer_store`
  - retuned the blocked layout to a wave64-compatible topology for `gfx942`
  - kept `num_warps` live and threaded it into both the blocked layout and the MFMA layout
  - used source-tree Triton/Gluon APIs through docker `python` plus explicit `PYTHONPATH`
- explicit_compatibility_downgrades:
  - `LHS_IN_REG` is preserved as a source-compatibility parameter but intentionally lowered to a validated no-op on AMD
  - `INSTR_SHAPE_N` is preserved as a source-compatibility parameter but intentionally lowered to a validated compatibility knob; the actual AMD execution uses fixed CDNA3 MFMA `instr_shape=[32, 32, 8]`
- nv_only_parts_removed:
  - `TensorDescriptor`
  - `tma`
  - `mbarrier`
  - `warpgroup_mma`
  - `warpgroup_mma_wait`
  - `fence_async_shared`
- known_non_parity:
  - this sample proves matmul semantic preservation, not WGMMA/TMA feature parity
  - there is no AMD same-name equivalent for NVIDIA warpgroup MMA in this sample
  - no claim of parity for `tcgen05_*`, `clc`, `gfx1250`, or `cdna4`

## Validation
- compile:
  - passed on `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
  - compile case: `[64, 32, 32, false, 16, 4]`
- correctness:
  - passed on `[64, 32, 32, false, 16, 4]`
  - passed on `[64, 256, 128, true, 64, 8]`
- profile:
  - passed on `[64, 32, 32, false, 16, 4]`
  - Metrix captured `small_mma_kernel` plus `__amd_rocclr_copyBuffer`
  - primary compute kernel duration: `4.714 us`
  - auxiliary copy kernel duration: `2.995 us`
  - aggregate profiler duration floor: `7.624 us`
  - bottleneck classification: `latency`
- benchmark:
  - passed on `[64, 256, 128, false, 64, 4]`
  - latency: `0.0181 ms`
  - throughput: `0.2328 TFLOPS`
- full benchmark:
  - passed on `[64, 32, 32, false, 16, 4]`
  - passed on `[64, 256, 128, false, 64, 4]`
  - aggregate latency: `0.0126 ms`
  - aggregate throughput: `0.1436 TFLOPS`
- preprocess baseline:
  - `benchmark_baseline.txt` aggregate latency: `0.0132 ms`
  - `benchmark_baseline.txt` aggregate throughput: `0.1458 TFLOPS`

## GEAK-native closure
- harness-only preprocess passed and wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only/resolved.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only/discovery.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only/harness_results.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only/testcase_selection.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only/CODEBASE_CONTEXT.md`
- full preprocess passed and additionally wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full/COMMANDMENT.md`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full/benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full/full_benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full/profile.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full/baseline_metrics.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full/harness_path.txt`
- the current GEAK harness-driven path canonicalizes both `benchmark_baseline.txt` and `full_benchmark_baseline.txt` to the full-benchmark output for this sample
- the generated `COMMANDMENT.md` also uses `--full-benchmark` in both the `BENCHMARK` and `FULL_BENCHMARK` sections

## Interpretation
- This sample is not blocked by Gluon API direction anymore; the remaining issues were implementation details, and they are now resolved for the current fixed scope.
- The successful path depends on two concrete constraints: source-tree Triton/Gluon must be first in `PYTHONPATH`, and the AMD blocked layout must be wave64-valid.
- It now also proves that the MFMA rewrite can be wrapped into the same GEAK-native preprocess closure used by the first sample.
- This sample still does not prove WGMMA/TMA feature parity, Arena readiness, task-validator readiness, CI readiness, or any `06+` tutorial feature support.
