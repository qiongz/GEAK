# Layer3 Summary (matmul sample)

## Outcome
- The third fixed Layer3 sample succeeded on `gfx942`.
- The required gates all pass in the fixed docker runtime: compile, correctness, profile, benchmark, full-benchmark, `GEAK_HARNESS_ONLY=1`, and full preprocess.
- This sample stays within the locked first-pass scope of `03-matrix-multiplication.py` and only covers:
  - `matmul_kernel`
  - `matmul`
  - tutorial benchmark intent for the `activation=""` fp16 path

## Lift notes
- source_tutorial: `/apps/qiongzhu/triton/python/tutorials/03-matrix-multiplication.py`
- source_symbols:
  - `matmul_kernel`
  - `matmul`
  - tutorial benchmark intent
- target_arch: `gfx942`
- kept_triton_semantics:
  - grouped launch / block scheduling
  - `offs_am` / `offs_bn` / `offs_k` pointer arithmetic
  - masked K-loop semantics
  - fp32 accumulation then fp16 output
  - PyTorch reference validation
- explicit_gluon_layout_decisions:
  - fixed the first-pass config to `64x64x32` with `GROUP_SIZE_M=4`
  - chose a wave64-compatible blocked register layout for `64x64` output tiles
  - lowered `tl.dot` to explicit CDNA3 MFMA via `AMDMFMALayout` and `DotOperandLayout`
- amd_specific_parts_added_or_changed:
  - lowered global reads to `ttgl.amd.cdna3.buffer_load`
  - lowered writes to `ttgl.amd.cdna3.buffer_store`
  - used `ttgl.amd.cdna3.mfma` as the explicit instruction path
  - added the shared HIP Gluon runtime compatibility shim
  - kept `--profile` inputs on CPU until the final `.to("cuda")`
- plain_triton_parts_not_carried_verbatim:
  - full autotune config surface
  - `leaky_relu` activation path
  - FP8 path
  - top-level demo printing / plotting
- known_non_parity:
  - this sample proves a bounded direct lift, not a generic matmul compiler
  - no full autotune parity claim
  - no `mfma` / `wmma` family coverage beyond this one CDNA3 MFMA path
  - no FP8 / persistent / activation-fused coverage claim

## Validation
- compile:
  - passed on `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
  - compile case: `[256, 320, 128]`
- correctness:
  - passed on `[129, 197, 80]`
  - passed on `[192, 128, 96]`
  - passed on `[384, 512, 160]`
  - passed on `[512, 512, 256]`
- profile:
  - passed on `[256, 320, 128]`
  - Metrix captured only `matmul_kernel`
  - primary kernel duration: `10.843 us`
  - primary kernel duration floor: `10.816 us`
  - bottleneck classification: `latency`
- benchmark:
  - passed on `[129, 197, 80]`
  - passed on `[256, 320, 128]`
  - passed on `[512, 512, 256]`
  - aggregate latency: `0.0180 ms`
  - aggregate throughput: `2.7450 TFLOPS`
- full benchmark:
  - passed on `[129, 197, 80]`
  - passed on `[192, 128, 96]`
  - passed on `[256, 320, 128]`
  - passed on `[384, 512, 160]`
  - passed on `[512, 512, 256]`
  - aggregate latency: `0.0175 ms`
  - aggregate throughput: `2.4175 TFLOPS`
- preprocess baseline:
  - `benchmark_baseline.txt` aggregate latency: `0.0161 ms`
  - `benchmark_baseline.txt` aggregate throughput: `2.5210 TFLOPS`

## GEAK-native closure
- harness-only preprocess passed and wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_harness_only/CODEBASE_CONTEXT.md`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_harness_only/discovery.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_harness_only/harness_results.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_harness_only/testcase_selection.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_harness_only/resolved.json`
- full preprocess additionally wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full/COMMANDMENT.md`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full/benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full/full_benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full/profile.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full/baseline_metrics.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_matmul_full/harness_path.txt`
- the current runtime path only needed `LLM_GATEWAY_KEY`; `AMD_LLM_API_KEY` was not required for this sample
- the current GEAK harness-driven path still canonicalizes `benchmark_baseline.txt` to the full-benchmark output when a full-benchmark baseline is present

## Interpretation
- This run proves that GEAK can lift the core tutorial matmul structure into an AMD Gluon sample on the fixed `gfx942` runtime while preserving grouped scheduling, masked K-loop semantics, and a deterministic harness / preprocess closure.
- The successful path depends on three concrete constraints:
  - explicit CDNA3 MFMA lowering for `tl.dot`
  - a fixed `64x64x32` config instead of broad autotune parity
  - CPU-side input creation in `--profile` so Metrix stays focused on `matmul_kernel`
- The authoring `CODEBASE_CONTEXT.md` still comes from the full `/apps/qiongzhu/GEAK` root and therefore is not blind-eval evidence.
- No changes were required in `preprocessor.py`, `dispatch.py`, or `task_generator.py` main logic.
