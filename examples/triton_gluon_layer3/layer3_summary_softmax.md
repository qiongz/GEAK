# Layer3 Summary (fused_softmax sample)

## Outcome
- The second fixed Layer3 sample succeeded on `gfx942`.
- The required gates all pass in the fixed docker runtime: compile, correctness, profile, benchmark, full-benchmark, `GEAK_HARNESS_ONLY=1`, and full preprocess.
- This sample stays within the locked scope of `02-fused-softmax.py` and only covers:
  - `softmax_kernel`
  - `softmax`
  - tutorial correctness / benchmark intent

## Lift notes
- source_tutorial: `/apps/qiongzhu/triton/python/tutorials/02-fused-softmax.py`
- source_symbols:
  - `softmax_kernel`
  - `softmax`
  - tutorial correctness / benchmark intent
- target_arch: `gfx942`
- kept_triton_semantics:
  - row-wise reduction semantics
  - `max`-shift numerical stabilization
  - `exp` / `sum` normalization
  - irregular-column mask / padding semantics
  - PyTorch reference validation
- explicit_gluon_layout_decisions:
  - chose an explicit wave64 row layout based on the next power-of-two row width
  - bounded the launcher to one row per program instead of replaying the tutorial's full occupancy heuristic
- amd_specific_parts_added_or_changed:
  - lowered row reads to `ttgl.amd.cdna3.buffer_load`
  - lowered writes to `ttgl.amd.cdna3.buffer_store`
  - added the same HIP Gluon runtime compatibility shim already needed by Layer3 on this container
  - kept `--profile` inputs on CPU until the final `.to("cuda")`
- plain_triton_parts_not_carried_verbatim:
  - full occupancy warmup / register heuristic
  - top-level demo execution
  - plot / notebook-style presentation flow
- known_non_parity:
  - this sample proves a bounded direct lift, not a generic reduction compiler
  - no shared-memory async-copy or cross-CTA reduction claim
  - no matmul / descriptor-path coverage claim

## Validation
- compile:
  - passed on `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`
  - compile case: `[1024, 781]`
- correctness:
  - passed on `[256, 97]`
  - passed on `[512, 257]`
  - passed on `[2048, 1024]`
  - passed on `[4096, 1536]`
- profile:
  - passed on `[1024, 781]`
  - Metrix captured only `softmax_kernel`
  - primary kernel duration: `13.621 us`
  - primary kernel duration floor: `13.14 us`
  - bottleneck classification: `latency`
- benchmark:
  - passed on `[256, 97]`
  - passed on `[1024, 781]`
  - passed on `[4096, 1536]`
  - aggregate latency: `0.0237 ms`
  - aggregate throughput: `511.3628 GB/s`
- full benchmark:
  - passed on `[256, 97]`
  - passed on `[512, 257]`
  - passed on `[1024, 781]`
  - passed on `[2048, 1024]`
  - passed on `[4096, 1536]`
  - aggregate latency: `0.0302 ms`
  - aggregate throughput: `315.7986 GB/s`
- preprocess baseline:
  - `benchmark_baseline.txt` aggregate latency: `0.0207 ms`
  - `benchmark_baseline.txt` aggregate throughput: `478.3987 GB/s`

## GEAK-native closure
- harness-only preprocess passed and wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_harness_only/CODEBASE_CONTEXT.md`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_harness_only/discovery.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_harness_only/harness_results.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_harness_only/testcase_selection.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_harness_only/resolved.json`
- full preprocess additionally wrote:
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full/COMMANDMENT.md`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full/benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full/full_benchmark_baseline.txt`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full/profile.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full/baseline_metrics.json`
  - `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/geak_preprocess_softmax_full/harness_path.txt`
- the current runtime path only needed `LLM_GATEWAY_KEY`; `AMD_LLM_API_KEY` was not required for this sample
- the current GEAK harness-driven path still canonicalizes `benchmark_baseline.txt` to the full-benchmark output when a full-benchmark baseline is present

## Interpretation
- This run proves that GEAK can lift a plain Triton row-wise reduction tutorial into an AMD Gluon sample on the fixed `gfx942` runtime and still close the deterministic harness and preprocess loop already used by Layers 1, 2, and the first Layer 3 sample.
- The successful path depends on three concrete constraints:
  - explicit wave64 row layout on `gfx942`
  - row-local reduction using `ttgl.max` and `ttgl.sum`
  - CPU-side input creation in `--profile` so Metrix stays focused on `softmax_kernel`
- The authoring `CODEBASE_CONTEXT.md` still comes from the full `/apps/qiongzhu/GEAK` root and therefore is not blind-eval evidence.
- No changes were required in `preprocessor.py`, `dispatch.py`, or `task_generator.py` main logic.
