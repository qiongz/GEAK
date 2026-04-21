# Layer2 Summary (gfx942 mainline)

## Outcome
- The locked `gfx942` Layer2 mainline now covers exactly three fixed samples:
  - `03-async-copy.py::elementwise_add*`
  - `03-async-copy.py::elementwise_add_cpasync*`
  - `05-wgmma.py::small_mma*`
- All three samples now have:
  - compile success
  - correctness success
  - at least one benchmark path success
  - GEAK-native harness-only preprocess output
  - GEAK-native full preprocess output

## Closure Status
- first sample (`elementwise_add*`):
  - harness-only output: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_harness_only`
  - full preprocess output: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_full`
  - per-sample notes: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/run_manifest.md`
  - per-sample summary: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/layer2_summary.md`
- second sample (`elementwise_add_cpasync*`):
  - harness-only output: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_harness_only`
  - full preprocess output: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_cpasync_full`
  - per-sample notes: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/run_manifest_cpasync.md`
  - per-sample summary: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/layer2_summary_cpasync.md`
- third sample (`small_mma*`):
  - harness-only output: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_harness_only`
  - full preprocess output: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/geak_preprocess_small_mma_full`
  - per-sample notes: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/run_manifest_small_mma.md`
  - per-sample summary: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer2/layer2_summary_small_mma.md`

## What Each Sample Proves
- `elementwise_add*` proves the simplest NV Gluon to AMD Gluon path on `gfx942`: common Gluon indexing, masked load/store semantics, wave64 retuning, deterministic harness execution, and full GEAK preprocess closure on a plain elementwise kernel.
- `elementwise_add_cpasync*` proves that an explicit NVIDIA `cp.async` tutorial fragment can still be brought onto `gfx942` when the async path is deliberately downgraded to synchronous semantics, and that this downgraded path can still enter the same GEAK-native harness and preprocess closure.
- `small_mma*` proves that a NV WGMMA/TMA-style small matmul tutorial fragment can be rewritten around AMD CDNA3 MFMA plus direct buffer movement on `gfx942`, while preserving host/test structure and correctness, and that this rewritten path can also enter the same GEAK-native harness and preprocess closure.

## What Remains Unmatched
- The `cpasync` sample does not prove async-copy overlap parity, shared-memory pipeline parity, or any same-name AMD `cp.async` capability.
- The `small_mma` sample does not prove same-name parity for `TensorDescriptor`, `tma`, `mbarrier`, `warpgroup_mma`, or `fence_async_shared`; it proves semantic rewrite, not feature equivalence.
- None of the three samples proves `04+` tutorial support beyond the explicitly locked third sample, and none proves `tcgen05_*`, `clc`, `gfx1250`, or `cdna4`.
- None of the three samples proves Arena integration, task-validator readiness, or CI readiness.
- The current GEAK harness-driven closure still emits `benchmark_baseline.txt` and `full_benchmark_baseline.txt` from the same full-benchmark text for these focused harnesses, and the generated `COMMANDMENT.md` uses `--full-benchmark` in both `BENCHMARK` and `FULL_BENCHMARK`.

## Optimization Readout
- The current pass was scoped to preprocess-depth alignment, not a full optimizer completion run.
- In the earlier short GEAK optimization observation, `elementwise_add*` showed a small real gain on the clean-mirror run, while `elementwise_add_cpasync*` and `small_mma*` were shown to enter the optimizer pipeline but were not driven to final `best_results.json` / `final_report.json` within that observation window.
- So the current `gfx942` mainline conclusion is: semantic translation coverage and preprocess closure are now aligned across all three samples, while full optimizer end-state evidence is still strongest for the first sample.
