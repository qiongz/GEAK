# Layer3 Summary (`gfx942` suite)

## Outcome
- Layer3 now has **three fixed samples** on `gfx942` with both authoring evidence and sample-specific blind-eval evidence:
  - `vector_add`
  - `fused_softmax`
  - `matmul`
- The existing `vector_add` authoring / blind-eval evidence was preserved.
- `softmax` and `matmul` now match that pattern: authoring closure under `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/` plus stripped blind-eval reruns in their own workspaces.

## Coverage Comparison
- `vector_add` proves the bounded direct-lift baseline for plain Triton elementwise indexing on `gfx942`:
  - one-dimensional launcher
  - offset + mask semantics
  - explicit AMD buffer load/store path
- `softmax` fills the Layer3 reduction gap left by `vector_add`:
  - row-wise reduction
  - `max`-shift numerical stabilization
  - `sum` normalization
  - irregular-column mask / padding
- `matmul` fills the Layer3 matrix-core gap left by `vector_add` and `softmax`:
  - grouped launch / block scheduling
  - 2D pointer arithmetic
  - masked K-loop
  - explicit `tl.dot` -> CDNA3 MFMA lowering

## Remaining Gaps
- The current Layer3 suite still does **not** cover:
  - shared-memory async movement
  - descriptor / TMA / TDM paths
  - cross-CTA reductions
  - full autotune parity
  - FP8 paths
  - activation-fused matmul variants
  - a generic Triton-to-Gluon compiler claim

## Authoring Evidence
- `vector_add` authoring:
  - summary: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/layer3_summary.md`
  - manifest: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/run_manifest.md`
- `softmax` authoring:
  - summary: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/layer3_summary_softmax.md`
  - manifest: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/run_manifest_softmax.md`
- `matmul` authoring:
  - summary: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/layer3_summary_matmul.md`
  - manifest: `/apps/qiongzhu/GEAK/examples/triton_gluon_layer3/run_manifest_matmul.md`

## Blind-Eval Evidence
- `vector_add` blind eval:
  - workspace: `/apps/qiongzhu/GEAK_layer3_blind_eval_20260420`
  - summary: `/apps/qiongzhu/GEAK_layer3_blind_eval_20260420/blind_eval_results/layer3_vector_add/layer3_summary.md`
  - manifest: `/apps/qiongzhu/GEAK_layer3_blind_eval_20260420/blind_eval_results/layer3_vector_add/run_manifest.md`
- `softmax` blind eval:
  - workspace: `/apps/qiongzhu/GEAK_layer3_softmax_blind_eval_20260420`
  - summary: `/apps/qiongzhu/GEAK_layer3_softmax_blind_eval_20260420/blind_eval_results/layer3_softmax/layer3_summary.md`
  - manifest: `/apps/qiongzhu/GEAK_layer3_softmax_blind_eval_20260420/blind_eval_results/layer3_softmax/run_manifest.md`
- `matmul` blind eval:
  - workspace: `/apps/qiongzhu/GEAK_layer3_matmul_blind_eval_20260420`
  - summary: `/apps/qiongzhu/GEAK_layer3_matmul_blind_eval_20260420/blind_eval_results/layer3_matmul/layer3_summary.md`
  - manifest: `/apps/qiongzhu/GEAK_layer3_matmul_blind_eval_20260420/blind_eval_results/layer3_matmul/run_manifest.md`

## Interpretation
- The suite now forms a bounded but meaningful Layer3 ladder on `gfx942`:
  - `vector_add` establishes the direct-lift baseline
  - `softmax` proves reduction + masking semantics
  - `matmul` proves matrix-core scheduling + MFMA lowering
- Every blind-eval rerun used a **sample-specific stripped workspace**, because a shared Layer3 stripped root would leak the other samples' AMD answers through `CODEBASE_CONTEXT.md`.
- No changes were required in `preprocessor.py`, `dispatch.py`, or `task_generator.py` main logic to close any of the three samples.
