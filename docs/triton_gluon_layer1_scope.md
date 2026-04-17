# Triton-Gluon Layer 1 Scope

This document defines **Layer 1** of the current `triton-gluon` roadmap:

> Given existing Triton-Gluon code, tests, or harnessable examples, GEAK can
> understand them, wrap them into GEAK-compatible evaluation paths, and
> produce Triton-Gluon outputs such as deterministic harnesses, preprocess
> artifacts, baseline summaries, and bounded task contracts.

This is the layer that has already been exercised in the current MI3xx/gfx942
baseline work. It is intentionally narrower than:

- **Layer 2**: given Triton + NVIDIA Gluon, generate Triton + AMD Gluon
- **Layer 3**: given Triton, generate Triton + AMD Gluon

## 1. What Layer 1 assumes as input

Layer 1 expects one of the following as its starting point:

1. An existing Triton-Gluon source file
2. An existing Triton-Gluon test file
3. An existing Triton-Gluon benchmark-like entry point
4. A repo where Triton-Gluon code already exists and can be adapted

Examples of valid Layer 1 starting points in the current work:

- `python/test/gluon/test_core.py`
- `python/test/gluon/test_frontend.py`

Layer 1 does **not** assume GEAK can synthesize AMD Gluon code from plain
Triton-only input. That belongs to Layer 3.

## 2. What Layer 1 is expected to output

Within scope, Layer 1 may produce:

- a deterministic four-mode GEAK harness
- `GEAK_HARNESS_ONLY=1` validation output
- full preprocess closure:
  - `COMMANDMENT.md`
  - `benchmark_baseline.txt`
  - `full_benchmark_baseline.txt`
  - `baseline_metrics.json`
  - `profile.json`
- a baseline summary document
- a task skeleton or bounded local task contract

It does **not** imply:

- self-contained Arena-ready tasks by default
- generic NV-Gluon to AMD-Gluon translation
- generic Triton to AMD Gluon generation

## 3. What has been proven in the current baseline run

Current evidence comes from:

- `optimization_logs/triton_gluon_mi3xx_20260417_055346/run_manifest.md`
- `optimization_logs/triton_gluon_mi3xx_20260417_055346/define-mi3xx-baseline-metrics.md`

### Proven capabilities

1. **GEAK can identify and select existing Triton-Gluon canaries**
   - `test_inline_with_amdgpu_dialect`
   - `test_buffer_load_store`
   - `test_amd_mfma`

2. **GEAK can generate deterministic four-mode harnesses**
   - `--correctness`
   - `--profile`
   - `--benchmark`
   - `--full-benchmark`

3. **GEAK can close the full preprocess loop on existing Triton-Gluon tasks**
   - all three canaries reached full preprocess completion

4. **GEAK can produce benchmark-compatible outputs**
   - `GEAK_SHAPES_USED=[...]`
   - `GEAK_RESULT_LATENCY_MS=<number>`

5. **GEAK can produce profiler-backed baselines for profiler-ready tasks**
   - `inline_with_amdgpu_dialect`
   - `amd_mfma`

6. **GEAK can produce local task contracts from validated Layer 1 tasks**
   - two bounded `AgentKernelArena` local task contracts were created for:
     - `inline_with_amdgpu_dialect`
     - `amd_mfma`
   - these contracts are locally runnable, but still coupled to the current workspace layout and runtime discovery assumptions

### Partially proven capabilities

1. **Baseline generation when profiling is unavailable**
   - `test_buffer_load_store` still produces a valid preprocess baseline
   - but only by wall-clock fallback, not by Metrix-backed profiling

This means Layer 1 can tolerate some task classes that are benchmark-valid but
not profiler-ready.

## 4. Validated task classes in Layer 1

### A. Existing AMD Gluon execution tasks

These are the strongest Layer 1 evidence:

- `inline_with_amdgpu_dialect`
- `amd_mfma`

They prove that GEAK can work with existing AMD Gluon code that:

- runs on `gfx942`
- passes correctness
- supports benchmark/full-benchmark
- can be profiled with Metrix

### B. Existing compile/IR validation tasks

- `buffer_load_store`

This proves a weaker but still useful Layer 1 capability:

- GEAK can wrap and benchmark a compile/IR-oriented Gluon task
- but such a task is not automatically profiler-ready or Arena-ready

## 5. What Layer 1 does NOT prove

Layer 1 should **not** be overstated. It does **not** prove that GEAK can:

1. generate AMD Gluon from plain Triton input
2. translate NVIDIA Gluon to AMD Gluon in a general way
3. make any Gluon task automatically Arena-ready
4. infer missing architecture-specific APIs without an existing Gluon reference
5. guarantee profiler readiness for every benchmark-valid Gluon task

## 6. Current Layer 1 boundary

At the current milestone, the boundary is:

- `inline_with_amdgpu_dialect`: Layer 1 proven
- `amd_mfma`: Layer 1 proven
- `buffer_load_store`: Layer 1 proven only as compile-latency / wall-clock fallback

This is why the current workstream does **not** treat all three canaries as
equally ready for later Arena promotion.

## 7. Layer 1 completion criteria

For the current MI3xx/gfx942 track, Layer 1 should be considered complete when:

1. the target repo already contains Triton-Gluon code or tests
2. GEAK can generate a deterministic four-mode harness
3. GEAK can complete preprocess closure
4. GEAK can summarize correctness and latency baselines
5. at least the profiler-ready subset can be promoted into bounded local task contracts

These criteria are already met for the current baseline work, with the explicit
exception that `buffer_load_store` remains non-profiler-ready.

## 8. Relationship to the other documents

- For reusable syntax, architecture mapping, and concept-level guidance, see
  `docs/triton_gluon_writing_guide.md`
- For a compact API-oriented summary, see
  `docs/triton_gluon_api_quick_reference.md`
- For the current MI3xx repo-specific workflow, see
  `docs/triton_gluon_mi3xx_baseline.md`
