# Triton-Gluon Layer3 Scope

This document defines **Layer 3** of the current `triton-gluon` roadmap:

> Given an existing plain Triton kernel or tutorial fragment, generate a bounded
> Triton + AMD Gluon version, then verify the lifted result through the same
> correctness/benchmark workflow used in Layers 1 and 2.

Layer 3 is intentionally narrower than a general Triton-to-Gluon compiler. It
is broader than Layer 2 because the input is no longer already written in
Gluon.

## 1. Relationship to the other layers

- **Layer 1**: existing Triton-Gluon input can already be wrapped and evaluated
  by GEAK.
- **Layer 2**: existing Triton + NVIDIA Gluon input is rewritten into Triton +
  AMD Gluon.
- **Layer 3**: existing plain Triton input is lifted into Triton + AMD Gluon.

What Layer 1 already proved is documented in:

- `docs/triton_gluon_layer1_scope.md`

What Layer 2 adds is a bounded NV-Gluon to AMD-Gluon translation step:

- `docs/triton_gluon_layer2_scope.md`
- `docs/triton_gluon_translation_rules.md`

What Layer 3 must add is a credible **direct lift step** from plain Triton into
an AMD-facing Gluon implementation, without routing through a Layer 2
intermediate representation.

## 2. Input assumptions

Layer 3 assumes one of the following as input:

1. A plain Triton tutorial whose kernel semantics are already clear
2. A plain Triton kernel plus a small host launcher that should stay recognizable
3. A bounded fragment inside a larger Triton tutorial that can be frozen as the
   first sample

Layer 3 does **not** assume:

- that the input already contains any Gluon API usage
- that implicit Triton layout or memory decisions can be copied 1:1 onto AMD
- that the first successful sample must also prove profiler readiness,
  Arena readiness, or matrix-instruction parity

## 3. First-sample policy

The first Layer 3 sample is fixed:

- source file: `python/tutorials/01-vector-add.py`
- locked scope:
  - `add_kernel`
  - `add`
  - the tutorial's correctness / benchmark intent

Why this sample is first:

- it is a real plain Triton tutorial, not already a Gluon example
- its semantics are simple enough to keep the first lift focused on:
  - launcher preservation
  - indexing preservation
  - mask preservation
  - explicit AMD Gluon layout and memory choices
- it avoids forcing the first sample to also prove:
  - reduction semantics
  - shared-memory async movement
  - matrix instructions
  - descriptor paths

This means the first Layer 3 success proves:

- GEAK can turn a plain Triton tutorial fragment into an AMD-facing Gluon sample
  on `gfx942`

It does **not** prove:

- a generic Triton-to-Gluon compiler
- reduction, softmax, or matmul coverage
- `cp.async`, `tma`, `tdm`, `wgmma`, `mfma`, or `wmma` parity

## 4. Runtime and model policy

Layer 3 inherits the current verified Layer 1 and Layer 2 runtime posture unless
explicitly changed later:

- same GEAK workspace
- same Triton workspace
- same target preference: `gfx942`
- same fixed container/runtime assumptions
- same bounded GEAK preprocess closure contract

Layer 3 also inherits the current model defaults unless explicitly changed
later:

- `model_class: amd_llm`
- `model_name: claude-opus-4.6`

Runtime or model switching is **not** part of Layer 3 execution unless called
out as a separate experiment.

## 5. Minimum success criteria

For the first sample, success means:

1. A frozen plain Triton input sample exists
2. An AMD Gluon direct-lift candidate exists
3. The AMD lift compiles
4. The AMD lift passes correctness
5. At least one benchmark path is stable enough to report latency
6. A Layer 3 summary is written that distinguishes:
   - what Triton semantics were preserved
   - what layout and memory decisions became explicit in Gluon
   - what AMD-specific pieces were added
   - what still has no parity

### Required

- compile
- correctness
- at least one benchmark path
- `run_manifest.md`
- `layer3_summary.md`

### Recommended

- deterministic four-mode harness
- `GEAK_HARNESS_ONLY=1` preprocess closure
- full preprocess closure
- benchmark / full-benchmark pair

### Optional on the first sample

- profiler-backed evidence
- local task promotion
- task-generation specialization
- CI integration

## 6. What Layer 3 should produce

Layer 3 should create:

- one frozen plain Triton example file
- one AMD direct-lift file
- one focused test or harness entry
- one run manifest
- one Layer 3 result summary

The first round should also add the supporting docs:

- `docs/triton_gluon_layer3_scope.md`
- `docs/triton_gluon_lift_rules.md`
- `examples/triton_gluon_layer3/README.md`

## 7. What Layer 3 should not do in the first round

- Do not change GEAK main routing
- Do not redesign the runtime/container strategy
- Do not introduce a second container or rebuild the venv unless a concrete
  blocker requires it
- Do not jump to `02-fused-softmax.py` or `03-matrix-multiplication.py` as the
  first sample
- Do not treat compile-only success as enough
- Do not claim profiler-ready, Arena-ready, or matrix-parity status by default
- Do not silently replace unsupported Triton semantics with guessed AMD APIs

## 8. Acceptance boundary for later expansion

Only after the first sample succeeds should Layer 3 expand to:

- reduction-oriented kernels such as `02-fused-softmax.py`
- matrix-op kernels such as `03-matrix-multiplication.py`
- stronger target-specific memory or instruction-path cases

The second branch of expansion should be selected based on what the first sample
fails to prove:

- if memory / reduction semantics are the gap, expand toward fused softmax
- if layout / instruction-path choices are the gap, expand toward matrix
  multiplication

## 9. Related documents

- `docs/triton_gluon_layer1_scope.md`
- `docs/triton_gluon_layer2_scope.md`
- `docs/triton_gluon_translation_rules.md`
- `docs/triton_gluon_lift_rules.md`
- `docs/triton_gluon_writing_guide.md`
- `docs/triton_gluon_mi3xx_baseline.md`
- `examples/triton_gluon_layer3/README.md`
