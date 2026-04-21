# Triton-Gluon Layer2 Scope

This document defines **Layer 2** of the current `triton-gluon` roadmap:

> Given existing Triton + NVIDIA Gluon code, generate Triton + AMD Gluon code,
> then verify the translated result through the same kind of bounded
> correctness/benchmark workflow used in Layer 1.

Layer 2 is intentionally narrower than a general source-to-source translator.
It is broader than Layer 1 because the input is no longer already AMD Gluon.

## 1. Relationship to the other layers

- **Layer 1**: existing Triton-Gluon input, usually already AMD-facing, can be
  wrapped, benchmarked, and summarized by GEAK.
- **Layer 2**: existing **NV Gluon** input is rewritten into **AMD Gluon**.
- **Layer 3**: plain Triton input is lifted into AMD Gluon.

What Layer 1 already proved is documented in:

- `docs/triton_gluon_layer1_scope.md`

What Layer 2 must add is a credible **translation step**, not just a harness or
baseline wrapper.

## 2. Input assumptions

Layer 2 assumes one of the following as input:

1. A Gluon tutorial or kernel that already uses NVIDIA-oriented Gluon idioms
2. A Gluon test file whose semantics are clear enough to preserve
3. A bounded code block inside a larger tutorial/test that can be isolated

Layer 2 does **not** assume:

- that the input is already AMD Gluon
- that every NVIDIA hardware feature has an AMD Gluon peer
- that the first successful sample must already be Arena-ready

## 3. First-sample policy

The first Layer 2 sample is fixed:

- Source file: `python/tutorials/gluon/03-async-copy.py`
- Locked scope:
  - `elementwise_add_kernel`
  - `elementwise_add`
  - `test_elementwise_add`

Why this sample is first:

- It lives in a NVIDIA tutorial file, so it is a real Layer 2 input.
- Its kernel body is still dominated by common Gluon syntax:
  - `@gluon.jit`
  - `program_id`
  - `BlockedLayout`
  - `SliceLayout`
  - `arange`
  - `load` / `store`
- It avoids forcing the first sample to also prove:
  - Hopper TMA parity
  - WGMMA parity
  - Blackwell `tcgen05_*`
  - Blackwell `clc`

This means the first Layer 2 success proves:

- GEAK can rewrite a **NVIDIA Gluon tutorial fragment** into an **AMD/gfx942**
  runnable Gluon fragment

It does **not** prove:

- full NVIDIA hardware-feature parity on AMD

## 4. Runtime and model policy

Layer 2 inherits the current verified Layer 1 runtime policy unless explicitly
changed later:

- same GEAK workspace
- same Triton workspace
- same target preference: `gfx942`
- same fixed container/runtime assumptions

Layer 2 also inherits the current model defaults unless explicitly changed
later:

- `model_class: amd_llm`
- `model_name: claude-opus-4.6`

Model switching is **not** part of Layer 2 execution unless called out as a
separate experiment.

## 5. Minimum success criteria

For the first sample, success means:

1. A frozen NV input sample exists
2. An AMD translation exists
3. The AMD translation compiles
4. The AMD translation passes correctness
5. At least one benchmark path is stable enough to report latency
6. A Layer 2 summary is written that distinguishes:
   - what stayed in the common Gluon subset
   - what changed for AMD
   - what still has no parity

### Required

- compile
- correctness
- at least one benchmark path

### Recommended

- GEAK preprocess closure
- deterministic harness
- benchmark/full-benchmark pair

### Optional on the first sample

- profiler-backed evidence
- Arena task promotion
- task validator
- CI integration

## 6. What Layer 2 should produce

Layer 2 should create:

- one frozen NV example file
- one AMD translation file
- one focused test or harness entry
- one run manifest
- one Layer 2 result summary

The first round should also add the supporting docs:

- `docs/triton_gluon_translation_rules.md`
- `examples/triton_gluon_layer2/README.md`

## 7. What Layer 2 should not do in the first round

- Do not change GEAK main routing
- Do not introduce new skill names
- Do not redesign the runtime/container strategy
- Do not jump to `04-tma.py`, `05-wgmma.py`, `06-tcgen05.py`, or later Blackwell
  tutorials as the first sample
- Do not treat compile-only success as enough
- Do not claim profiler-ready or Arena-ready status by default

## 8. Acceptance boundary for later expansion

Only after the first sample succeeds should Layer 2 expand to:

- stronger async-copy parity cases
- descriptor/TDM-themed cases
- matrix-op migration cases
- eventually `gfx1250` / `cdna4`-biased capability branches

The second branch of expansion should be selected based on what the first sample
fails to prove:

- if the common Gluon subset is easy, expand toward AMD matrix ops
- if memory movement is the gap, expand toward async-copy or descriptor paths

## 9. Related documents

- `docs/triton_gluon_layer1_scope.md`
- `docs/triton_gluon_writing_guide.md`
- `docs/triton_gluon_api_quick_reference.md`
- `docs/triton_gluon_mi3xx_baseline.md`
- `docs/triton_gluon_translation_rules.md`
- `examples/triton_gluon_layer2/README.md`
