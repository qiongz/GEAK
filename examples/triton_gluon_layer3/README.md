# Triton-Gluon Layer3 Examples

This directory is for **Layer 3** examples:

> existing plain Triton input -> Triton + AMD Gluon output

Unlike Layer 1, this directory is not primarily about runtime bootstrap.
Unlike Layer 2, the input here is not already Gluon. Each Layer 3 sample keeps
one bounded plain Triton reference, one AMD direct lift, one focused four-mode
harness, and per-sample authoring notes together.

## Fixed sample set

The current `gfx942` Layer 3 mainline is locked to exactly three tutorial
sources:

1. `/apps/qiongzhu/triton/python/tutorials/01-vector-add.py`
2. `/apps/qiongzhu/triton/python/tutorials/02-fused-softmax.py`
3. `/apps/qiongzhu/triton/python/tutorials/03-matrix-multiplication.py`

Locked symbol scopes:

- `01-vector-add.py`
  - `add_kernel`
  - `add`
  - tutorial correctness / benchmark intent
- `02-fused-softmax.py`
  - `softmax_kernel`
  - `softmax`
  - tutorial correctness / benchmark intent
- `03-matrix-multiplication.py`
  - `matmul_kernel`
  - `matmul`
  - tutorial benchmark intent

The current Layer 3 examples must **not** expand to:

- extra tutorial files beyond `01`, `02`, and `03`
- generic compiler claims
- descriptor / async-copy / TMA parity claims
- full autotune parity claims
- Arena / CI integration work

## Intended authoring layout

The authoring tree now uses one bounded file set per sample plus one suite
summary:

```text
examples/triton_gluon_layer3/
├── README.md
├── 01_vector_add_triton.py
├── 01_vector_add_amd.py
├── test_vector_add_layer3.py
├── run_manifest.md
├── layer3_summary.md
├── 02_fused_softmax_triton.py
├── 02_fused_softmax_amd.py
├── test_fused_softmax_layer3.py
├── run_manifest_softmax.md
├── layer3_summary_softmax.md
├── 03_matmul_triton.py
├── 03_matmul_amd.py
├── test_matmul_layer3.py
├── run_manifest_matmul.md
├── layer3_summary_matmul.md
└── layer3_summary_gfx942.md
```

### File roles

- `*_triton.py`
  - frozen plain Triton reference from the tutorial, trimmed to the locked
    symbol scope
- `*_amd.py`
  - bounded AMD Gluon direct-lift candidate for `gfx942`
- `test_*_layer3.py`
  - focused compile / correctness / profile / benchmark / full-benchmark entry
- `run_manifest*.md`
  - runtime, target, commands, gate status, and direct-lift decisions
- `layer3_summary*.md`
  - what Triton semantics were preserved, what became explicit in Gluon, and
    what still has no parity
- `layer3_summary_gfx942.md`
  - suite-level authoring summary across `vector_add`, `softmax`, and `matmul`

## Blind-eval policy

Authoring outputs under `examples/triton_gluon_layer3/` are **gold-sample
authoring results**, not blind-eval evidence.

Each Layer 3 sample must therefore also have its own **sample-specific stripped
blind-eval workspace**. A shared Layer 3 stripped workspace is not sufficient,
because `CODEBASE_CONTEXT.md` would expose the other samples' AMD answers.

The final blind-eval evidence for each sample must be regenerated in its own
workspace and must keep only:

- GEAK runtime / framework code
- `docs/docker_env.md`
- the current sample's plain Triton file
- the current sample's AMD file
- the current sample's focused harness

## Runtime policy

Layer 3 reuses the validated Layer 1 / Layer 2 execution path by default:

- container: `feature-triton-gluon-mi3xx-baseline`
- target: `gfx942`
- caches:
  - `/apps/qiongzhu/.triton`
  - `/apps/qiongzhu/.ccache`
  - `/apps/qiongzhu/.pip-cache`

The example source files themselves should **not** hardcode:

- container names
- branch names
- host-only absolute paths

Those belong in run manifests and blind-eval workspace notes, not in the sample
source.

## Harness contract

Every Layer 3 sample harness must support:

- `--mode compile`
- `--correctness`
- `--profile`
- `--benchmark`
- `--full-benchmark`

And must:

- keep one ordered case stream
- reuse that stream across correctness / profile / benchmark / full-benchmark
- print `GEAK_SHAPES_USED=[...]`
- end benchmark-capable modes with `GEAK_RESULT_LATENCY_MS=<number>`

## Validation ladder

Every Layer 3 sample should close in this order:

1. compile
2. correctness
3. profile
4. benchmark
5. full-benchmark
6. `GEAK_HARNESS_ONLY=1`
7. full preprocess
8. sample-specific blind-eval rerun

## What counts as success

Each Layer 3 sample is successful if:

- the AMD lift compiles on `gfx942`
- correctness passes against a trusted PyTorch reference
- at least one stable benchmark path works
- full preprocess closes
- the authoring manifest / summary are complete
- the sample-specific blind-eval `CODEBASE_CONTEXT.md` passes a no-answer audit

## What this directory is not for

- not for Arena task creation by default
- not for CI enablement
- not for claiming a generic Triton-to-Gluon compiler
- not for proving complete reduction / matmul / autotune / descriptor parity

Those belong to later Layer 3 or post-Layer 3 stages.

## Related documents

- `docs/triton_gluon_layer3_scope.md`
- `docs/triton_gluon_lift_rules.md`
- `docs/triton_gluon_writing_guide.md`
- `skills/triton-gluon-mi3xx/SKILL.md`
- `docs/triton_gluon_mi3xx_baseline.md`
