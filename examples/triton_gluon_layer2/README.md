# Triton-Gluon Layer2 Examples

This directory is for **Layer 2** examples:

> existing Triton + NVIDIA Gluon input -> Triton + AMD Gluon output

Unlike the Layer 1 examples, this directory is not primarily about runtime
bootstrap. It is about keeping a translation example, a focused verification
entry, and a short explanation together.

## First-sample policy

The first sample is fixed:

- source file: `/apps/qiongzhu/triton/python/tutorials/gluon/03-async-copy.py`
- allowed symbols:
  - `elementwise_add_kernel`
  - `elementwise_add`
  - `test_elementwise_add`

The first sample must **not** expand to:

- `memcpy_1d_cpasync_kernel`
- `elementwise_add_cpasync_kernel`
- any path that depends on NVIDIA `cp.async`
- any `04+` tutorial feature such as TMA, WGMMA, `tcgen05_*`, or CLC

## Intended file layout

The first Layer 2 sample should eventually create:

```text
examples/triton_gluon_layer2/
├── README.md
├── 03_elementwise_add_nv.py
├── 03_elementwise_add_amd.py
├── test_elementwise_add_layer2.py
├── run_manifest.md
└── layer2_summary.md
```

### File roles

- `03_elementwise_add_nv.py`
  - frozen input reference from the NVIDIA tutorial
- `03_elementwise_add_amd.py`
  - first AMD translation candidate
- `test_elementwise_add_layer2.py`
  - focused execution entry for compile/correctness/benchmark
- `run_manifest.md`
  - runtime, target, paths, and decisions
- `layer2_summary.md`
  - what was preserved, what changed, what still has no parity

## Runtime policy

Layer 2 reuses the current Layer 1 execution environment by default:

- same workspace
- same `gfx942` target preference
- same GEAK/Triton roots
- same model defaults

The example files themselves should **not** hardcode:

- current container name
- current branch name
- current host-only absolute paths

Those belong in the run manifest, not in the example source.

## Validation ladder

The first sample should pass in this order:

1. compile
2. correctness
3. benchmark
4. optional full preprocess

Profiler evidence is welcome but not required for the first Layer 2 success.

## What counts as success

The first sample is successful if:

- the AMD translation compiles on the chosen AMD target
- correctness passes against a trusted reference
- at least one stable benchmark path works
- the translation notes clearly explain:
  - what stayed in the common Gluon subset
  - what changed for AMD
  - what NVIDIA-specific features were intentionally not carried over

## What this directory is not for

- not for Arena task creation
- not for CI enablement
- not for `task_validator`
- not for proving TMA/TDM or WGMMA/tcgen05 parity

Those belong to later Layer 2 or post-Layer 2 stages.

## Related documents

- `docs/triton_gluon_layer2_scope.md`
- `docs/triton_gluon_translation_rules.md`
- `docs/triton_gluon_writing_guide.md`
- `docs/triton_gluon_api_quick_reference.md`
- `docs/triton_gluon_layer1_scope.md`
