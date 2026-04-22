# Triton-family Gluon input fixtures

This directory is the only supported example surface for the current
`triton + amd_gluon` feature work.

It intentionally keeps just **three representative input fixtures**:

- `01_plain_triton_input.py`
- `02_nv_gluon_input.py`
- `03_amd_gluon_input.py`

These files are **inputs only**. GEAK should generate and benchmark candidate
outputs at run time; this directory does **not** store golden outputs, run
manifests, preprocess snapshots, or benchmark artifacts.

## Files

- `config.yaml`: thin example config aligned with the main-branch example style
- `test_inputs.py`: thin harness shared by all three fixtures
- `01_plain_triton_input.py`: plain Triton input
- `02_nv_gluon_input.py`: `nv_gluon` input that still uses common Gluon syntax
- `03_amd_gluon_input.py`: existing `amd_gluon` input for `hip/gfx942`

## Quick local checks

```bash
cd examples/triton_gluon_inputs
python3 test_inputs.py --kernel-file 01_plain_triton_input.py --mode compile
python3 test_inputs.py --kernel-file 01_plain_triton_input.py --correctness
python3 test_inputs.py --kernel-file 01_plain_triton_input.py --full-benchmark
```

Swap `--kernel-file` to `02_nv_gluon_input.py` or `03_amd_gluon_input.py` to
exercise the other input dialects.

## Using with `geak`

Keep `kernel_type=triton`. The current explicit feature gate is
`gluon_feature_mode` (the docs call `auto` / `force` "gluon-on").

Example: plain Triton input, but ask GEAK to prefer an `amd_gluon` candidate:

```bash
cd /path/to/GEAK
geak \
  --repo /path/to/GEAK/examples/triton_gluon_inputs \
  --kernel-url /path/to/GEAK/examples/triton_gluon_inputs/01_plain_triton_input.py \
  --test-command "cd /path/to/GEAK/examples/triton_gluon_inputs && python3 test_inputs.py --kernel-file 01_plain_triton_input.py --correctness && python3 test_inputs.py --kernel-file 01_plain_triton_input.py --full-benchmark" \
  --task "Optimize this Triton kernel. Keep kernel_type=triton. Turn gluon-on. Input dialect is plain_triton. Prefer an amd_gluon output candidate on hip/gfx942."
```

Example: `nv_gluon` input that should translate toward AMD-facing Gluon:

```bash
cd /path/to/GEAK
geak \
  --repo /path/to/GEAK/examples/triton_gluon_inputs \
  --kernel-url /path/to/GEAK/examples/triton_gluon_inputs/02_nv_gluon_input.py \
  --test-command "cd /path/to/GEAK/examples/triton_gluon_inputs && python3 test_inputs.py --kernel-file 02_nv_gluon_input.py --correctness && python3 test_inputs.py --kernel-file 02_nv_gluon_input.py --full-benchmark" \
  --task "Optimize this Triton kernel. Keep kernel_type=triton. Turn gluon-on. Input dialect is nv_gluon. Translate NVIDIA-facing Gluon assumptions into an amd_gluon candidate on hip/gfx942."
```

Example: existing `amd_gluon` input that should stay on the AMD Gluon route:

```bash
cd /path/to/GEAK
geak \
  --repo /path/to/GEAK/examples/triton_gluon_inputs \
  --kernel-url /path/to/GEAK/examples/triton_gluon_inputs/03_amd_gluon_input.py \
  --test-command "cd /path/to/GEAK/examples/triton_gluon_inputs && python3 test_inputs.py --kernel-file 03_amd_gluon_input.py --correctness && python3 test_inputs.py --kernel-file 03_amd_gluon_input.py --full-benchmark" \
  --task "Optimize this Triton kernel. Keep kernel_type=triton. Turn gluon-on. Input dialect is amd_gluon. Continue searching for a better amd_gluon candidate on hip/gfx942."
```
