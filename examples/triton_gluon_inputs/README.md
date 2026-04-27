# Triton-family Gluon input fixtures

This directory is the only supported example surface for the current
`triton + amd_gluon` feature work.

It intentionally keeps a small set of **five representative input fixtures**:

- `01_plain_triton_input.py`
- `02_nv_gluon_input.py`
- `03_amd_gluon_input.py`
- `04_plain_triton_pa_decode.py`
- `05_plain_triton_pa_mqa_logits.py`

These files are **inputs only**. GEAK should generate and benchmark candidate
outputs at run time; this directory does **not** store golden outputs, run
manifests, preprocess snapshots, or benchmark artifacts.

## Files

- `config.yaml`: thin example config aligned with the main-branch example style
- `test_inputs.py`: thin harness shared by all fixtures
- `01_plain_triton_input.py`: plain Triton input
- `02_nv_gluon_input.py`: `nv_gluon` input that still uses common Gluon syntax
- `03_amd_gluon_input.py`: existing `amd_gluon` input for `hip/gfx942`
- `04_plain_triton_pa_decode.py`: cropped plain Triton paged-attention decode fixture for MI300X/gfx942
- `05_plain_triton_pa_mqa_logits.py`: cropped plain Triton paged-MQA logits fixture for MI300X/gfx942

## Quick local checks

```bash
cd examples/triton_gluon_inputs
python3 test_inputs.py --kernel-file 04_plain_triton_pa_decode.py --mode compile
python3 test_inputs.py --kernel-file 04_plain_triton_pa_decode.py --correctness
python3 test_inputs.py --kernel-file 04_plain_triton_pa_decode.py --full-benchmark
```

Swap `--kernel-file` to `05_plain_triton_pa_mqa_logits.py`,
`02_nv_gluon_input.py`, or `03_amd_gluon_input.py` to exercise the other
fixtures.

## Using with `geak`

Keep `kernel_type=triton`. GEAK infers the input dialect from the kernel and,
for Triton inputs, defaults to exploring AMD Gluon as an optimization candidate.
The fixture-level `DIALECT` constants are documentation and test signals; users
do not need to repeat them in the task text.

Example: MI300X-first plain Triton decode input:

```bash
cd /path/to/GEAK
geak \
  --repo /path/to/GEAK/examples/triton_gluon_inputs \
  --kernel-url /path/to/GEAK/examples/triton_gluon_inputs/04_plain_triton_pa_decode.py \
  --test-command "cd /path/to/GEAK/examples/triton_gluon_inputs && python3 test_inputs.py --kernel-file 04_plain_triton_pa_decode.py --correctness && python3 test_inputs.py --kernel-file 04_plain_triton_pa_decode.py --full-benchmark" \
  --task "Optimize this Triton kernel for hip/gfx942."
```

For the second MI300X-first plain Triton candidate, swap the filename above to
`05_plain_triton_pa_mqa_logits.py`.

Example: `nv_gluon` input that should translate toward AMD-facing Gluon:

```bash
cd /path/to/GEAK
geak \
  --repo /path/to/GEAK/examples/triton_gluon_inputs \
  --kernel-url /path/to/GEAK/examples/triton_gluon_inputs/02_nv_gluon_input.py \
  --test-command "cd /path/to/GEAK/examples/triton_gluon_inputs && python3 test_inputs.py --kernel-file 02_nv_gluon_input.py --correctness && python3 test_inputs.py --kernel-file 02_nv_gluon_input.py --full-benchmark" \
  --task "Optimize this Triton-family kernel for hip/gfx942."
```

Example: existing `amd_gluon` input that should stay on the AMD Gluon route:

```bash
cd /path/to/GEAK
geak \
  --repo /path/to/GEAK/examples/triton_gluon_inputs \
  --kernel-url /path/to/GEAK/examples/triton_gluon_inputs/03_amd_gluon_input.py \
  --test-command "cd /path/to/GEAK/examples/triton_gluon_inputs && python3 test_inputs.py --kernel-file 03_amd_gluon_input.py --correctness && python3 test_inputs.py --kernel-file 03_amd_gluon_input.py --full-benchmark" \
  --task "Optimize this Triton-family kernel for hip/gfx942."
```
