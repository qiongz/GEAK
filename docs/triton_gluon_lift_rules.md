# Triton-to-AMD-Gluon Lift Rules

This document is the Layer 3 direct-lift playbook.

It is not a full compiler design document. It is a bounded rule set for taking
an existing **plain Triton** example and rewriting it into an **AMD-facing
Gluon** example while preserving the highest-value semantics first:

1. compile
2. correctness
3. benchmark
4. optional preprocess / profiler evidence

For Layer 3 scope and acceptance criteria, see:

- `docs/triton_gluon_layer3_scope.md`

For the underlying Gluon writing model, see:

- `docs/triton_gluon_writing_guide.md`
- `skills/triton-gluon-mi3xx/SKILL.md`

## 1. Lift workflow

### Step 1: Freeze the plain Triton source fragment

Before changing code, isolate the exact source symbols that belong to the first
sample:

- kernel body
- host launcher
- correctness intent
- benchmark intent

Do **not** bring along tutorial-only prints, plots, or notebook-style demo code
when those can live in the Layer 3 harness instead.

### Step 2: Split the source into semantic buckets

#### Bucket A: semantics that should stay recognizable

These usually survive the lift as concepts, even though the syntax changes:

- `triton.jit` kernel structure
- host-side `grid`
- `triton.cdiv`
- `tl.program_id`
- `tl.constexpr`
- pointer arithmetic
- offsets
- masks
- eager PyTorch reference checks
- benchmark intent

#### Bucket B: implicit Triton decisions that must become explicit

These need design choices, not line-by-line copying:

- register / thread-data layout
- warp-width assumptions
- memory-path choice
- target-specific load/store operations
- instruction-path choice when later samples need matrix ops

### Step 3: Preserve the host-side shape first

Do **not** redesign the host launcher unless the current target forces it.

The first Layer 3 sample should preserve as much as possible of:

- launcher signature
- block sizing intent
- indexing pattern
- correctness interface
- benchmarkable call structure

### Step 4: Make layout explicit before lowering memory ops

Plain Triton often leaves layout implicit. Gluon does not.

Before replacing `tl.load` / `tl.store`, choose:

1. target family
2. wave / warp assumptions
3. `BlockedLayout`
4. any derived `SliceLayout`
5. memory path

On `gfx942`, prefer wave64-valid layouts. Do **not** blindly keep wave32-like
assumptions from examples that were not written for AMD.

### Step 5: Choose the narrowest AMD-specific path that works

When a plain Triton op maps to common Gluon plus an AMD-specific memory path:

1. keep the common indexing structure
2. replace only the memory / instruction pieces that must become AMD-specific
3. document the explicit choices in the manifest and summary

For `gfx942` / CDNA3, prefer:

- common Gluon syntax
- explicit `BlockedLayout` / `SliceLayout`
- `ttgl.amd.cdna3.buffer_load`
- `ttgl.amd.cdna3.buffer_store`

## 2. Lift decision table

| Plain Triton pattern | Layer 3 rule |
|----------------------|--------------|
| `@triton.jit` kernel skeleton | Rewrite to `@gluon.jit`, keeping kernel purpose and argument roles recognizable |
| `tl.program_id(axis=0)` | Map to `ttgl.program_id(0)` and keep the same launch dimension story |
| `tl.arange(0, BLOCK_SIZE)` | Replace with `ttgl.arange(0, BLOCK_SIZE, layout=...)` after choosing an explicit layout |
| `tl.load(ptr + offsets, mask=mask, other=...)` | Keep the pointer arithmetic and mask semantics; on `gfx942`, prefer `ttgl.amd.cdna3.buffer_load(...)` when a direct buffer path is valid |
| `tl.store(ptr + offsets, value, mask=mask)` | Keep the same store semantics; on `gfx942`, prefer `ttgl.amd.cdna3.buffer_store(...)` when a direct buffer path is valid |
| `triton.cdiv(...)` in host launcher | Keep it unless the lift fundamentally changes the tiling model |
| tutorial correctness check against PyTorch | Preserve it |
| tutorial benchmark / perf-report intent | Preserve the benchmark intent, but move plotting / demo UI into the harness if needed |

## 3. First-sample-specific rules

The first sample is fixed to:

- `python/tutorials/01-vector-add.py`
- symbols:
  - `add_kernel`
  - `add`
  - the tutorial benchmark intent

### What must be kept

- the one-dimensional launcher shape
- `block_start`, `offsets`, and `mask` semantics
- the elementwise `x + y` kernel semantics
- the PyTorch reference as the correctness oracle
- the benchmarkable execution structure

### What must change

- `tl.*` syntax becomes Gluon syntax
- layout becomes explicit
- the memory path becomes explicit for AMD
- target assumptions become `gfx942` / wave64-aware

### What should not be carried over verbatim

- top-level tutorial prints
- plot rendering
- notebook-style demo flow

Those belong in the Layer 3 harness or documentation, not in the AMD example.

## 4. AMD target selection rules

### First sample target

- target family: `gfx942`
- architecture class: `CDNA3`

### Why this matters

The first Layer 3 sample is **not** trying to prove the richest AMD parity
surface. It is trying to prove that:

- a plain Triton tutorial fragment can be lifted into a valid AMD-facing Gluon
  sample
- under the same GEAK evaluation story used in Layers 1 and 2

### Target-specific guidance

#### `gfx942` / CDNA3

Prefer:

- common Gluon syntax for launcher and indexing
- explicit wave64-valid `BlockedLayout`
- `ttgl.amd.cdna3.buffer_load`
- `ttgl.amd.cdna3.buffer_store`

Avoid treating the first Layer 3 sample as if it must also prove:

- `async_copy`
- shared-memory pipelines
- `mfma`
- `wmma`
- `tma`
- `tdm`

Those belong to later samples, not the first direct lift.

## 5. Example and harness rules

The first Layer 3 example should be represented by:

- one frozen plain Triton input
- one AMD lift candidate
- one focused harness

### Why both example and harness are required

- the example lets humans inspect the lift delta
- the harness lets GEAK verify compile / correctness / benchmark / preprocess
  closure without reinterpreting the full tutorial every time

### Minimum acceptable validation form

- compile: required
- correctness: required
- benchmark: required
- preprocess closure: recommended
- profiler evidence: optional for the first sample

Compile-only is **not** enough to claim Layer 3 success.

## 6. Lift notes template

Each Layer 3 sample should leave a short lift note with these fields:

```markdown
## Lift notes
- source_tutorial:
- source_symbols:
- target_arch:
- kept_triton_semantics:
- explicit_gluon_layout_decisions:
- amd_specific_parts_added_or_changed:
- plain_triton_parts_not_carried_verbatim:
- known_non_parity:
```

This keeps Layer 3 from becoming an undocumented code fork.

## 7. Anti-patterns

### Do not do this

- Rename `tl.*` APIs to guessed AMD names
- Keep implicit Triton layout decisions implicit in the Gluon version
- Keep wave32-style assumptions unchanged on `gfx942`
- Claim "AMD equivalent" when the result only compiles
- Expand the first sample into reduction, softmax, or matmul in the same pass
- Change runtime strategy and lift logic in the same iteration

### Be explicit instead

- If a layout choice was added, say what it was
- If a memory path was changed, say how
- If a tutorial artifact was dropped, say it was dropped
- If a feature has no parity, record it

## 8. Known non-parity classes

The following are explicitly **not first-sample obligations**:

- reduction parity
- shared-memory async-copy parity
- `mfma` / `wmma` / tensor-core-style parity
- descriptor-driven memory paths
- profiler-ready status for every benchmark-valid harness

## 9. Related files

- `docs/triton_gluon_layer3_scope.md`
- `examples/triton_gluon_layer3/README.md`
- `docs/triton_gluon_writing_guide.md`
- `skills/triton-gluon-mi3xx/SKILL.md`
- `docs/triton_gluon_mi3xx_baseline.md`
