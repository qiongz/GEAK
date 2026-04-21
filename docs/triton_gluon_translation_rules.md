# Triton-Gluon NV-to-AMD Translation Rules

This document is the Layer 2 translation playbook.

It is not a full compiler design document. It is a bounded rule set for taking
an existing **NVIDIA-facing Gluon** example and rewriting it into an
**AMD-facing Gluon** example while preserving the highest-value semantics first:

1. compile
2. correctness
3. benchmark
4. optional profiler evidence

For Layer 2 scope and acceptance criteria, see:

- `docs/triton_gluon_layer2_scope.md`

For the underlying language model, see:

- `docs/triton_gluon_writing_guide.md`
- `docs/triton_gluon_api_quick_reference.md`

## 1. Translation workflow

### Step 1: Classify the source fragment

Before changing code, split the source into two buckets:

#### Bucket A: common Gluon subset

These usually survive translation with little or no semantic change:

- `@gluon.jit`
- host-side `grid`
- `triton.cdiv`
- `program_id`
- `constexpr`
- `BlockedLayout`
- `SliceLayout`
- `arange`
- `load`
- `store`
- pointer arithmetic
- correctness scaffolding

#### Bucket B: target-specific path

These usually need redesign, not mechanical replacement:

- NVIDIA `async_copy`
- NVIDIA `tma`
- NVIDIA `TensorDescriptor`
- `warpgroup_mma`
- `tcgen05_*`
- `clc`
- NVIDIA matrix/tensor-memory layout classes

### Step 2: Preserve the common subset first

Do **not** rewrite common Gluon syntax just because the source came from a
NVIDIA tutorial.

The first Layer 2 sample should preserve as much as possible of:

- launcher structure
- indexing pattern
- layout-first reasoning
- correctness logic

### Step 3: Replace only the target-specific parts

When a source line clearly encodes a NVIDIA-only mechanism:

1. identify the concept
2. decide whether AMD has:
   - a conceptually similar API
   - only a weaker equivalent
   - no practical equivalent on the current target
3. choose one of:
   - keep the common Gluon subset and delete the NV-only fast path
   - replace with an AMD-specific path
   - defer the feature and document it as non-parity

## 2. Translation decision table

| Source pattern | Layer 2 rule |
|----------------|--------------|
| Common `gl` kernel structure | Keep it unless target layout assumptions are invalid |
| `threads_per_warp=[..., 32]` for a path meant to run on `gfx942` | Re-evaluate the layout using the AMD target's warp size; do not blindly keep 32-lane assumptions |
| NVIDIA `cp.async` path | Do **not** mechanically rename it; either remove it for the first sample or move to an AMD-specific async path only when the target family supports it |
| NVIDIA `tma` / `TensorDescriptor` | Not a first-sample obligation on `gfx942`; only revisit when a later branch targets descriptor-capable AMD paths |
| Hopper `warpgroup_mma` | Do not map line-by-line; if the sample is really a matrix-op migration, redesign around AMD `mfma` or `wmma` depending on target |
| Blackwell `tcgen05_*` / `TensorMemoryLayout` / `clc` | Out of scope for the first Layer 2 sample on `gfx942` |

## 3. First-sample-specific rules

The first sample is fixed to:

- `python/tutorials/gluon/03-async-copy.py`
- symbols:
  - `elementwise_add_kernel`
  - `elementwise_add`
  - `test_elementwise_add`

### What must be kept

- the host-side launcher shape
- the row/column block structure
- the correctness intent
- the benchmarkable execution structure

### What must not be brought over

- `cp.async`
- `commit_group` / `wait_group` logic from the NVIDIA path
- any requirement to prove descriptor or Tensor Core parity

### What should change

- target assumptions
- layout choices if they embed NVIDIA warp-width assumptions
- imports, if the resulting AMD version needs explicit AMD APIs

## 4. AMD target selection rules

### First sample target

- target family: `gfx942`
- architecture class: `CDNA3`

### Why this matters

The first Layer 2 sample is **not** trying to prove the richest AMD parity
surface. It is trying to prove that:

- a NVIDIA tutorial fragment can be turned into a valid AMD-facing Gluon sample
- under the same GEAK evaluation story used in Layer 1

### Target-specific guidance

#### `gfx942` / CDNA3

Prefer:

- common Gluon syntax
- explicit layout fixes
- `ttgl.amd.cdna3.buffer_load`
- `ttgl.amd.cdna3.buffer_store`
- `ttgl.amd.cdna3.mfma` only when the sample is truly matrix-op based

Avoid treating `gfx942` as if it must prove:

- `cp.async`
- `tma`
- `tdm`
- `wmma`
- `tcgen05_*`

#### `cdna4` / `gfx1250`

These are later expansion targets, not first-sample defaults.

Use them only after the first sample is successful and the Layer 2 summary says
why a stronger parity case is needed.

## 5. Example and test rules

The first Layer 2 example should be represented by:

- one **frozen NV input**
- one **AMD translation**
- one **focused test or harness**

### Why both example and test are recommended

- the example lets humans inspect the translation delta
- the test/harness lets GEAK verify compile/correctness/benchmark without
  reinterpreting a large tutorial file every time

### Minimum acceptable validation form

- compile: required
- correctness: required
- benchmark: required
- profiler: optional for the first sample

Compile-only is **not** enough to claim Layer 2 success.

## 6. Translation notes template

Each Layer 2 sample should leave a short translation note with these fields:

```markdown
## Translation notes
- source_tutorial:
- source_symbols:
- target_arch:
- kept_common_gluon_parts:
- changed_layout_parts:
- changed_runtime_or_device_assumptions:
- nv_only_parts_removed:
- amd_specific_parts_added:
- known_non_parity:
```

This keeps Layer 2 from becoming an undocumented code fork.

## 7. Anti-patterns

### Do not do this

- Rename NVIDIA APIs to guessed AMD names
- Keep NVIDIA warp-width assumptions unchanged on `gfx942`
- Claim "AMD equivalent" when the result only compiles
- Pull in `04+` tutorial hardware features when the first sample has not yet
  passed correctness
- Change runtime strategy and translation logic in the same iteration

### Be explicit instead

- If a feature is dropped, say it was dropped
- If a feature has no parity, record it
- If a sample is benchmark-valid but not profiler-ready, say so

## 8. Known non-parity classes

The following are explicitly **not first-sample obligations**:

- NVIDIA TMA parity on `gfx942`
- Hopper WGMMA parity
- Blackwell `tcgen05_*`
- Blackwell `clc`

This does not mean AMD can never support comparable concepts. It means the
first Layer 2 success should not depend on proving them.

For a public signal that AMD backend optimization and higher-performance Gluon
paths are still evolving, see the paged attention RFC:

- [RFC #8281](https://github.com/triton-lang/triton/issues/8281)

## 9. Related files

- `docs/triton_gluon_layer2_scope.md`
- `examples/triton_gluon_layer2/README.md`
- `docs/triton_gluon_writing_guide.md`
- `docs/triton_gluon_api_quick_reference.md`
