---
name: triton-gluon-mi3xx
description: Use when writing, reviewing, adapting, or evaluating Triton-Gluon kernels, especially when the task needs a concise guide to Gluon syntax, explicit layouts, shared memory, synchronization, and architecture-specific APIs across AMD and NVIDIA.
tier: authoring_safe
---

# Triton-Gluon Writing Guide

## When to use
- You are writing or adapting a Gluon kernel.
- You need to decide which parts of a kernel are common Gluon syntax vs target-specific APIs.
- You are reviewing AMD/NVIDIA Gluon code and need a compact architecture map.
- You are wrapping an existing Gluon test into a GEAK harness and want the code shape to stay close to upstream Gluon style.
- You are lifting a plain Triton kernel into an AMD Gluon implementation.

## Mental model
- Gluon shares Triton's host-side launcher model: `@gluon.jit`, `kernel[grid](...)`, `triton.cdiv`, `program_id`, and `constexpr` arguments still matter.
- Gluon differs from Triton by making low-level choices explicit:
  - layout selection
  - shared memory allocation and reuse
  - synchronization
  - target-specific coarse-grained ops like async copy, matrix instructions, and tensor-descriptor paths
- Start from **layout + target family**, not from a copied tutorial snippet.

## Stable syntax patterns

### 1. Basic kernel shape
```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def kernel(x_ptr, y_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel
    x = gl.load(x_ptr + offsets, mask=mask)
    gl.store(y_ptr + offsets, x, mask=mask)
```

- Keep host-side launch style close to Triton: `grid`, `triton.cdiv`, `num_warps`.
- Use `constexpr` for block sizes, layouts, and instruction-specific metadata.
- Treat `program_id`, pointer arithmetic, and masks as first-class parts of the kernel.

### 2. Layout-first design
- In Gluon, layout is not an afterthought. Choose a `BlockedLayout` first, then derive:
  - `SliceLayout`
  - `DotOperandLayout`
  - target-specific matrix layouts
- Prefer:
  - `arange(..., layout=...)`
  - `zeros(..., layout=...)`
  - `convert_layout(...)`
  over ad-hoc index juggling.
- The most common failure mode in Gluon is picking the wrong layout family for the instruction path.

### 3. Shared memory and async paths
- Shared memory is explicit:
  - `allocate_shared_memory(...)`
  - shared layout types like `SwizzledSharedLayout`
- Async movement follows a common pattern even when API names differ:
  1. issue async copy
  2. `commit_group()`
  3. `wait_group(...)`
  4. read from shared / continue compute

### 4. Correctness and benchmark patterns
- For correctness, compare against a trusted reference such as `torch.matmul`, eager PyTorch, or the upstream test oracle.
- For benchmark-capable harnesses:
  - preserve one ordered case stream
  - reuse that same case stream across correctness/profile/benchmark/full-benchmark
  - emit `GEAK_SHAPES_USED=[...]`
  - end with `GEAK_RESULT_LATENCY_MS=<number>`
- Compile-only IR validation is useful, but do not confuse it with a profiler-ready kernel path.

## Common Gluon vocabulary
- `@gluon.jit`
- `program_id`
- `constexpr`
- `BlockedLayout`
- `SliceLayout`
- `DotOperandLayout`
- `convert_layout`
- `allocate_shared_memory`
- `load`, `store`, `zeros`
- `num_warps`, `num_ctas`, `triton.cdiv`

## Architecture split

### Common Gluon layer
- Common syntax lives under `triton.experimental.gluon.language`:
  - `load`, `store`
  - `arange`, `zeros`
  - `BlockedLayout`, `SliceLayout`, `DotOperandLayout`
  - `allocate_shared_memory`
  - `convert_layout`
- The common layer gives the kernel skeleton. Performance-critical ops usually come from the architecture submodule.

### NVIDIA families
- **Ampere**:
  - async copy
  - `mma_v2`
- **Hopper**:
  - `tma`
  - `mbarrier`
  - `cluster`
  - `warpgroup_mma`
- **Blackwell**:
  - `tcgen05_*`
  - `TensorMemoryLayout`
  - `clc`
  - richer TMA gather/scatter paths

### AMD families
- **CDNA3**:
  - `ttgl.amd.cdna3.buffer_load`
  - `ttgl.amd.cdna3.buffer_store`
  - `ttgl.amd.cdna3.mfma`
  - `ttgl.amd.AMDMFMALayout`
- **CDNA4**:
  - CDNA3 primitives
  - `async_copy`
  - `mfma_scaled`
  - `get_mfma_scale_layout`
- **RDNA3 / RDNA4**:
  - `wmma`
- **gfx1250**:
  - `wmma`
  - `wmma_scaled`
  - `tdm`
  - `async_copy`
  - `mbarrier`
  - `cluster`
  - `ttgl.amd.AMDWMMALayout`

## Concept mapping: same idea, different names
- **Matrix op family**
  - NVIDIA: `mma_v2`, `warpgroup_mma`, `tcgen05_mma`
  - AMD: `mfma`, `wmma`
- **Descriptor-driven async path**
  - NVIDIA: `tma`
  - AMD: `gfx1250.tdm`
  - Do not assume CDNA3/CDNA4 has full TMA-equivalent API parity.
- **Global <-> shared async movement**
  - NVIDIA tutorials often use `async_copy_global_to_shared`
  - AMD paths are split across `cdna4.async_copy` and `gfx1250.async_copy`
- **Cluster / barrier**
  - Both vendors expose cluster-like and barrier-like concepts
  - Signatures and required companion ops are not identical

## Example patterns

### NVIDIA-style async copy pattern
```python
smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)
cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
cp.commit_group()
cp.wait_group(0)
value = smem.load(layout)
gl.store(out_ptr + offsets, value, mask=mask)
```

### AMD CDNA MFMA pattern
```python
blocked = ttgl.BlockedLayout(size_per_thread=[4, 4], threads_per_warp=[4, 16], warps_per_cta=[num_warps, 1], order=[1, 0])
mfma_layout = ttgl.amd.AMDMFMALayout(version=3, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[num_warps, 1])
a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a)
b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, mfma_layout, 4))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, mfma_layout, 4))
acc = ttgl.zeros([M, N], ttgl.float32, mfma_layout)
c = ttgl.amd.cdna3.mfma(a, b, acc)
```

### AMD gfx1250 WMMA pattern
```python
blocked = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
wmma_layout = ttgl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, instr_shape_k])
a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, wmma_layout, k_width))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, wmma_layout, k_width))
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

## Architecture selection checklist
1. Detect `backend`, `arch`, and `warp_size` first.
2. Pick one family-specific module:
   - `gl.nvidia.hopper`, `gl.nvidia.blackwell`
   - `gl.amd.cdna3`, `gl.amd.cdna4`, `gl.amd.gfx1250`, `gl.amd.rdna3`, `gl.amd.rdna4`
3. Choose the matching layout class before selecting the matrix instruction.
4. Keep one kernel path target-specific. Do not mix AMD and NVIDIA layout families in the same code path.
5. If the code is compile-only or IR-only, do not present it as profiler-ready.

## Layer 3 direct-lift notes
- When the source is plain Triton, preserve the launcher shape, indexing, masks,
  correctness oracle, and benchmark intent before changing lower-level details.
- Plain Triton leaves layout and memory decisions implicit more often than
  Gluon. Recover those decisions explicitly before replacing `tl.load` /
  `tl.store`.
- On `gfx942`, prefer wave64-valid layouts. For simple first-sample paths,
  prefer common Gluon indexing plus `ttgl.amd.cdna3.buffer_load` /
  `ttgl.amd.cdna3.buffer_store`.
- Keep demo prints, plots, and notebook-style presentation logic out of the AMD
  example when the harness can own them instead.
- If a direct lift drops or weakens a source path, record it as a deliberate
  non-parity rather than renaming APIs and implying full equivalence.

## Anti-patterns
- Do not assume NVIDIA tutorial code ports 1:1 to AMD.
- Do not treat `tma` and `tdm` as the same API with renamed symbols.
- Do not mix `AMDMFMALayout`, `AMDWMMALayout`, and NVIDIA matrix layout classes in one path.
- Do not bury current container names, branch names, or absolute repo paths inside a reusable syntax skill.
- Do not change environment, harness contract, task choice, and kernel code in the same step.

## Current repo-specific defaults
- For the current `/apps/qiongzhu/triton` MI3xx baseline workflow, see `docs/triton_gluon_mi3xx_baseline.md`.
- For what the current Layer 1 scope has already proven, see `docs/triton_gluon_layer1_scope.md`.
- For the bounded NV→AMD translation layer, see `docs/triton_gluon_layer2_scope.md`.
- For the Layer 2 playbook, see `docs/triton_gluon_translation_rules.md`.
- For the bounded plain-Triton→AMD direct-lift layer, see `docs/triton_gluon_layer3_scope.md`.
- For the Layer 3 direct-lift playbook, see `docs/triton_gluon_lift_rules.md`.
- For a compact API and architecture cheat sheet, see `docs/triton_gluon_api_quick_reference.md`.
- For a longer architecture-and-writing reference, see `docs/triton_gluon_writing_guide.md`.
