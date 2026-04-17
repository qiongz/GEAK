# Triton-Gluon API Quick Reference

This is a compact reference for agents and contributors who need a short,
high-signal summary of Triton-Gluon writing patterns.

For the longer explanation, see `docs/triton_gluon_writing_guide.md`.

## 1. Core syntax

### Imports

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

AMD-heavy code often uses:

```python
from triton.experimental.gluon import language as ttgl
```

### Basic kernel skeleton

```python
@gluon.jit
def kernel(x_ptr, y_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel
    x = gl.load(x_ptr + offsets, mask=mask)
    gl.store(y_ptr + offsets, x, mask=mask)
```

### Host launcher skeleton

```python
def launch(x, y, XBLOCK=256, num_warps=4):
    xnumel = x.numel()
    grid = (triton.cdiv(xnumel, XBLOCK),)
    kernel[grid](x, y, xnumel, XBLOCK, num_warps=num_warps)
```

## 2. Layout vocabulary

| Term | Purpose |
|------|---------|
| `BlockedLayout` | Base thread/warp/CTA distribution |
| `SliceLayout` | Select a sub-dimension from a parent layout |
| `DotOperandLayout` | Operand layout required by matrix instructions |
| `convert_layout` | Move a tensor into another layout |
| `zeros(..., layout=...)` | Create accumulator/register tensors with explicit layout |

### Layout-first rule

The most stable Gluon writing principle is:

1. choose the layout
2. build indices with that layout
3. load values
4. convert layout if needed
5. run the target-specific op

## 3. Shared memory and async pattern

### Common pattern

```python
smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)
copy_op(...)
commit_group()
wait_group(0)
value = smem.load(layout)
```

The phase structure is stable across vendors even when exact API names differ:

1. issue async transfer
2. commit
3. wait
4. consume from shared / continue compute

## 4. Architecture families

### NVIDIA

| Family | Typical modules / ops |
|--------|------------------------|
| Ampere | `async_copy`, `mma_v2` |
| Hopper | `tma`, `mbarrier`, `cluster`, `warpgroup_mma` |
| Blackwell | `tcgen05_*`, `TensorMemoryLayout`, `clc`, richer TMA gather/scatter |

### AMD

| Family | Typical modules / ops |
|--------|------------------------|
| CDNA3 | `buffer_load`, `buffer_store`, `mfma`, `AMDMFMALayout` |
| CDNA4 | CDNA3 ops + `async_copy`, `mfma_scaled`, `get_mfma_scale_layout` |
| RDNA3 | `wmma` |
| RDNA4 | `wmma` |
| gfx1250 | `wmma`, `wmma_scaled`, `tdm`, `async_copy`, `mbarrier`, `cluster`, `AMDWMMALayout` |

## 5. Concept map: same idea, different API

| Concept | NVIDIA | AMD |
|--------|---------|-----|
| Matrix op | `mma_v2`, `warpgroup_mma`, `tcgen05_mma` | `mfma`, `wmma` |
| Async global -> shared | `async_copy_global_to_shared`, `tma.*` | `cdna4.async_copy.*`, `gfx1250.async_copy.*` |
| Descriptor path | `tma`, `TensorDescriptor` | `gfx1250.tdm` |
| Barrier / cluster | `mbarrier`, `cluster` | `gfx1250.mbarrier`, `gfx1250.cluster` |
| Scaled low-precision matrix path | `tcgen05_mma_scaled` | `mfma_scaled`, `wmma_scaled` |

Important: these are **conceptual** correspondences, not 1:1 API substitutions.

## 6. AMD quick patterns

### CDNA3 MFMA pattern

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

### gfx1250 WMMA pattern

```python
wmma_layout = ttgl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, instr_shape_k])
a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, wmma_layout, k_width))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, wmma_layout, k_width))
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

## 7. NVIDIA quick patterns

### Ampere/Hopper-style async copy

```python
cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
cp.commit_group()
cp.wait_group(0)
value = smem.load(layout)
```

### Hopper WGMMA pattern

```python
fence_async_shared()
acc = warpgroup_mma(a, b_smem, acc, is_async=True)
warpgroup_mma_wait(acc, deps=(...))
```

### Blackwell TCGen05 pattern

```python
tmem = allocate_tensor_memory(...)
tcgen05_copy(...)
tcgen05_mma(...)
tcgen05_commit(...)
```

## 8. Common pitfalls

### General

- Do not start from copied tutorial code without first checking target family.
- Do not treat compile/IR validation as proof that profiling will work.
- Do not mix environment changes, harness changes, and kernel changes in one iteration.

### NVIDIA-specific pitfalls

- Do not assume `tma` concepts port directly to AMD.
- Do not assume Hopper and Blackwell matrix paths are interchangeable.
- Do not forget proxy ordering / fence semantics around async shared paths.

### AMD-specific pitfalls

- Do not mix `mfma` and `wmma` mental models.
- Do not assume descriptor-style paths are uniformly available across CDNA3/CDNA4 and gfx1250.
- Do not assume every benchmark-valid task is Metrix-ready.

## 9. Use this reference with

- `skills/triton-gluon-mi3xx/SKILL.md` for reusable agent-facing guidance
- `docs/triton_gluon_writing_guide.md` for the longer explanation
- `docs/triton_gluon_layer1_scope.md` for what Layer 1 has already proven
- `docs/triton_gluon_mi3xx_baseline.md` for the current repo-specific workflow
