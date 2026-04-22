# Triton-Gluon

This is the single GEAK document for the Triton-family Gluon feature.

It is based on:

- the actual Gluon implementation under `/apps/qiongzhu/triton`
- the real Gluon kernels and host integration patterns under `/apps/qiongzhu/aiter`

The goal is not just to explain the API surface. The goal is to help GEAK
correctly read, translate, generate, and optimize:

- `plain_triton`
- `nv_gluon`
- `amd_gluon`

while staying grounded in what Triton and aiter actually implement today.

## 1. Product contract

GEAK treats Gluon as a **feature extension of Triton**, not as a new top-level
kernel type.

- keep `kernel_type = triton`
- use `input_dialect` to classify the input:
  - `plain_triton`
  - `nv_gluon`
  - `amd_gluon`
- use `gluon_feature_mode` as the current explicit feature gate:
  - `off`
  - `auto`
  - `force`
- prefer `amd_gluon` output when the feature is on and that output is allowed

Valid optimized outputs:

- `plain_triton`
- `amd_gluon`

GEAK should never create a new optimized `nv_gluon` output path.

## 2. Runtime, version, and compilation model

### 2.1 What Gluon is in Triton

In Triton itself, Gluon is not just syntax sugar. It has its own JIT source
path and language mode.

- `triton.experimental.gluon.jit` produces a `GluonJITFunction`
- `GluonJITFunction` uses `GluonASTSource`
- `GluonASTSource` sets the language to `Language.GLUON`
- the generated module is marked with launch-time attributes such as:
  - `ttg.target`
  - `ttg.num-warps`
  - `ttg.num-ctas`
  - `ttg.threads-per-warp`

That means host launch shape is part of the correctness contract, not a
secondary afterthought.

### 2.2 JIT and AOT are both real

Real code in aiter uses both:

- direct `@gluon.jit` kernels launched from Python
- AOT-compiled Gluon kernels wrapped from C++ / PyTorch integration code

Do not assume every Gluon path is "just JIT". In practice:

- one operator may use Gluon JIT
- another may use Gluon AOT
- a mixed pipeline may use a Gluon main kernel and a normal Triton reduce kernel

GEAK should therefore think in terms of **kernel-path compatibility**, not just
"can I write the kernel body".

### 2.3 Triton version compatibility matters

The real code under aiter explicitly carries version branches.

High-value compatibility facts:

- Gluon is expected only on newer Triton builds in practice
- Triton `3.5` and `3.6+` differ in `AMDMFMALayout.instr_shape`
- some AOT kernels built against Triton `3.5` may not load under Triton `3.6`
  because metadata requirements changed

Practical implication for GEAK:

- when reading or generating code that uses `AMDMFMALayout`, check whether the
  codebase expects a 2D or 3D `instr_shape`
- when interacting with prebuilt kernels or AOT assets, treat Triton minor
  version as part of the benchmark and integration contract

## 3. How GEAK should apply this feature

When the Gluon skill or this guide is in context, GEAK should **act**, not stop
at explanation.

1. Classify the input as `plain_triton`, `nv_gluon`, or `amd_gluon`.
2. Preserve source semantics first:
   - launcher shape
   - indexing
   - masks and boundaries
   - correctness behavior
   - benchmark intent
3. Choose the path:
   - `plain_triton`: decide whether generating an `amd_gluon` candidate is
     structurally promising
   - `nv_gluon`: translate vendor-specific APIs, layouts, descriptors, and
     memory paths into AMD-facing Gluon first, then optimize
   - `amd_gluon`: keep optimizing inside AMD Gluon space unless benchmark
     evidence strongly favors a fallback
4. Compare the result against the source baseline or an allowed fallback path.

## 4. Writing model and input paths

### 4.1 General writing model

Gluon keeps Triton's host-side launcher model:

- `@gluon.jit`
- `kernel[grid](...)`
- `triton.cdiv`
- `program_id`
- `constexpr`

What changes is that Gluon makes low-level decisions explicit:

- layout selection
- shared-memory usage
- synchronization
- descriptor and tensor-memory usage
- target-specific memory paths
- target-specific matrix instruction paths

Start from **target family + layout**, not from copied tutorial syntax.

### 4.2 `plain_triton` input

Use this path when the source is normal Triton and you want to decide whether an
AMD-facing Gluon candidate is worth generating.

Recommended order:

1. keep launcher and correctness semantics recognizable
2. recover implicit layout and memory decisions
3. make layout explicit
4. only then lower selected operations into AMD-facing Gluon

Stay in plain Triton when:

- the kernel is simple and already expresses the right execution shape
- explicit layouts add complexity without a plausible performance upside
- the target-specific path would only be compile-valid, not benchmark-valid

### 4.3 `nv_gluon` input

Treat this as a translation problem, not a rename problem.

Preserve first:

- common Gluon control flow
- launcher shape
- indexing semantics
- masks
- correctness scaffolding

Re-evaluate before carrying to AMD:

- wave32-centric layouts
- NVIDIA async-copy paths
- `tma` / descriptor assumptions
- Hopper or Blackwell-specific matrix instructions
- tensor-memory and cluster control features

The goal is not to keep a better `nv_gluon` output. The goal is to translate
into a valid `amd_gluon` candidate and then optimize it.

### 4.4 `amd_gluon` input

This is the most direct optimization path.

- keep the search inside AMD Gluon space first
- preserve the AMD-facing structure unless benchmark evidence clearly says a
  fallback should win
- prefer wave64-valid decomposition on MI3xx-class targets
- keep architecture-specific capabilities aligned with the actual target family

## 5. Layout, synchronization, and descriptor mental model

### 5.1 Layout is first-class

Real Triton Gluon code uses more than just `BlockedLayout`.

Common high-value layout concepts:

| Layout / helper | Main purpose |
|-----------------|--------------|
| `BlockedLayout` | Base thread/warp/CTA distribution |
| `SliceLayout` | Select a sub-dimension from a parent layout |
| `DotOperandLayout` | Operand layout for matrix instructions |
| `DistributedLinearLayout` | Explicit thread/register mapping when the default blocked view is not enough |
| `SwizzledSharedLayout` | Shared-memory swizzle for conflict-aware access |
| `PaddedSharedLayout` | Shared-memory layout with padding semantics |
| `PartitionedSharedLayout` | gfx1250 / RDNA-style partitioned shared-memory path |
| `AMDMFMALayout` | AMD MFMA result layout |
| `AMDWMMALayout` | AMD WMMA result layout |
| `TensorMemoryLayout` | Blackwell tensor-memory layout |

Recommended workflow:

1. choose the target family
2. choose the base layout
3. derive slice, dot-operand, shared, or descriptor layouts
4. build indices with that layout
5. lower memory or instruction paths

### 5.2 Synchronization and pipeline concepts

Useful concepts across the codebase:

- `allocate_shared_memory`
- `barrier`
- `mbarrier`
- `cluster`
- `fence_async_shared`
- `warp_pipeline_stage`

Do not assume these are interchangeable across vendors.

- Hopper and Blackwell expose one cluster/barrier family
- gfx1250 exposes another
- AMD also has optional wave or warp pipeline guidance

### 5.3 Descriptor and tensor-memory concepts

This is one of the easiest places to make wrong assumptions.

NVIDIA-side concepts:

- host `TensorDescriptor`
- device-side `tma`
- Blackwell tensor memory
- `TensorMemoryLayout`
- `tcgen05_*`

AMD-side concepts:

- gfx1250 host `TensorDescriptor`
- `tdm`
- `wmma`
- `PartitionedSharedLayout`

These are **not** 1:1 renames. Treat them as separate capability families.

## 6. NVIDIA, AMD, and architecture differences

### 6.1 Common Gluon layer

Reusable syntax usually comes from `triton.experimental.gluon.language`:

- `program_id`
- `constexpr`
- `BlockedLayout`
- `SliceLayout`
- `DotOperandLayout`
- `arange`
- `load`
- `store`
- `zeros`
- `convert_layout`
- `allocate_shared_memory`
- `barrier`

This layer gives you the kernel skeleton. Vendor-specific modules provide the
performance-critical memory or instruction paths.

### 6.2 NVIDIA families

| Family | Typical focus |
|--------|----------------|
| Ampere | `async_copy`, `mma_v2` |
| Hopper | `tma`, `mbarrier`, `cluster`, `warpgroup_mma` |
| Blackwell | `tensor_memory_descriptor`, `TensorMemoryLayout`, `tcgen05_*`, `clc`, richer TMA patterns |

Important practical notes:

- Hopper is the first place where `warpgroup_mma` and richer `tma` flows become
  central
- Blackwell adds tensor-memory and `tcgen05_*` concepts that are not just
  "Hopper but larger"
- some NVIDIA APIs are re-exported through later-generation modules, so always
  verify the actual import path in your local Triton build

NVIDIA-oriented Gluon code often assumes wave32-style execution and newer
descriptor or warpgroup features. Those assumptions must be rechecked before
moving to AMD.

### 6.3 AMD families

| Family | Typical focus |
|--------|----------------|
| CDNA3 / `gfx942` | `buffer_load`, `buffer_store`, `mfma`, `AMDMFMALayout` |
| CDNA4 / `gfx950` | CDNA3 ops plus `async_copy`, `mfma_scaled`, `get_mfma_scale_layout` |
| RDNA3 / RDNA4 | `wmma` |
| `gfx1250` | `wmma`, `wmma_scaled`, `tdm`, `async_copy`, `mbarrier`, `cluster`, `AMDWMMALayout` |

Real layout-version mapping in Triton matters:

- `AMDMFMALayout(version=1)` -> `gfx908`
- `AMDMFMALayout(version=2)` -> `gfx90a`
- `AMDMFMALayout(version=3)` -> `gfx942`
- `AMDMFMALayout(version=4)` -> `gfx950`

and:

- `AMDWMMALayout(version=1)` -> RDNA3
- `AMDWMMALayout(version=2)` -> RDNA4
- `AMDWMMALayout(version=3)` -> `gfx1250`

### 6.4 Practical differences by target

#### `gfx942` / CDNA3

Best first target for GEAK's current feature work.

- prefer explicit wave64-valid layouts
- prefer `buffer_load` / `buffer_store`
- use `mfma` only when the kernel is actually matrix-op based
- do not force descriptor or async-copy parity just because the source used it

#### `gfx950` / CDNA4

Use when the kernel truly benefits from CDNA4-only surfaces.

- can expose newer async-copy and scaled-MFMA paths
- `mfma_scaled` and `get_mfma_scale_layout` are specific high-value additions
- should not be treated as mandatory for every AMD Gluon optimization

#### `gfx1250` and RDNA-style paths

Treat these as separate families rather than as "CDNA with different names".

- `wmma` mental model differs from `mfma`
- `tdm` is not a direct rename of NVIDIA `tma`
- cluster and barrier APIs must be re-evaluated on their own terms
- current aiter Gluon code does not represent gfx1250 as a drop-in extension of
  the current CDNA3/4 kernels

### 6.5 Module path vs architecture version is not always the same thing

One subtle but important practical detail from real aiter code:

- you may see `gl.amd.cdna3.*` memory or instruction namespaces
- while the layout or arch branch still passes `AMDMFMALayout(version=4)` for
  `gfx950`

Do not assume the Python namespace name alone tells you the full architecture
contract. Read:

- the actual target arch
- the layout version
- the version guard
- any feature branch around scaled ops or async-copy

### 6.6 Concept map

These are conceptual correspondences, not 1:1 substitutions:

| Concept | NVIDIA | AMD |
|--------|---------|-----|
| Matrix path | `mma_v2`, `warpgroup_mma`, `tcgen05_mma` | `mfma`, `wmma` |
| Async global -> shared | `async_copy_*`, `tma.*` | family-specific `async_copy.*` |
| Descriptor-like path | `tma`, `TensorDescriptor` | `tdm` |
| Barrier / cluster | `mbarrier`, `cluster` | family-specific barrier / cluster APIs |
| Tensor memory | Blackwell tensor memory | no direct global AMD equivalent; use gfx1250-specific `tdm` / WMMA family where appropriate |

## 7. Real patterns from aiter

### 7.1 Attention path on `gfx942` / `gfx950`

The paged-attention Gluon code in aiter is useful because it shows real
production-style composition:

- `BlockedLayout`
- `SliceLayout`
- `AMDMFMALayout(version=CDNA_VERSION, ...)`
- `DotOperandLayout`
- `allocate_shared_memory`
- shared-memory swizzle
- `buffer_load` / `buffer_store`
- explicit stride-rich host arguments

This is more representative than a toy vector-add kernel.

### 7.2 GEMM and FP8 paths on `gfx950`

The GEMM side adds patterns that the current GEAK docs should explicitly cover:

- `mfma_scaled`
- `get_mfma_scale_layout`
- architecture-conditioned K widths and instruction shapes
- JSON-config or heuristic-driven launch selection instead of only online
  autotune

### 7.3 JIT and AOT can coexist in one operator family

aiter uses:

- pure JIT Gluon kernels
- AOT-compiled Gluon kernels
- mixed pipelines where only some stages are Gluon

So GEAK should not force every downstream rewrite into one execution mode.

### 7.4 Optional scheduling hints are real but secondary

aiter also carries optional scheduling or barrier-style hints in some paths.

Treat these as:

- second-stage tuning features
- target-specific hints
- not first-pass mandatory portability features

## 8. Representative examples

These examples are intentionally different from the checked-in test fixtures.
They are documentation examples, not golden outputs.

### 8.1 Example: `plain_triton -> amd_gluon` row-wise affine transform

Suppose the source kernel applies `output[row, col] = input[row, col] * scale[row] + bias[row]`.
The plain Triton version may leave layout implicit. The AMD-facing Gluon version
should make the one-dimensional row tile explicit before lowering memory ops.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.jit
def row_affine_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_cols,
    COL_BLOCK: ttgl.constexpr,
    layout: ttgl.constexpr,
):
    row = ttgl.program_id(0)
    cols = ttgl.arange(0, COL_BLOCK, layout=layout)
    mask = cols < n_cols

    x = ttgl.amd.cdna3.buffer_load(ptr=x_ptr + row * n_cols, offsets=cols, mask=mask, other=0.0)
    scale = ttgl.load(scale_ptr + row)
    bias = ttgl.load(bias_ptr + row)
    y = x * scale + bias
    ttgl.amd.cdna3.buffer_store(
        ptr=out_ptr + row * n_cols,
        offsets=cols,
        stored_value=y,
        mask=mask,
    )
```

What this example is showing:

- the launcher stays Triton-like
- the layout is explicit
- the memory path is AMD-facing
- the operation is not just vector add, so it does not duplicate the checked-in
  example fixtures

### 8.2 Example: `nv_gluon -> amd_gluon` tile reader rewrite

Suppose an NVIDIA-facing Gluon kernel reads a two-dimensional tile using a
wave32-centric blocked layout and later plans to use an async-copy path. The
first AMD rewrite should often keep the tile logic but replace the layout and
drop the unsupported fast path until parity is justified.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.jit
def tile_reader_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    n_cols,
    ROW_BLOCK: ttgl.constexpr,
    COL_BLOCK: ttgl.constexpr,
    layout: ttgl.constexpr,
):
    pid = ttgl.program_id(0)
    row_offsets = pid * ROW_BLOCK + ttgl.arange(0, ROW_BLOCK, ttgl.SliceLayout(1, layout))
    col_offsets = ttgl.arange(0, COL_BLOCK, ttgl.SliceLayout(0, layout))
    mask = (row_offsets < n_rows)[:, None] & (col_offsets < n_cols)[None, :]

    tile = ttgl.amd.cdna3.buffer_load(
        ptr=in_ptr,
        offsets=[row_offsets[:, None], col_offsets[None, :]],
        mask=mask,
        other=0.0,
    )
    ttgl.amd.cdna3.buffer_store(
        ptr=out_ptr,
        offsets=[row_offsets[:, None], col_offsets[None, :]],
        stored_value=tile,
        mask=mask,
    )
```

What this example is showing:

- preserve the common tile logic first
- switch to an AMD-valid layout and memory path
- do not mechanically carry over an NVIDIA-only async or descriptor path

### 8.3 Example: Triton-version-compatible MFMA layout guard

Some real code has to survive Triton minor-version differences in
`AMDMFMALayout.instr_shape`.

```python
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

TRITON_VERSION_GE_3_6_0 = tl.constexpr(True)

@gluon.jit
def mfma_guarded_kernel(x_ptr, y_ptr, out_ptr, K_BLOCK: gl.constexpr, CDNA_VERSION: gl.constexpr):
    if TRITON_VERSION_GE_3_6_0:
        instr_shape: gl.constexpr = [16, 16, K_BLOCK]
    else:
        instr_shape: gl.constexpr = [16, 16]

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=instr_shape,
        transposed=True,
        warps_per_cta=[1, 4],
    )
```

What this example is showing:

- Triton version can be part of the kernel-generation contract
- layout generation may need compatibility guards
- this is especially relevant for mixed JIT and AOT environments

### 8.4 Example: JIT with explicit AOT fallback

Some production code wants a JIT path when `triton.experimental.gluon` exists,
but a fallback path when only prebuilt kernels are available.

```python
try:
    from triton.experimental import gluon
    GLUON_JIT_ENABLED = True
except ImportError:
    GLUON_JIT_ENABLED = False

def launch_or_fallback(x, y, out):
    if GLUON_JIT_ENABLED:
        grid = (1,)
        return gluon_kernel[grid](x, y, out, num_warps=4)
    return call_prebuilt_gluon_aot_kernel(x, y, out)
```

What this example is showing:

- JIT availability is not guaranteed in every downstream environment
- some real integrations intentionally carry both paths
- GEAK should preserve that decision surface when the codebase already depends
  on it

## 9. Current repo-local notes

GEAK's current checked-in defaults for this feature are intentionally small:

- default target backend: `hip/gfx942`
- current repo-local profile focus: `mi3xx`
- example surface: `examples/triton_gluon_inputs/`
- no checked-in golden outputs
- no checked-in run manifests, preprocess snapshots, profile JSON, or benchmark
  dumps in `examples/`

The accepted checked-in input forms are:

- `plain_triton`
- `nv_gluon`
- `amd_gluon`

GEAK should generate and benchmark candidate outputs at run time.

## 10. Benchmark-aware rules

- compile-only success is not enough to claim the path is valid
- preserve correctness before optimizing for speed
- when a harness supports multiple modes, keep one ordered case stream across
  correctness, profile, and benchmark
- if `plain_triton` remains an allowed output, compare it against `amd_gluon`
  instead of assuming AMD Gluon wins automatically
- treat Triton minor version, JIT vs AOT availability, and target architecture
  as part of the benchmark contract

## 11. Anti-patterns

- introducing a new top-level `gluon` kernel type
- treating conceptual correspondences as 1:1 API renames
- keeping NVIDIA layout assumptions unchanged on AMD
- mixing AMD and NVIDIA layout families in one kernel path
- hardcoding repo-local absolute paths, branch names, or container names in
  reusable syntax guidance
- treating checked-in examples or optimization logs as benchmark truth
- assuming every AMD target should use the same CDNA3/4-style recipe
- assuming descriptor, tensor-memory, or cluster APIs are portable by name

## Appendix A: API usage

### A.1 Imports

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

AMD-heavy code often uses:

```python
from triton.experimental.gluon import language as ttgl
```

### A.2 JIT entry and host launcher

```python
@gluon.jit
def kernel(x_ptr, y_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel
    x = gl.load(x_ptr + offsets, mask=mask)
    gl.store(y_ptr + offsets, x, mask=mask)

def launch(x, y, XBLOCK=256, num_warps=4, num_ctas=1):
    xnumel = x.numel()
    grid = (triton.cdiv(xnumel, XBLOCK),)
    kernel[grid](x, y, xnumel, XBLOCK, num_warps=num_warps, num_ctas=num_ctas)
```

Relevant runtime facts:

- `GluonASTSource` uses `Language.GLUON`
- launch attributes include target, warps, CTAs, and threads-per-warp

### A.3 Core language and layout surface

| API | Purpose |
|-----|---------|
| `BlockedLayout` | Base thread/warp/CTA distribution |
| `SliceLayout` | Select a sub-dimension from a parent layout |
| `DotOperandLayout` | Operand layout required by matrix instructions |
| `DistributedLinearLayout` | Explicit register and lane mapping |
| `SwizzledSharedLayout` | Shared-memory swizzle |
| `PaddedSharedLayout` | Shared-memory padding semantics |
| `PartitionedSharedLayout` | gfx1250-style partitioned shared layout |
| `AMDMFMALayout` | AMD MFMA result layout |
| `AMDWMMALayout` | AMD WMMA result layout |
| `TensorMemoryLayout` | Blackwell tensor-memory layout |
| `convert_layout` | Move a tensor into another layout |
| `zeros(..., layout=...)` | Create accumulator/register tensors with explicit layout |
| `to_linear_layout` | Convert to a linearized layout view when needed |
| `set_auto_layout` | Allow automatic layout selection in constrained flows |

Recommended order:

1. choose the layout
2. build indices with that layout
3. load values
4. convert layout if needed
5. run the target-specific op

### A.4 Shared memory, synchronization, and cluster surface

```python
smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)
copy_op(...)
commit_group()
wait_group(0)
value = smem.load(layout)
```

Common concepts:

- `allocate_shared_memory`
- `barrier`
- `mbarrier`
- `cluster`
- `fence_async_shared`
- `warp_pipeline_stage`

The high-level phase structure is usually:

1. issue async transfer
2. commit
3. wait
4. consume from shared memory or continue compute

### A.5 Descriptor and tensor-memory surface

NVIDIA-side:

- host `TensorDescriptor`
- device `tma`
- `tensor_memory_descriptor`
- `TensorMemoryLayout`
- `tcgen05_*`

AMD-side:

- gfx1250 host `TensorDescriptor`
- `tdm`
- `async_load`
- `async_wait`
- `prefetch`
- `async_scatter`

### A.6 AMD quick patterns

#### CDNA3 MFMA pattern

```python
blocked = ttgl.BlockedLayout(
    size_per_thread=[4, 4],
    threads_per_warp=[4, 16],
    warps_per_cta=[num_warps, 1],
    order=[1, 0],
)
mfma_layout = ttgl.amd.AMDMFMALayout(
    version=3,
    instr_shape=[32, 32, 8],
    transposed=True,
    warps_per_cta=[num_warps, 1],
)
a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a)
b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, mfma_layout, 4))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, mfma_layout, 4))
acc = ttgl.zeros([M, N], ttgl.float32, mfma_layout)
c = ttgl.amd.cdna3.mfma(a, b, acc)
```

#### CDNA4 scaled-MFMA pattern

```python
mfma_layout = ttgl.amd.AMDMFMALayout(
    version=4,
    instr_shape=[16, 16, 32],
    transposed=True,
    warps_per_cta=[1, 4],
)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, mfma_layout, 16))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, mfma_layout, 16))
a_scale = ttgl.amd.cdna4.get_mfma_scale_layout(a.type.layout, [M, K])
b_scale = ttgl.amd.cdna4.get_mfma_scale_layout(b.type.layout, [K, N])
c = ttgl.amd.cdna4.mfma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
```

#### `gfx1250` WMMA pattern

```python
wmma_layout = ttgl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, instr_shape_k])
a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, wmma_layout, k_width))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, wmma_layout, k_width))
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

### A.7 NVIDIA quick patterns

#### Ampere or Hopper async copy

```python
cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
cp.commit_group()
cp.wait_group(0)
value = smem.load(layout)
```

#### Hopper WGMMA pattern

```python
fence_async_shared()
acc = warpgroup_mma(a, b_smem, acc, is_async=True)
warpgroup_mma_wait(acc, deps=(...))
```

#### Blackwell tensor-memory or TCGen05 pattern

```python
tmem = allocate_tensor_memory(...)
tcgen05_copy(...)
tcgen05_mma(...)
tcgen05_commit(...)
```

### A.8 Version and compatibility checklist

- check the installed Triton version before assuming a Gluon feature exists
- treat Triton `3.5` vs `3.6+` as a compatibility boundary for some Gluon code
- confirm whether `AMDMFMALayout.instr_shape` is expected in 2D or 3D form
- confirm whether the code path expects JIT, AOT, or both
- confirm target backend, arch, and warp or wave assumptions before changing
  layouts

### A.9 Common pitfalls

#### General

- do not start from copied tutorial code without checking the target family
- do not treat compile or IR validation as proof that profiling will work
- do not mix environment changes, harness changes, and kernel changes in one
  unexplained iteration

#### NVIDIA-side

- do not assume `tma` concepts port directly to AMD
- do not assume Hopper and Blackwell matrix paths are interchangeable
- do not forget async ordering or fence semantics around shared-memory paths

#### AMD-side

- do not mix `mfma` and `wmma` mental models
- do not assume descriptor-style paths are uniformly available across CDNA3,
  CDNA4, and `gfx1250`
- do not assume every benchmark-valid task is profiler-ready
- do not assume the namespace name alone fully determines the architecture
  contract
