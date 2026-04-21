# Triton-Gluon Writing Guide

This document summarizes the writing model of Triton-Gluon before it is
collapsed into shorter skill-level guidance. The goal is to separate:

- reusable language and API knowledge
- architecture-specific patterns
- current-repo execution policy

The first two belong in a skill. The third belongs in baseline/runbook docs.

For adjacent summaries:

- `docs/triton_gluon_layer1_scope.md` explains what the current Layer 1 work
  has already proven.
- `docs/triton_gluon_layer2_scope.md` defines the bounded NV-Gluon to AMD-Gluon translation layer.
- `docs/triton_gluon_translation_rules.md` contains the Layer 2 playbook and first-sample constraints.
- `docs/triton_gluon_layer3_scope.md` defines the bounded plain-Triton to AMD-Gluon direct-lift layer.
- `docs/triton_gluon_lift_rules.md` contains the Layer 3 direct-lift playbook and first-sample constraints.
- `docs/triton_gluon_api_quick_reference.md` provides a compact syntax and
  architecture cheat sheet.

## 1. Core mental model

Gluon shares Triton's Python DSL, JIT model, and host-side launcher shape:

- `@gluon.jit`
- `kernel[grid](...)`
- `triton.cdiv(...)`
- `program_id(...)`
- `constexpr` kernel parameters

What Gluon changes is the level of control. The user must make more decisions
explicit:

- tensor layout
- shared-memory layout
- synchronization
- matrix-instruction path
- target-specific coarse-grained operations

In practice, this means you should design a Gluon kernel from:

1. target family
2. layout plan
3. memory path
4. instruction path

not from a copied high-level Triton loop body.

When the input is plain Triton rather than Gluon, the rule is stricter: keep
the source kernel's launcher, indexing, and correctness intent, but recover the
target/layout/memory decisions explicitly instead of transliterating `tl.*`
syntax line by line.

## 2. Stable syntax patterns

### 2.1 Basic kernel skeleton

From `python/tutorials/gluon/01-intro.py`:

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

Key takeaways:

- host-side launch still looks like Triton
- `program_id` and `grid` stay central
- `constexpr` still carries block/meta choices
- offsets and masks are still explicit
- layout is explicit much earlier than in ordinary Triton code

### 2.2 Layout-first design

From `python/tutorials/gluon/02-layouts.py` and `python/test/gluon/test_core.py`:

- choose a `BlockedLayout` first
- derive `SliceLayout` from it when indexing sub-dimensions
- derive `DotOperandLayout` from the matrix-op layout
- use `convert_layout(...)` before the matrix op

Good Gluon style usually looks like:

1. define the blocked/register/shared layout
2. build indices with `arange(..., layout=...)`
3. load tensors using those indices
4. convert to operand layout if a matrix op needs it
5. run the target-specific instruction

### 2.3 Shared memory and async movement

From `python/tutorials/gluon/03-async-copy.py`:

```python
smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)
cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
cp.commit_group()
cp.wait_group(0)
value = smem.load(layout)
gl.store(out_ptr + offsets, value, mask=mask)
```

The exact function names vary by target, but the phase structure is stable:

1. issue async copy
2. `commit_group()`
3. `wait_group(...)`
4. load from shared / continue compute

### 2.4 Correctness and benchmark style

Across tests and tutorials:

- correctness typically compares with eager PyTorch or another trusted reference
- performance usually uses:
  - `torch.cuda.Event`
  - or `triton.testing.do_bench`
- warmup + median is the common measurement pattern

For GEAK harnesses, keep one ordered case stream and reuse it across:

- `--correctness`
- `--profile`
- `--benchmark`
- `--full-benchmark`

## 3. Architecture map

### 3.1 Common Gluon layer

Common Gluon vocabulary includes:

- `@gluon.jit`
- `program_id`
- `constexpr`
- `load`, `store`, `zeros`
- `BlockedLayout`
- `SliceLayout`
- `DotOperandLayout`
- `convert_layout`
- `allocate_shared_memory`

These define the kernel structure. The instruction path usually comes from the
target-specific submodule.

### 3.2 NVIDIA families

#### Ampere

- async copy path
- `mma_v2`

Representative file:

- `python/tutorials/gluon/03-async-copy.py`

#### Hopper

- `tma`
- `mbarrier`
- `cluster`
- `warpgroup_mma`

Representative files:

- `python/tutorials/gluon/04-tma.py`
- `python/tutorials/gluon/05-wgmma.py`

#### Blackwell

- `tcgen05_*`
- `TensorMemoryLayout`
- `TensorMemoryScalesLayout`
- `clc`
- richer TMA gather/scatter paths

Representative files:

- `python/tutorials/gluon/06-tcgen05.py`
- `python/tutorials/gluon/12-cluster-launch-control.py`

### 3.3 AMD families

#### CDNA3

Representative API surface:

- `ttgl.amd.cdna3.buffer_load`
- `ttgl.amd.cdna3.buffer_store`
- `ttgl.amd.cdna3.mfma`
- `ttgl.amd.AMDMFMALayout`

Representative file:

- `python/test/gluon/test_core.py`

Example pattern:

```python
a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a)
b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b)
a = ttgl.convert_layout(a, dot_a_layout)
b = ttgl.convert_layout(b, dot_b_layout)
acc = ttgl.zeros([M, N], ttgl.float32, mfma_layout)
c = ttgl.amd.cdna3.mfma(a, b, acc)
ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c)
```

#### CDNA4

Representative API surface:

- CDNA3 buffer ops
- `async_copy`
- `mfma_scaled`
- `get_mfma_scale_layout`

Representative file:

- `python/test/gluon/test_core.py`

#### RDNA3 / RDNA4

Representative API surface:

- `wmma`

Representative file references appear in:

- `python/test/gluon/test_core.py`
- `python/test/gluon/test_frontend.py`

#### gfx1250

Representative API surface:

- `ttgl.amd.gfx1250.wmma`
- `wmma_scaled`
- `tdm`
- `async_copy`
- `mbarrier`
- `cluster`
- `ttgl.amd.AMDWMMALayout`

Representative file:

- `third_party/amd/python/test/test_gluon_gfx1250.py`

Example pattern:

```python
WMMA_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, INSTR_SHAPE_K])
a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
a = ttgl.convert_layout(a, ttgl.DotOperandLayout(0, WMMA_LAYOUT, K_WIDTH))
b = ttgl.convert_layout(b, ttgl.DotOperandLayout(1, WMMA_LAYOUT, K_WIDTH))
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

## 4. Concept mapping: same idea, different names

### Matrix ops

- NVIDIA:
  - `mma_v2`
  - `warpgroup_mma`
  - `tcgen05_mma`
- AMD:
  - `mfma`
  - `wmma`

Do not treat these as a single interchangeable API. The associated layout types
and hardware assumptions differ.

### Descriptor-driven async paths

- NVIDIA: `tma`
- AMD: `gfx1250.tdm`

This is an important place where "same concept" does **not** mean "same level of
coverage". AMD's descriptor-style path is strongest on `gfx1250`; do not assume
CDNA3/CDNA4 exposes the same path that Hopper TMA does.

### Global <-> shared async movement

- NVIDIA tutorials often use `async_copy_global_to_shared`
- AMD paths are split across:
  - `cdna4.async_copy`
  - `gfx1250.async_copy`

### Barriers and cluster sync

Both ecosystems expose barrier/cluster-like building blocks, but:

- signatures differ
- supporting ops differ
- mbarrier stories are not 1:1 portable

## 5. Direct lift from plain Triton

Layer 3 starts from plain Triton rather than Gluon. That changes the job from
"retarget an existing Gluon program" to "recover the implicit decisions that
plain Triton left to the compiler, then make them explicit for AMD Gluon."

### 5.1 What to preserve first

When lifting plain Triton, preserve these semantics first:

- host launcher shape
- `program_id` usage
- pointer arithmetic
- offset generation
- mask semantics
- correctness oracle
- benchmark intent

### 5.2 What must become explicit

Before lowering `tl.load` / `tl.store`, choose:

1. target family
2. wave / warp assumptions
3. `BlockedLayout`
4. any derived `SliceLayout`
5. memory path

On `gfx942`, prefer wave64-valid layouts. For simple first-sample memory paths,
prefer common Gluon indexing plus `ttgl.amd.cdna3.buffer_load` /
`ttgl.amd.cdna3.buffer_store` rather than guessed AMD-only rewrites.

### 5.3 What should move out of the example file

Plain Triton tutorials often include demo prints, plots, or notebook-style
execution. Keep the benchmark intent, but move those presentation details into
the harness or run manifest when they are not part of the kernel contract.

### 5.4 What to document explicitly

Each direct lift should record:

- what Triton semantics were kept
- what layout decisions were added
- what AMD-specific memory or instruction path was chosen
- what plain Triton pieces were intentionally not carried over verbatim
- what remains non-parity

## 6. What is stable enough for a skill

The following belongs in a reusable Gluon skill:

- the mental model
- layout-first design
- common syntax building blocks
- architecture split
- concept mapping
- common anti-patterns

The following does **not** belong in the reusable syntax part of the skill:

- container names
- branch names
- absolute repo paths
- current P0 canary order
- current cache or venv paths

Those belong in:

- `docs/triton_gluon_mi3xx_baseline.md`
- `examples/triton_gluon_mi3xx/README.md`
- run manifests
- execution runbooks

## 7. Known non-equivalences and current caution points

1. NVIDIA tutorial code does not port 1:1 to AMD.
2. `tma` and `tdm` are conceptually related, but not drop-in equivalents.
3. Compile-only IR validation is useful, but it is not profiler-ready execution.
4. A task can be correctness-ready and benchmark-ready without being Metrix-ready.
5. The public AMD optimization direction is still evolving. For example, the
   paged attention RFC and follow-up discussions point to ongoing AMD backend
   work around matmul partitioning, layout conversion, and memory movement
   optimization rather than a finished 1:1 parity story with mature NV paths:
   [RFC #8281](https://github.com/triton-lang/triton/issues/8281)

## 8. Current repo-specific defaults

For the current `/apps/qiongzhu/triton` MI3xx baseline workflow, see:

- `docs/triton_gluon_mi3xx_baseline.md`

That document keeps the current-repo execution defaults, while the skill should
stay focused on reusable Gluon writing knowledge.
