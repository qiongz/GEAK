# Triton-Gluon Examples

Read at most one relevant example after reading `00_always_read.md` and the
needed component traits. These examples are schematic; they are not golden
outputs.

## Internal Index

- `plain_triton_to_amd_gluon_minimal`
- `nv_gluon_to_amd_gluon_translation`
- `amd_gluon_in_dialect`
- `jit_aot_fallback`
- `triton_version_mfma_guard`
- `shared_transplant_example`
- `how_to_use_examples`

## how_to_use_examples

Examples verify shape of reasoning, not benchmark truth:

- `plain_triton_to_amd_gluon_minimal` proves host-created layout and launcher
  alignment are part of the rewrite.
- `nv_gluon_to_amd_gluon_translation` proves translation preserves common tile
  logic and drops unsupported NVIDIA fast paths before tuning.
- `triton_version_mfma_guard` proves Triton minor-version guards can be real
  module-level code, not placeholder comments.
- `jit_aot_fallback` proves import availability and package fallback can be
  part of the integration contract.
- `shared_transplant_example` proves a final plain Triton patch can still be
  Gluon-informed if it transplants evidence from Shared/Extension work.

Do not copy example constants blindly. Re-derive layout from the current
launcher, target arch, and benchmark shapes.

## plain_triton_to_amd_gluon_minimal

Use when `input_dialect = plain_triton` and `required_output_dialect =
amd_gluon`.

Suppose the source kernel applies
`output[row, col] = input[row, col] * scale[row] + bias[row]`. The plain Triton
version may leave layout implicit. The AMD-facing Gluon version should make the
one-dimensional row tile explicit before lowering memory ops. This example
includes host-side layout construction because first rewrites often fail there.

```python
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

def make_cdna_1d_layout(block_size: int, num_warps: int):
    lanes_per_cta = 64 * num_warps
    assert block_size % lanes_per_cta == 0
    return ttgl.BlockedLayout(
        size_per_thread=[block_size // lanes_per_cta],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )

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

    x = ttgl.amd.cdna3.buffer_load(
        ptr=x_ptr + row * n_cols,
        offsets=cols,
        mask=mask,
        other=0.0,
    )
    scale = ttgl.load(scale_ptr + row)
    bias = ttgl.load(bias_ptr + row)
    y = x * scale + bias
    ttgl.amd.cdna3.buffer_store(
        ptr=out_ptr + row * n_cols,
        offsets=cols,
        stored_value=y,
        mask=mask,
    )

def launch_row_affine(x, scale, bias, out, col_block=256, num_warps=4):
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    layout = make_cdna_1d_layout(col_block, num_warps)
    grid = (n_rows,)
    row_affine_kernel[grid](
        x,
        scale,
        bias,
        out,
        n_cols,
        COL_BLOCK=col_block,
        layout=layout,
        num_warps=num_warps,
    )
```

This shows:

- launcher stays Triton-like;
- host code may construct and pass layout;
- layout must agree with `col_block` and `num_warps`;
- main row tile can use AMD buffer ops while scalar row parameters remain
  generic;
- non-contiguous tensors need explicit strides instead of `row * n_cols`.

## nv_gluon_to_amd_gluon_translation

Use when `input_dialect = nv_gluon`.

Rules:

- preserve common Gluon control flow first;
- re-evaluate wave32 layouts;
- do not rename TMA / WGMMA / tensor-memory APIs into guessed AMD APIs;
- translate to AMD-facing semantics before tuning.

Example:

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
    row_offsets = pid * ROW_BLOCK + ttgl.arange(
        0,
        ROW_BLOCK,
        ttgl.SliceLayout(1, layout),
    )
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

This keeps tile logic recognizable, switches to AMD-valid layout and memory
path, and deliberately drops unsupported NVIDIA async/descriptor fast paths
until AMD parity is justified.

## amd_gluon_in_dialect

Use when source is already `amd_gluon`.

Rules:

- preserve existing AMD-facing structure first;
- optimize in dialect before drifting back to plain Triton;
- read operator-local architecture guards;
- keep JIT/AOT fallback behavior.

## jit_aot_fallback

Use when codebase mixes JIT, AOT, or prebuilt Gluon kernels.

Pattern:

```python
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    HAS_GLUON = True
except ImportError:
    HAS_GLUON = False
```

For required AMD Gluon tasks, an import failure must come from this supported
path. Do not probe `from triton import gluon`.

When production code intentionally has a prebuilt fallback, preserve that
decision surface:

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

## triton_version_mfma_guard

Use when `AMDMFMALayout.instr_shape` may differ across Triton minor versions.

```python
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

def parse_triton_version(version: str) -> tuple[int, ...]:
    version = version.split("+")[0].split("-")[0]
    parts = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)

TRITON_VERSION_GE_3_6_0 = tl.constexpr(
    parse_triton_version(triton.__version__) >= (3, 6, 0)
)

@gluon.jit
def mfma_guarded_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    K_BLOCK: gl.constexpr,
    CDNA_VERSION: gl.constexpr,
):
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

This is a real module-level compatibility pattern, not placeholder pseudocode.

## shared_transplant_example

Use when a Gluon/Shared task discovered a portable component but the safe anchor
is still plain Triton.

Prompt shape:

```text
Composition type: shared_transplant
Safe anchor: round_1/base-best/patch_N
Source component: memory_access_policy from round_1/ext-l1-gluon-buffer/patch_M
Comparison target: safe_anchor
Allowed change: transplant one portable memory/load component
Reject if: any shape regresses against the safe anchor or the safe-anchor algorithm changes

Preserve the safe-anchor algorithm. The output may remain plain Triton.
```

The output may remain `plain_triton`. This is Gluon-informed evidence, not a
Gluon-positive result.
