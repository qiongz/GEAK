# Docker Runtime Notes

This document records the container and environment pitfalls observed during the
MI3xx/gfx942 Triton-Gluon baseline work.

It is intentionally narrower than `docs/env_install.md`:

- `docs/env_install.md` is a general ROCm library reference
- this document is about the known-good Docker runtime, reuse policy, and the
  failure modes we already hit in practice

## Scope

These notes come from the validated MI3xx baseline path:

- GEAK root: `/apps/qiongzhu/GEAK`
- Triton root: `/apps/qiongzhu/triton`
- baseline container: `feature-triton-gluon-mi3xx-baseline`
- baseline venv: `/apps/qiongzhu/.venvs/triton-gluon-mi3xx`
- target GPU: `gfx942`

The main lesson is simple: once a path is green, prefer reuse over re-bootstrap.

## Known-good baseline

The most stable runtime posture was:

- reuse one fixed Docker container for the whole baseline cycle
- reuse one fixed venv inside a mounted workspace path
- keep caches under `/apps/qiongzhu`, not under `/root` or an unmounted home path
- run container commands as the matching host UID/GID instead of root
- keep `PYTHONPATH` layered so GEAK and the Triton checkout come first, while
  Torch and container site-packages remain visible

Useful mounted paths:

- `TRITON_HOME=/apps/qiongzhu/.triton`
- `CCACHE_DIR=/apps/qiongzhu/.ccache`
- `PIP_CACHE_DIR=/apps/qiongzhu/.pip-cache`

## Reuse-first policy

If the current container and venv already satisfy smoke, correctness, benchmark,
and profiling, do not:

- create a new container variant
- rerun bootstrap from scratch
- switch compiler routes
- move cache directories
- rebuild the venv in a different location

Only consider reinstall or rebuild when at least one of these is true:

- the container no longer starts
- the venv is missing required modules
- GPU visibility changed
- the Triton import stack is broken
- profiling tools are missing and cannot be repaired in-place

## Reinstall decision guide

### Prefer in-place repair when

- a Python dependency is missing
- a cache path is wrong
- `PATH` or `PYTHONPATH` is incomplete
- Docker exec is running as the wrong user/group
- a profiler dependency can be added without replacing the whole runtime

### Rebuild only when

- the container image itself is corrupted
- the venv is unrecoverable
- the compiler toolchain is fundamentally incompatible
- a clean environment is required for reproducibility and the current runtime is
  not trusted anymore

## Common pitfalls

### 1. Mount and cache permissions

Symptom:

- Docker reported permission problems when trying to create or use
  `/home/qiongzhu/.triton` or `/home/qiongzhu/.ccache`

Cause:

- the chosen cache location was not writable from the runtime path being used

Fix:

- move runtime caches into mounted, user-owned workspace paths:
  - `/apps/qiongzhu/.triton`
  - `/apps/qiongzhu/.ccache`
  - `/apps/qiongzhu/.pip-cache`

### 2. Git dubious ownership inside the container

Symptom:

- `git rev-parse` or other git commands failed with "detected dubious ownership"

Cause:

- container commands were running as a user that did not match the workspace
  ownership

Fix:

- run the real workload as the host UID/GID
- for this baseline, `setpriv --reuid=100352 --regid=100352 --groups=100352,44,110`
  was the working pattern

### 3. `pip install --user` inside a venv

Symptom:

- installation failed with "Can not perform a '--user' install. User
  site-packages are not visible in this virtualenv."

Cause:

- `--user` was used inside an activated venv

Fix:

- create the venv in a mounted shared path
- install directly into that venv without `--user`

### 4. `cmake` not found during editable install

Symptom:

- editable Triton install failed with `FileNotFoundError: 'cmake'`

Cause:

- the venv's `bin` directory was not at the front of `PATH`

Fix:

- prepend `/apps/qiongzhu/.venvs/triton-gluon-mi3xx/bin` to `PATH` before
  bootstrap or editable install steps

### 5. ROCm clang + Ninja RPATH failure

Symptom:

- CMake failed with an RPATH error when using the Ninja generator

Cause:

- the default CMake/RPATH combination was not compatible with that toolchain
  path

Fix:

- set:
  `TRITON_APPEND_CMAKE_ARGS=-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`

This should be treated as a bounded workaround, not a new baseline to keep
re-exploring once the path is already green.

### 6. ROCm clang warnings promoted to errors

Symptom:

- Triton build failed on deprecated declarations under `-Werror`

Cause:

- the active ROCm clang route was too strict for that code path

Fix:

- install and use a compatible `clang` / `lld` path that can complete the build
- do not keep bouncing between compiler variants once one route works

### 7. `apt-get` mirror/network flakiness

Symptom:

- package installation failed with transient connection errors

Cause:

- mirror instability, not necessarily a package definition issue

Fix:

- retry with bounded retries/timeouts rather than immediately redesigning the
  environment

## Runtime pitfalls after bootstrap

### 8. GPU not visible from the container

Symptom:

- Triton failed with "0 active drivers ([])"

Cause:

- the runtime user was not in the render/video groups required for GPU access

Fix:

- run the workload with the correct supplementary groups
- in this baseline, the working group set included `44` and `110`

### 9. Wrong `PYTHONPATH` ordering

Symptom:

- imports alternated between:
  - missing `torch`
  - missing `triton.backends.amd`
  - missing AMD Gluon symbols

Cause:

- `PYTHONPATH` order hid either the source checkout or the container site-packages

Fix:

- put these first:
  - `/apps/qiongzhu/GEAK/src`
  - `/apps/qiongzhu/triton/python`
- then append the container Python site-packages paths that provide Torch and
  related packages

### 10. Editable-source Triton import is not enough by itself

Symptom:

- `triton.backends.amd` or `triton.language.extra.hip` could not be imported
  from a fresh container path even though the Triton source tree was present

Cause:

- those AMD pieces are registered dynamically in editable-install flows

Fix:

- do not assume "source tree visible" means "runtime complete"
- for local task contracts, use the runtime helper layer that explicitly checks
  AMD Gluon availability and registers the required source-tree packages

### 11. Missing `metrix`

Symptom:

- full preprocess profiling failed with `No module named 'metrix'`

Cause:

- `metrix` was not present in the baseline venv

Fix:

- install the pinned Git dependency rather than searching for a PyPI release:

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git@bcbfa0252df9d55f3aab68c95dd3ce45ccbe5b46#subdirectory=metrix"
```

### 12. Not every benchmark-valid task is profiler-ready

Symptom:

- a task could benchmark but still failed to generate a meaningful profile

Cause:

- the harness `--profile` path did not actually launch a GPU kernel

Observed example:

- `test_buffer_load_store` remained compile-latency-only and not profiler-ready

Fix:

- validate the profiler stack first on the fastest profiler-ready canary
- for this baseline, `inline_with_amdgpu_dialect` was the right smoke target

## GEAK-specific Docker pitfalls

### 13. `eval-command` generated a brittle `COMMANDMENT.md`

Symptom:

- `BENCHMARK` and `FULL_BENCHMARK` sections could end up with broken nested
  shell quoting

Cause:

- the `--correctness-command` / `--performance-command` path was less robust
  than a real four-mode harness

Fix:

- when possible, provide a harness script that supports:
  - `--correctness`
  - `--profile`
  - `--benchmark`
  - `--full-benchmark`

### 14. GEAK sub-agent editor permission issue

Symptom:

- a generated sub-agent hit `PermissionError: /root/.swe-agent-env` when using
  the editor tool

Cause:

- the editor helper expected root-scoped agent state that was not accessible in
  the current Docker execution pattern

Fix:

- the agent could still fall back to shell-based editing and complete the run
- but this should be treated as a real environment compatibility issue if we
  want longer or more tool-heavy optimization runs

## Recommended command posture

When invoking GEAK inside the validated baseline container:

- export `HOME=/apps/qiongzhu`
- export both `LLM_GATEWAY_KEY` and `AMD_LLM_API_KEY` when model access is needed
- prepend the baseline venv `bin` directory to `PATH`
- prepend `GEAK/src` and `triton/python` to `PYTHONPATH`
- use `bash --noprofile --norc -lc`
- run the real Python workload under the matching UID/GID and GPU-visible groups

## What not to do after the runtime is green

- do not start a second "just to compare" container unless there is a concrete blocker
- do not rerun old bootstrap or apt-install routes out of habit
- do not move the venv from a mounted path back into an ephemeral location
- do not switch from reuse to reinstall without naming the exact blocker
- do not validate profiler readiness on a compile-only harness

## Relationship to the other documents

- For the current MI3xx baseline workflow, see `docs/triton_gluon_mi3xx_baseline.md`
- For generic ROCm library/source layout, see `docs/env_install.md`
- For the current Triton-Gluon Layer 1 boundary, see `docs/triton_gluon_layer1_scope.md`
