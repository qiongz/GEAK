# GEAK documentation

GEAK is an AI-driven GPU kernel optimization framework. The main user-facing overview is **`README.md` at the repository root** (not duplicated here).

## In this folder

- **[Quick start](quick_start.md)** — install, model setup, and first `geak` runs.
- **[Configuration files](configuration.md)** — YAML merge order, **`--config`** resolution, **`rag_config.yaml`**.
- **[Development guidelines](development_guidelines.md)** — branches, PR workflow, CI, coding standards.
- **[Developer guide](developer/index.md)** — extend prompts, add MCP servers, native tools.
- **[ROCm environment reference](env_install.md)** — ROCm layout and library source paths useful for kernel work.
- **[Docker runtime notes](docker_env.md)** — reuse-first container guidance and known MI3xx baseline pitfalls around permissions, venvs, profiling, and GEAK execution.
- **[RAG filter sub-agent](subagent_guide.md)** — optional RAG filtering utilities in the codebase.
- **[Triton-Gluon MI3xx baseline](triton_gluon_mi3xx_baseline.md)** — GEAK-side integration points, canary order, and run assets for the `triton-gluon` baseline workflow.
- **[Triton-Gluon Layer 1 scope](triton_gluon_layer1_scope.md)** — what Goal/Layer 1 has already proven for existing Triton-Gluon inputs.
- **[Triton-Gluon Layer 2 scope](triton_gluon_layer2_scope.md)** — the bounded `nv-gluon -> amd-gluon` translation layer and its success criteria.
- **[Triton-Gluon translation rules](triton_gluon_translation_rules.md)** — NV→AMD translation playbook, non-parity classes, and first-sample constraints.
- **[Triton-Gluon API quick reference](triton_gluon_api_quick_reference.md)** — compact syntax and architecture cheat sheet for Gluon writing.
- **[Triton-Gluon writing guide](triton_gluon_writing_guide.md)** — reusable Gluon syntax patterns, architecture mapping, and AMD/NVIDIA differences.

To preview the same Markdown as a static site on your machine (optional):

```bash
pip install mkdocs-material
mkdocs serve
```

There is no hosted documentation site; all content lives in this repository as Markdown.
