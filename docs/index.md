# GEAK documentation

GEAK is an AI-driven GPU kernel optimization framework. The main user-facing overview is **`README.md` at the repository root** (not duplicated here).

## In this folder

- **[Quick start](quick_start.md)** — install, model setup, and first `geak` runs.
- **[Configuration files](configuration.md)** — YAML merge order, **`--config`** resolution, **`rag_config.yaml`**.
- **[Contribution guidelines](developer/contribution_guidelines.md)** — branches, PR workflow, CI, code standards, and source-file requirements.
- **[Developer guide](developer/index.md)** — extend prompts, add MCP servers, native tools.
- **[ROCm environment reference](env_install.md)** — ROCm layout and library source paths useful for kernel work.
- **[Docker runtime notes](docker_env.md)** — reuse-first container guidance and known MI3xx baseline pitfalls around permissions, venvs, profiling, and GEAK execution.
- **[RAG filter sub-agent](subagent_guide.md)** — optional RAG filtering utilities in the codebase.
- **[Triton-Gluon guide](triton_gluon.md)** — single feature guide covering overall writing patterns, `nv_gluon` / `amd_gluon` differences, architecture notes, repo-local guidance, and appendix API usage.

To preview the same Markdown as a static site on your machine (optional):

```bash
pip install mkdocs-material
mkdocs serve
```

There is no hosted documentation site; all content lives in this repository as Markdown.
