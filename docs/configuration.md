# Configuration files

How GEAK loads **YAML** for **`geak`**, where builtin files live, and how **`--config`** resolves paths. For **model / CLI env** quick reference, use **[Quick start](quick_start.md)** §2.

## Main CLI

1. **Base template** — **`src/minisweagent/config/mini_kernel_strategy_list.yaml`** is loaded first.
2. **User config** — GEAK then loads either:
   - the default **`src/minisweagent/config/geak.yaml`**, or
   - the file passed via **`-c` / `--config`**
3. **Task-provided config** — if the natural-language task mentions a config path and you did **not** already pass `--config`, that file is merged next.
4. **CLI overrides** — command-line options still win last.

This matches the merge order in **`src/minisweagent/run/mini.py`**.

## Triton-family feature metadata

Gluon does **not** create a new top-level `kernel_type`. The product model stays:

- `hip`
- `triton`
- `other`

Within the Triton path, GEAK now treats AMD Gluon as a default candidate
strategy. The explicit **`gluon_feature_mode`** field remains for ablation and
forced-path debugging.

Primary user-facing fields:

| Field | Meaning |
|------|---------|
| **`input_dialect`** | Optional override; normally inferred as **`plain_triton`**, **`nv_gluon`**, or **`amd_gluon`** |
| **`gluon_feature_mode`** | Optional Gluon control: **`auto`** by default for Triton, **`off`** for no-Gluon ablation, **`force`** for AMD-Gluon-only debugging |
| **`gluon_baseline_profile`** | Baseline profile such as **`raw`** or **`mi3xx`** |
| **`target_backend`** | Target backend string such as **`hip/gfx942`** |

Derived planner fields:

| Field | Meaning |
|------|---------|
| **`allowed_output_dialects`** | Search space GEAK is allowed to consider |
| **`preferred_output_dialects`** | Ordered preference within that search space |
| **`output_dialect_search_policy`** | Planner policy such as “prefer AMD Gluon first” |

Current product policy:

- keep **`kernel_type = triton`**
- by default, allow both **`plain_triton`** and **`amd_gluon`** outputs and
  prefer an **`amd_gluon`** candidate first when structurally viable
- for **`nv_gluon`** inputs, let the agent translate vendor-specific APIs or
  assumptions before optimizing on AMD
- use **`gluon_feature_mode=off`** only for ablation against the no-Gluon path


## What’s in the default config file (**`src/minisweagent/config/geak.yaml`**)


### **`model:`**

| Key | Purpose |
|-----|---------|
| **`model_class`** | Backend short name for **`get_model_class`** (here **`amd_llm`** — AMD LLM gateway). |
| **`model_name`** | Gateway model id (e.g. **`claude-opus-4.6`**, **`claude-sonnet-4.5`**, **`gpt-5`**, **`gpt-5.1`**, **`gpt-5-codex`**). Routed inside **`AmdLlmModel`** to Claude / OpenAI / Gemini clients by name pattern. |
| **`api_key`** | Empty string **`""`** means “read **`AMD_LLM_API_KEY`** or **`LLM_GATEWAY_KEY`** from the environment”; a non-empty value is sent to the gateway instead. |
| **`model_kwargs`** | Passed through to the vendor implementation: **`temperature`**, **`max_tokens`**, plus gateway-specific blocks. **`reasoning.effort`** and **`text.verbosity`** apply to **GPT**-style models on the gateway (see inline comments in the YAML). |

### **`agent:`**

| Key | Purpose |
|-----|---------|
| **`step_limit`** | Step cap for **`DefaultAgent`**. **`0`** means **disabled** (limits apply only when **`0 < step_limit`**). |
| **`cost_limit`** | Cost cap (same class). **`0`** means **disabled** (limits apply only when **`0 < cost_limit`**). |
| **`mode`** | **`confirm`** = interactive confirmation for tool actions; **`yolo`** = auto-run. Parallel workers force **`yolo`** regardless. |


### **`env:`**

| Key | Purpose |
|-----|---------|
| **`env`** | Nested map of **process environment** variables forwarded to the tool runtime / subprocesses (e.g. **`PAGER`**, **`MANPAGER`**, **`LESS`**, **`PIP_PROGRESS_BAR`**, **`TQDM_DISABLE`**) so logs stay non-interactive in automation. |
| **`timeout`** | Default **command timeout** in seconds (here **`3600`**) for environment executions where applicable. |

*(The large **`system_template`** / **`instance_template`** blocks live in **`mini_kernel_strategy_list.yaml`** unless you override them in another **`--config`** file.)*
