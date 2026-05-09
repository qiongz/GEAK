"""Dispatch helpers: run task files via ParallelAgent pool mode.

This module provides ``run_task_batch()`` which converts a list of task
file paths into ``AgentTask`` objects and feeds them into the existing
``ParallelAgent.run_parallel(tasks=...)`` pool mode.  The orchestrator
calls this; so does the ``run-tasks`` CLI indirectly.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from minisweagent import get_repo_root
from minisweagent.debug_runtime import emit_debug_log
from minisweagent.run.preprocess.discovery_types import (
    allowed_skill_tiers_for_feature,
    build_gluon_feature_metadata,
)
from minisweagent.run.gluon_doc_profiles import (
    MANDATORY_GLUON_DOC_KEYS,
    add_unique_doc_key,
    required_doc_keys_for_profile,
    task_requires_gluon_worker_docs,
)
from minisweagent.run.target_contracts import target_symbols_from_scoped_text

_GEAK_REPO_ROOT = get_repo_root()
_GLUON_GATE_FALLBACK_RELS = {
    "gluon_skill_path": "skills/triton-gluon/SKILL.md",
    "gluon_always_read_path": "skills/triton-gluon/docs/00_always_read.md",
    "gluon_search_policies_path": "skills/triton-gluon/docs/10_search_policies.md",
    "gluon_component_traits_path": "skills/triton-gluon/docs/20_component_traits.md",
    "gluon_architecture_notes_path": "skills/triton-gluon/docs/30_architecture_notes.md",
    "gluon_examples_doc_path": "skills/triton-gluon/docs/40_examples.md",
    "gluon_api_reference_path": "skills/triton-gluon/docs/50_api_reference.md",
    "gluon_real_patterns_path": "skills/triton-gluon/docs/60_real_patterns.md",
    "gluon_backup_details_path": "skills/triton-gluon/docs/70_backup_details.md",
}

# ── model ensemble support ───────────────────────────────────────────


def _build_ensemble_factory(base_factory):
    """Wrap *base_factory* to rotate through models in GEAK_MODEL_ENSEMBLE.

    When ``GEAK_MODEL_ENSEMBLE`` is set (comma-separated model names), each
    call to the returned factory creates an entirely new model instance with
    the next name in the round-robin list.  Because ``AmdLlmModel`` selects
    its vendor backend (OpenAI / Claude / Gemini) at construction time based
    on the model name, we must create a fresh instance per call rather than
    mutating an existing one.

    If the env var is unset or empty, returns *base_factory* unchanged.
    """
    ensemble_str = os.environ.get("GEAK_MODEL_ENSEMBLE", "").strip()
    if not ensemble_str:
        return base_factory

    model_names = [n.strip() for n in ensemble_str.split(",") if n.strip()]
    if len(model_names) < 2:
        return base_factory

    logger.info("Model ensemble enabled: %s", model_names)
    name_cycle = itertools.cycle(model_names)

    def _ensemble_factory():
        next_name = next(name_cycle)
        try:
            from minisweagent.models.amd_llm import AmdLlmModel

            model = AmdLlmModel(model_name=next_name)
            logger.info("Ensemble: created AmdLlmModel(%s)", next_name)
            return model
        except Exception:
            logger.warning(
                "Ensemble: failed to create model %s, falling back to base",
                next_name,
                exc_info=True,
            )
            return base_factory()

    return _ensemble_factory


def _read_commandment_section(commandment_path: str, section: str) -> str | None:
    """Read a section from a COMMANDMENT.md file verbatim.

    Returns the raw command lines for the given section (e.g. ``"SETUP"``,
    ``"CORRECTNESS"``, ``"PROFILE"``, ``"BENCHMARK"``,
    ``"FULL_BENCHMARK"``), exactly as written.  No parsing, no extraction,
    no transformation.

    Fenced code blocks (```bash, ```, etc.) are stripped automatically.
    """
    try:
        text = Path(commandment_path).read_text()
    except OSError:
        logger.debug("Could not read commandment at %s", commandment_path)
        return None

    lines: list[str] = []
    in_section = False
    # Pattern to match fenced code block markers (```bash, ```sh, ```, etc.)
    fence_pattern = re.compile(r"^```\w*$")

    for raw_line in text.splitlines():
        header = re.match(r"^##\s+(\w+)", raw_line.strip())
        if header:
            if header.group(1) == section:
                in_section = True
                continue
            elif in_section:
                break
            continue
        if in_section:
            stripped = raw_line.strip()
            # Skip fenced code block markers
            if fence_pattern.match(stripped):
                continue
            if stripped:
                lines.append(stripped)

    return "\n".join(lines) if lines else None


def _commandment_test_command(commandment_path: str) -> str | None:
    """Build a test command that executes SETUP then CORRECTNESS then BENCHMARK.

    The COMMANDMENT is the single source of truth.  Commands are executed
    *as-is* -- no parsing, no unwrapping, no modification.  The runtime
    must set ``GEAK_WORK_DIR`` and ``GEAK_GPU_DEVICE`` so that variable
    references in the commands resolve correctly.

    We write a temporary shell script instead of ``bash -c '...'`` because
    COMMANDMENT sections often contain single-quoted strings (e.g.
    ``printf '...'``) that cannot be nested inside a single-quoted
    ``bash -c`` wrapper.

    Note: This script is an internal implementation detail for agent tools
    (save_and_test).  OpenEvolve reads COMMANDMENT.md directly and does
    not use this script.
    """
    setup = _read_commandment_section(commandment_path, "SETUP")
    correctness = _read_commandment_section(commandment_path, "CORRECTNESS")
    benchmark = _read_commandment_section(commandment_path, "BENCHMARK")
    if not benchmark:
        benchmark = _read_commandment_section(commandment_path, "FULL_BENCHMARK")

    if not correctness:
        return None

    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if setup:
        lines.append(setup)
    lines.append(correctness)
    if benchmark:
        lines.append(benchmark)
    script_body = "\n".join(lines) + "\n"

    cmd_dir = Path(commandment_path).parent
    # Use unique filename to avoid race conditions when multiple agents
    # run concurrently with the same COMMANDMENT directory
    import tempfile

    fd, script_path = tempfile.mkstemp(
        prefix="_geak_test_cmd_",
        suffix=".sh",
        dir=str(cmd_dir),
    )
    os.close(fd)
    script_path = Path(script_path)
    script_path.write_text(script_body)
    script_path.chmod(0o755)

    return str(script_path)


def _task_body_field(task_body: str, field: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(task_body or "")
    return match.group(1).strip() if match else ""


def _task_required_patch_target_symbols(meta: dict[str, Any], task_body: str = "") -> list[str]:
    raw = meta.get("required_patch_target_symbols")
    if isinstance(raw, str):
        symbols = [part.strip().strip("`") for part in raw.split(",")]
    elif isinstance(raw, list):
        symbols = [str(part).strip().strip("`") for part in raw]
    else:
        symbols = []

    for match in re.finditer(
        r"^\s*(?:Target symbol|Target component|Required patch target symbols?)\s*:\s*(.+?)\s*$",
        task_body or "",
        re.IGNORECASE | re.MULTILINE,
    ):
        value = match.group(1)
        symbols.extend(part.strip().strip("`") for part in value.split(","))
        symbols.extend(target_symbols_from_scoped_text(value))
    for field in ("Allowed change", "Target component"):
        tagged = _task_body_field(task_body, field)
        if tagged:
            symbols.extend(target_symbols_from_scoped_text(tagged))

    unique: list[str] = []
    for symbol in symbols:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol) and symbol not in unique:
            unique.append(symbol)
    return unique


def _task_required_output_dialect(meta: dict[str, Any], task_body: str = "") -> str:
    label_text = str(meta.get("label") or "").strip().lower()
    required = str(meta.get("required_output_dialect") or "").strip().lower()
    implementation_layer = str(meta.get("implementation_layer") or _task_body_field(task_body, "Implementation layer")).strip().lower()
    extension_layer = str(meta.get("extension_layer") or _task_body_field(task_body, "Extension layer")).strip().lower()

    layer_required = ""
    if "hybrid" in label_text or "mixed" in label_text:
        layer_required = "mixed"
    elif "mixed" in implementation_layer or "hybrid" in implementation_layer or extension_layer == "hybrid":
        layer_required = "mixed"
    elif "amd_gluon" in implementation_layer or extension_layer in {"l0", "l1"}:
        layer_required = "amd_gluon"
    elif (
        label_text.startswith(("ext-", "extension-"))
        or "extension-l" in label_text
        or "ext_l" in label_text
    ) and ("gluon" in label_text or "amd-gluon" in label_text or "amd_gluon" in label_text):
        layer_required = "amd_gluon"

    if required:
        if required in {"any", "plain_triton"} and layer_required in {"amd_gluon", "mixed"}:
            return layer_required
        return required
    return layer_required or "any"


def _task_uses_skills(meta: dict[str, Any], task_body: str = "") -> bool:
    """Return whether a task should enable the skill runtime."""
    if _task_requires_amd_gluon(meta, task_body):
        return True
    if "use_skills" in meta:
        raw = meta.get("use_skills")
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    return str(meta.get("kernel_type", "")).strip().lower() == "triton"


def _task_requires_amd_gluon(meta: dict[str, Any], task_body: str = "") -> bool:
    """Return whether task metadata requires an actual AMD Gluon output."""
    return (
        str(meta.get("kernel_type") or "").strip().lower() == "triton"
        and _task_required_output_dialect(meta, task_body) == "amd_gluon"
    )


def _task_requires_gluon_worker_docs(meta: dict[str, Any], task_body: str = "") -> bool:
    """Return whether this task should receive the Gluon worker documentation gate."""
    return task_requires_gluon_worker_docs(
        kernel_type=str(meta.get("kernel_type") or ""),
        required_output=_task_required_output_dialect(meta, task_body),
        implementation_layer=meta.get("implementation_layer") or _task_body_field(task_body, "Implementation layer"),
        extension_layer=meta.get("extension_layer") or _task_body_field(task_body, "Extension layer"),
        doc_profile=meta.get("gluon_doc_profile"),
        task_body=task_body,
        label=str(meta.get("label") or ""),
    )


def _task_required_amd_gluon_contract_tags(meta: dict[str, Any], task_body: str = "") -> list[str]:
    text = "\n".join(
        str(part or "")
        for part in (
            task_body,
            meta.get("label"),
            meta.get("gluon_doc_profile"),
            meta.get("optimization_direction"),
            meta.get("performance_hypothesis"),
            meta.get("allowed_change"),
            meta.get("gluon_overlay_reason"),
        )
    ).lower()
    tags: list[str] = []
    if any(
        marker in text
        for marker in (
            "matrix",
            "dotoperandlayout",
            "operand layout",
            "mfma",
            "wmma",
            "matrix_lowering",
        )
    ):
        tags.append("matrix_lowering")
    return tags


def _task_feature_metadata(meta: dict[str, Any], task_body: str = "") -> dict[str, Any]:
    """Rebuild the normalized Gluon feature metadata from task frontmatter."""
    kernel_path = Path(str(meta.get("kernel_path") or "unknown.py"))
    required_output_dialect = _task_required_output_dialect(meta, task_body)
    feature_meta = build_gluon_feature_metadata(
        kernel_path,
        str(meta.get("kernel_type") or "unknown"),
        input_dialect=meta.get("input_dialect"),
        gluon_feature_mode=meta.get("gluon_feature_mode"),
        gluon_baseline_profile=meta.get("gluon_baseline_profile"),
        allowed_output_dialects=meta.get("allowed_output_dialects"),
        preferred_output_dialects=meta.get("preferred_output_dialects"),
        output_dialect_search_policy=meta.get("output_dialect_search_policy"),
        target_backend=meta.get("target_backend"),
        benchmark_shape_count=meta.get("benchmark_shape_count"),
        benchmark_test_cases=meta.get("benchmark_test_cases"),
        shape_coverage_profile=meta.get("shape_coverage_profile"),
    )
    if meta.get("search_set"):
        feature_meta["search_set"] = str(meta.get("search_set"))
    if meta.get("label"):
        feature_meta["label"] = str(meta.get("label"))
    if required_output_dialect:
        feature_meta["required_output_dialect"] = required_output_dialect
    if meta.get("gluon_doc_profile"):
        feature_meta["gluon_doc_profile"] = str(meta.get("gluon_doc_profile"))
    if meta.get("required_gluon_docs"):
        feature_meta["required_gluon_docs"] = meta.get("required_gluon_docs")
    if meta.get("source_base_family"):
        feature_meta["source_base_family"] = str(meta.get("source_base_family"))
    if meta.get("plain_competitor"):
        feature_meta["plain_competitor"] = str(meta.get("plain_competitor"))
    for key in (
        "extension_intent",
        "expected_outcome",
        "not_viable_for_l1_if_slower_than_base",
        "overhead_source_to_record",
        "target_symbol",
        "target_component",
        "forbidden_change",
    ):
        if key in meta and meta.get(key) not in (None, ""):
            feature_meta[key] = meta.get(key)
    implementation_layer = meta.get("implementation_layer") or _task_body_field(task_body, "Implementation layer")
    extension_layer = meta.get("extension_layer") or _task_body_field(task_body, "Extension layer")
    if implementation_layer:
        feature_meta["implementation_layer"] = str(implementation_layer)
    if extension_layer:
        feature_meta["extension_layer"] = str(extension_layer)
    required_patch_target_symbols = _task_required_patch_target_symbols(meta, task_body)
    if required_patch_target_symbols:
        feature_meta["required_patch_target_symbols"] = required_patch_target_symbols
    return feature_meta


def _task_allowed_skill_tiers(meta: dict[str, Any]) -> list[str]:
    """Return the skill tiers visible to this dispatched task."""
    explicit = meta.get("allowed_skill_tiers")
    if explicit:
        if isinstance(explicit, str):
            return [explicit]
        return [str(item) for item in explicit]
    feature_meta = _task_feature_metadata(meta)
    return allowed_skill_tiers_for_feature(
        str(meta.get("kernel_type") or "unknown"),
        gluon_feature_mode=feature_meta["gluon_feature_mode"],
        gluon_baseline_profile=feature_meta["gluon_baseline_profile"],
    )


def _normalize_gate_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except (OSError, RuntimeError):
        return text


def _add_gate_path(paths: list[str], meta: dict[str, Any], key: str) -> None:
    value = meta.get(key)
    if not value and key in _GLUON_GATE_FALLBACK_RELS:
        candidate = _GEAK_REPO_ROOT / _GLUON_GATE_FALLBACK_RELS[key]
        if candidate.exists():
            value = str(candidate)
    path = _normalize_gate_path(value)
    if path and path not in paths:
        paths.append(path)


def _required_gluon_docs_from_metadata(meta: dict[str, Any]) -> list[str]:
    raw_docs = meta.get("required_gluon_docs")
    if isinstance(raw_docs, str):
        doc_keys = [part.strip() for part in raw_docs.split(",") if part.strip()]
    elif isinstance(raw_docs, list):
        doc_keys = [str(part).strip() for part in raw_docs if str(part).strip()]
    else:
        return []

    required: list[str] = []
    for key_or_path in doc_keys:
        if key_or_path in _GLUON_GATE_FALLBACK_RELS or key_or_path in meta:
            _add_gate_path(required, meta, key_or_path)
        else:
            path = _normalize_gate_path(key_or_path)
            if path and path not in required:
                required.append(path)
    return required


def _gluon_doc_gate_required_paths(meta: dict[str, Any], task_body: str) -> list[str]:
    """Return split-doc paths that a Gluon worker must view before save_and_test."""
    if not _task_requires_gluon_worker_docs(meta, task_body):
        return []

    required: list[str] = []

    def add_doc_key(key: str) -> None:
        before = list(required)
        _add_gate_path(required, meta, key)
        if required == before and key not in _GLUON_GATE_FALLBACK_RELS:
            add_unique_doc_key(required, key)

    for key in MANDATORY_GLUON_DOC_KEYS:
        add_doc_key(key)
    for key in required_doc_keys_for_profile(meta.get("gluon_doc_profile")):
        add_doc_key(key)
    for path in _required_gluon_docs_from_metadata(meta):
        if path and path not in required:
            required.append(path)

    text = "\n".join(
        str(part or "")
        for part in (
            task_body,
            meta.get("label"),
            meta.get("search_set"),
            meta.get("required_output_dialect"),
            meta.get("input_dialect"),
            meta.get("target_backend"),
            meta.get("target_arch"),
            meta.get("gluon_doc_profile"),
            meta.get("implementation_layer"),
            meta.get("extension_layer"),
            meta.get("source_base_family"),
            meta.get("plain_competitor"),
        )
    ).lower()

    # Gluon implementation workers need the trait file for the planner-emitted
    # route, even when the prompt does not name a specific trait explicitly.
    if any(
        marker in text
        for marker in (
            "gluon",
            "amd_gluon",
            "nv_gluon",
            "shared set",
            "layout",
            "blockedlayout",
            "slicelayout",
            "dotoperandlayout",
            "convert_layout",
            "amdmfmalayout",
            "amdwmmalayout",
            "threads_per_warp",
            "warps_per_cta",
        )
    ):
        _add_gate_path(required, meta, "gluon_component_traits_path")

    if any(
        marker in text
        for marker in (
            "gfx",
            "gfx942",
            "gfx950",
            "gfx1250",
            "cdna",
            "cdna3",
            "cdna4",
            "rdna",
            "mfma",
            "wmma",
            "jit",
            "aot",
            "prebuilt",
            "compile_gluon",
            "compilegluon",
            "signature",
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "kpack",
            "sanitize_overflow",
            "global_scratch",
            "profile_scratch",
            "instr_shape",
            "k_width",
            "amdmfmalayout",
            "amdwmmalayout",
            "mfma_scaled",
            "wmma_scaled",
            "wave64",
            "wave32",
            "descriptor",
            "tensordescriptor",
            "tdm",
            "current_target",
            "translator",
            "triton_to_gluon",
            "target_backend",
        )
    ):
        _add_gate_path(required, meta, "gluon_architecture_notes_path")

    if _task_required_output_dialect(meta, task_body) == "amd_gluon" or any(
        marker in text
        for marker in (
            "api",
            "syntax",
            "gl.",
            "ttgl",
            "@gluon.jit",
            "buffer_load",
            "buffer_store",
            "buffer_atomic",
            "atomic_",
            "convert_layout",
            "dotoperandlayout",
            "amdmfmalayout",
            "amdwmmalayout",
            "mfma_scaled",
            "wmma_scaled",
            "get_mfma_scale_layout",
            "get_wmma_scale_layout",
            "k_width",
            "gl.full",
            "gl.full_like",
            "gl.reduce",
            "gl.sum",
            "gl.max",
            "gl.min",
            "gl.softmax",
            "gl.scan",
            "gl.associative_scan",
            "gl.histogram",
            "gl.reshape",
            "gl.permute",
            "gl.split",
            "gl.join",
        )
    ):
        _add_gate_path(required, meta, "gluon_api_reference_path")

    if any(
        marker in text
        for marker in (
            "aiter",
            "attention",
            "decode",
            "gemm",
            "fp8",
            "fp4",
            "mqa",
            "blockscale",
            "afp4",
            "wfp4",
            "preshuffle",
            "per-shape",
            "shape regression",
            "source-first",
            "translator",
            "current_target",
            "tensordescriptor",
            "tdm",
        )
    ):
        _add_gate_path(required, meta, "gluon_real_patterns_path")

    return required


def task_file_to_agent_task(task_file: Path):
    """Read a task markdown file and convert it to an AgentTask.

    This is the canonical task-construction path used by both the
    orchestrator (``run_task_batch``) and the standalone ``run-tasks``
    CLI.  It:
      - Applies agent-type filtering (``filter_agent_type``)
      - Sets per-agent-type config (mode, strategy_manager, etc.)
      - Injects full pipeline context (COMMANDMENT, baseline metrics,
        profiling data, codebase context) into the task body
    """
    from minisweagent.agents.agent_spec import AgentTask
    from minisweagent.run.task_file import read_task_file

    meta, body = read_task_file(task_file)

    from minisweagent.agents.agent_spec import _agent_type_to_class, filter_agent_type
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    agent_type = filter_agent_type(meta.get("agent_type", "strategy_agent"))
    agent_class = _agent_type_to_class().get(agent_type, StrategyInteractiveAgent)

    try:
        inherited_step_limit = int(os.environ.get("GEAK_AGENT_STEP_LIMIT", "200"))
    except ValueError:
        inherited_step_limit = 200
    task_step_limit = int(meta.get("step_limit", 0) or 0)
    effective_step_limit = task_step_limit or inherited_step_limit

    required_output_dialect = _task_required_output_dialect(meta, body)
    cfg: dict = {
        "save_patch": True,
        "step_limit": effective_step_limit,
        "cost_limit": 0.0,
        "mode": "yolo",
        "use_strategy_manager": True,
        "use_skills": _task_uses_skills(meta, body),
    }
    gluon_doc_gate_paths = _gluon_doc_gate_required_paths(meta, body)
    if gluon_doc_gate_paths:
        cfg["gluon_doc_gate_enabled"] = True
        cfg["gluon_doc_gate_required_paths"] = gluon_doc_gate_paths
    if required_output_dialect and required_output_dialect != "any":
        cfg["required_output_dialect"] = required_output_dialect
    required_amd_gluon_contract_tags = _task_required_amd_gluon_contract_tags(meta, body)
    if required_amd_gluon_contract_tags:
        cfg["required_amd_gluon_contract_tags"] = required_amd_gluon_contract_tags
    required_patch_target_symbols = _task_required_patch_target_symbols(meta, body)
    if required_patch_target_symbols:
        cfg["required_patch_target_symbols"] = required_patch_target_symbols

    # COMMANDMENT is the single source of truth for test commands.
    # Its SETUP + CORRECTNESS + BENCHMARK sections are executed verbatim.
    # BENCHMARK is the canonical latency path and intentionally mirrors
    # FULL_BENCHMARK in the generated COMMANDMENT.
    if meta.get("commandment") and Path(meta["commandment"]).exists():
        derived = _commandment_test_command(meta["commandment"])
        if derived:
            cfg["test_command"] = derived
            logger.info("test_command from COMMANDMENT (verbatim): %s", derived)
    if not cfg.get("test_command") and meta.get("test_command"):
        cfg["test_command"] = meta["test_command"]
        logger.warning(
            "No COMMANDMENT available; falling back to raw test_command: %s",
            meta["test_command"],
        )

    # Prepend pipeline context so the sub-agent has all necessary information.
    # IMPORTANT: Paths from metadata use the ORIGINAL repo root.  The parallel
    # agent's _replace_paths() rewrites them to the worktree path before the
    # agent sees the task text.  We must include these paths verbatim here.
    from minisweagent.run.pipeline_helpers import inject_pipeline_context

    commandment_text: str | None = None
    _cmd_path = meta.get("commandment")
    if _cmd_path and Path(_cmd_path).exists():
        commandment_text = Path(_cmd_path).read_text().strip()

    baseline_metrics: dict | None = None
    _bm_path = meta.get("baseline_metrics")
    if _bm_path and Path(_bm_path).exists():
        import json as _json

        baseline_metrics = _json.loads(Path(_bm_path).read_text())

    codebase_ctx_text: str | None = None
    _cb_path = meta.get("codebase_context")
    if _cb_path and Path(_cb_path).exists():
        codebase_ctx_text = Path(_cb_path).read_text().strip()

    benchmark_baseline_text: str | None = None
    _bb_path = meta.get("benchmark_baseline")
    if _bb_path and Path(_bb_path).exists():
        benchmark_baseline_text = Path(_bb_path).read_text().strip()

    feature_meta = _task_feature_metadata(meta, body)

    body, cfg = inject_pipeline_context(
        body,
        cfg,
        commandment_text=commandment_text,
        baseline_metrics=baseline_metrics,
        profiling_path=meta.get("profiling"),
        kernel_path=meta.get("kernel_path"),
        repo_root=meta.get("repo_root"),
        test_command=cfg.get("test_command"),
        codebase_context=codebase_ctx_text,
        benchmark_baseline=benchmark_baseline_text,
        feature_metadata=feature_meta,
        knowledge_base_path=meta.get("knowledge_base_path"),
        gluon_skill_path=meta.get("gluon_skill_path"),
        gluon_kb_path=meta.get("gluon_kb_path"),
        gluon_examples_path=meta.get("gluon_examples_path"),
        gluon_always_read_path=meta.get("gluon_always_read_path"),
        gluon_search_policies_path=meta.get("gluon_search_policies_path"),
        gluon_component_traits_path=meta.get("gluon_component_traits_path"),
        gluon_architecture_notes_path=meta.get("gluon_architecture_notes_path"),
        gluon_examples_doc_path=meta.get("gluon_examples_doc_path"),
        gluon_api_reference_path=meta.get("gluon_api_reference_path"),
        gluon_real_patterns_path=meta.get("gluon_real_patterns_path"),
        gluon_backup_details_path=meta.get("gluon_backup_details_path"),
        gluon_doc_gate_required_paths=gluon_doc_gate_paths,
    )

    try:
        from minisweagent.memory.integration import assemble_memory_context

        _bm = baseline_metrics or {}
        _mem_ctx = assemble_memory_context(
            kernel_path=meta.get("kernel_path", ""),
            bottleneck_type=_bm.get("bottleneck", ""),
            profiling_metrics=_bm,
        )
        if _mem_ctx and len(_mem_ctx) > 50:
            body += "\n\n## Optimization Patterns from Similar Kernels (cross-session memory)\n" + _mem_ctx
            logger.info("Cross-session memory injected into sub-agent task (%d chars)", len(_mem_ctx))
    except Exception as _mem_exc:
        logger.warning("Cross-session memory injection failed in dispatch: %s", _mem_exc)

    if meta.get("starting_patch"):
        cfg["starting_patch"] = meta["starting_patch"]

    for _passthrough_key in ("baseline_metrics", "benchmark_baseline"):
        if meta.get(_passthrough_key):
            cfg[_passthrough_key] = meta[_passthrough_key]
    cfg["allowed_skill_tiers"] = _task_allowed_skill_tiers(meta)

    return AgentTask(
        agent_class=agent_class,
        task=body,
        label=meta.get("label", task_file.stem),
        priority=int(meta.get("priority", 10)),
        kernel_language=meta.get("kernel_language", "python"),
        config=cfg,
        step_limit=task_step_limit,
        num_gpus=int(meta.get("num_gpus", 1)),
    )


def run_task_batch(
    task_files: list[Path],
    gpu_ids: list[int],
    output_dir: Path,
    model_factory,
    *,
    console=None,
    deadline=None,
    soft_stop=None,
    registry=None,
) -> dict[str, Any]:
    """Run a batch of task files via ParallelAgent pool mode.

    Parameters
    ----------
    task_files:
        List of task markdown file paths.
    gpu_ids:
        GPU device IDs to use.
    output_dir:
        Base output directory for results.
    model_factory:
        Callable returning a new model instance.
    console:
        Optional Rich console.
    deadline / soft_stop / registry:
        Optional wall-clock budget primitives forwarded to
        ``ParallelAgent.run_parallel`` so it can register spawned subprocesses
        in the registry, poll ``soft_stop`` between submissions, and clamp
        per-agent timeouts via ``deadline.cap()``.

    Returns
    -------
    dict with 'completed', 'failed', and 'results' keys.
    """
    from minisweagent.agents.parallel_agent import ParallelAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.run.task_file import read_task_file

    if not task_files:
        return {"completed": 0, "failed": 0, "results": []}

    tasks = [task_file_to_agent_task(f) for f in task_files]
    labels = [t.label for t in tasks]
    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicate_labels:
        raise ValueError(
            "Duplicate task labels are not allowed because result directories and task metadata are keyed by label: "
            + ", ".join(duplicate_labels)
        )

    # Determine repo_path and harness_path from first task's metadata
    meta_0, _ = read_task_file(task_files[0])
    repo_root = meta_0.get("repo_root")
    repo_path = Path(repo_root).resolve() if repo_root else Path.cwd()
    harness_path = meta_0.get("harness_path", "")

    is_git = False
    if repo_path.is_dir():
        is_git = (repo_path / ".git").exists() or (repo_path / ".git").is_file()

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent_config: dict[str, Any] = {
        "save_patch": True,
    }

    # Pre-seed GEAK_REPO_ROOT and GEAK_HARNESS so COMMANDMENT commands
    # can reference them as variables (no hardcoded paths).
    from minisweagent.run.pipeline_helpers import DEFAULT_AGENT_BENCHMARK_ITERATIONS
    from minisweagent.run.preprocess.harness_utils import harness_supports_iterations

    base_env_vars: dict[str, str] = {
        "GEAK_REPO_ROOT": str(repo_path.resolve()),
        "GEAK_BENCHMARK_ITERATIONS": str(DEFAULT_AGENT_BENCHMARK_ITERATIONS),
    }
    if harness_path:
        base_env_vars["GEAK_HARNESS"] = harness_path
        if harness_supports_iterations(harness_path):
            base_env_vars["GEAK_BENCHMARK_EXTRA_ARGS"] = f"--iterations {DEFAULT_AGENT_BENCHMARK_ITERATIONS}"
        else:
            logger.debug(
                "run_task_batch: harness %s does not declare --iterations; "
                "relying on GEAK_BENCHMARK_ITERATIONS=%s only",
                harness_path,
                DEFAULT_AGENT_BENCHMARK_ITERATIONS,
            )
    else:
        # No harness path available (e.g. eval_command flow). Preserve the
        # legacy behaviour of pre-seeding the EXTRA_ARGS so downstream
        # COMMANDMENT scripts that rely on the env var still see it; the
        # COMMANDMENT itself is responsible for matching its harness's
        # contract.
        base_env_vars["GEAK_BENCHMARK_EXTRA_ARGS"] = f"--iterations {DEFAULT_AGENT_BENCHMARK_ITERATIONS}"

    def env_factory():
        return LocalEnvironment(**{"cwd": str(repo_path.resolve()), "timeout": 3600, "env": base_env_vars})

    effective_model_factory = _build_ensemble_factory(model_factory)

    # region agent log
    emit_debug_log(
        "dispatch.py:run_task_batch:before_parallel",
        "Preparing to dispatch task batch",
        {
            "task_count": len(tasks),
            "gpu_ids": gpu_ids,
            "labels": labels,
            "duplicate_labels": duplicate_labels,
            "ensemble": os.environ.get("GEAK_MODEL_ENSEMBLE", "").strip() or None,
            "excluded_agents": os.environ.get("GEAK_EXCLUDED_AGENTS", "").strip() or None,
            "allowed_agents": os.environ.get("GEAK_ALLOWED_AGENTS", "").strip() or None,
            "results_dir": str(results_dir),
        },
        hypothesis_id="H5",
    )
    # endregion

    logger.info(
        "[bold yellow]Running %d sub-agent(s) in parallel:[/bold yellow]%s",
        len(tasks),
        "".join(f"\n  - {t.label} (priority={t.priority})" for t in tasks),
    )
    logger.info("[dim]Sub-agents are working — expect no output for several minutes.[/dim]")

    try:
        raw_results = ParallelAgent.run_parallel(
            num_parallel=len(gpu_ids),
            repo_path=repo_path,
            is_git_repo=is_git,
            task_content="",
            agent_class=tasks[0].agent_class if tasks else type(None),
            agent_config=agent_config,
            model_factory=effective_model_factory,
            env_factory=env_factory,
            base_patch_dir=results_dir,
            output=None,
            gpu_ids=gpu_ids,
            console=console,
            tasks=tasks,
            deadline=deadline,
            soft_stop=soft_stop,
            registry=registry,
        )
    except Exception as exc:
        logger.error("Task batch execution failed: %s", exc, exc_info=True)
        return {
            "completed": 0,
            "failed": len(tasks),
            "error": str(exc),
            "results": [],
        }

    completed = 0
    failed = 0
    summaries = []

    for entry in raw_results:
        agent_idx, _agent, exit_status, result = entry
        label = tasks[agent_idx].label if agent_idx < len(tasks) else f"task_{agent_idx}"
        success = exit_status not in ("error", "Error", None)
        if success:
            completed += 1
        else:
            failed += 1

        # Count patches written to the task's result directory
        task_result_dir = results_dir / label
        patch_count = len(list(task_result_dir.glob("*.patch"))) if task_result_dir.is_dir() else 0

        summaries.append(
            {
                "index": agent_idx,
                "label": label,
                "exit": str(exit_status),
                "patches": patch_count,
            }
        )

    # region agent log
    emit_debug_log(
        "dispatch.py:run_task_batch:after_parallel",
        "Parallel task batch completed",
        {
            "completed": completed,
            "failed": failed,
            "results_dir": str(results_dir),
            "summaries": summaries,
        },
        hypothesis_id="H5",
    )
    # endregion

    return {
        "completed": completed,
        "failed": failed,
        "results": summaries,
        "results_dir": str(results_dir),
    }


def run_from_task(
    task_file: Path,
    gpu_id: int = 0,
    output_dir: Path | None = None,
    model_factory=None,
    *,
    console=None,
) -> dict[str, Any]:
    """Run a single task file. Python-callable wrapper around geak --from-task.

    Shares the same underlying code as the CLI ``--from-task`` path but
    returns a results dict instead of printing to console.
    """
    # Default: tasks/round_N/00_label.md -> results/round_N/
    if output_dir:
        out = output_dir
    else:
        round_name = task_file.parent.name
        out = task_file.parent.parent.parent / "results" / round_name
    return run_task_batch(
        task_files=[task_file],
        gpu_ids=[gpu_id],
        output_dir=out,
        model_factory=model_factory,
        console=console,
    )
