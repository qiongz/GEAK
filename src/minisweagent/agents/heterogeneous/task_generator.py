"""Agent-based task generator -- produces optimization tasks by running a
read-only planning agent that inspects profiling data and kernel metadata.

The agent reads files via ``str_replace_editor view`` and submits a JSON
task list via the ``submit`` tool.  No rule-based fallback: an LLM model
is required.

Priority scheme (lower = higher priority, runs first):
  0  -- Algorithmic kernel-body rewrites (highest impact)
  5  -- Kernel fusion / advanced tuning
  10 -- Targeted optimization (autotune, memory, launch config)
  15 -- Profile-guided (generic fallback)

Usage (Python):
    from minisweagent.agents.heterogeneous.task_generator import generate_tasks
    tasks = generate_tasks(
        base_task_context=task_text,
        agent_class=StrategyAgent,
        model=model,
        kernel_path="/path/to/kernel.py",
        kernel_name="my_kernel",
        kernel_type="triton",
        profiling_path=Path("profile.json"),
        commandment_path=Path("COMMANDMENT.md"),
    )

Usage (CLI):
    python -m minisweagent.agents.heterogeneous.task_generator \\
        --kernel-path /path/to/kernel.py \\
        --profiling profiler_output.json \\
        --commandment COMMANDMENT.md \\
        --baseline-metrics baseline_metrics.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from minisweagent import get_repo_root
from minisweagent.agents.agent_spec import AgentTask
from minisweagent.agents.heterogeneous.prompts import (
    TASKGEN_INSTANCE_TEMPLATE as _INSTANCE_TEMPLATE,
)
from minisweagent.agents.heterogeneous.prompts import (
    TASKGEN_SYSTEM_PROMPT as _SYSTEM_PROMPT,
)
from minisweagent.agents.heterogeneous.prompts import (
    build_agent_restriction_addendum as _build_agent_restriction_addendum,
)
from minisweagent.agents.heterogeneous.result_scanning import (  # noqa: F401
    scan_previous_results as _scan_previous_results,
)
from minisweagent.agents.heterogeneous.result_scanning import (
    scan_previous_tasks as _scan_previous_tasks,
)
from minisweagent.agents.heterogeneous.workload_guidance import _build_workload_guidance  # noqa: F401
from minisweagent.debug_runtime import emit_debug_log, model_tools_snapshot, tool_names
from minisweagent.run.preprocess.discovery_types import (
    DEFAULT_SHAPE_COVERAGE_PROFILE,
    GLUON_BASELINE_PROFILE_MI3XX,
    GLUON_FEATURE_MODE_OFF,
    PREFER_AMD_GLUON_IF_VIABLE_POLICY,
    REQUIRE_AMD_GLUON_POLICY,
    SHAPE_COVERAGE_BUCKETED,
    SHAPE_COVERAGE_MULTI,
    SHAPE_COVERAGE_SINGLE,
    SHAPE_COVERAGE_UNKNOWN,
    _infer_kernel_language,
    allowed_skill_tiers_for_feature,
    build_gluon_feature_metadata,
    build_gluon_feature_prompt_block,
    derive_shape_coverage_profile,
    feature_uses_gluon_guidance,
    feature_uses_gluon_guidance_from_meta,
)

logger = logging.getLogger(__name__)
_GEAK_REPO_ROOT = get_repo_root()

_KNOWLEDGE_BASE_REL = "knowledge_base/optimization_strategies.py"
_GLUON_GUIDE_REL = "docs/triton_gluon.md"
_GLUON_KB_REL = "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-gluon-on-rocm.md"
_GLUON_EXAMPLES_REL = "examples/triton_gluon_inputs/README.md"
_TRITON_FAMILY_MARKERS = (
    "@triton",
    "tl.",
    "@gluon.jit",
    "triton.experimental.gluon",
    "from triton.experimental import gluon",
)


# ============================================================================
# Kernel metadata extraction
# ============================================================================


def _infer_kernel_type(kernel_path: Path) -> str:
    """Infer kernel_type from file content/extension when discovery.json is absent.

    For Python files, checks for direct Triton markers first, then follows
    ``from X import ...`` statements (up to 2 levels) to detect wrapper files
    that import Triton kernels from other modules.
    """
    ext = kernel_path.suffix.lower()
    if ext == ".py":
        try:
            text = kernel_path.read_text(errors="ignore")
            if any(marker in text for marker in _TRITON_FAMILY_MARKERS):
                logger.debug("_infer_kernel_type: triton markers found in %s", kernel_path.name)
                return "triton"
            if "import triton" in text or "triton.experimental.gluon" in text:
                if _check_imported_triton(text, kernel_path):
                    logger.debug("_infer_kernel_type: triton detected via import-follow in %s", kernel_path.name)
                    return "triton"
                logger.debug("_infer_kernel_type: Triton-family import in %s; classifying as triton.", kernel_path.name)
                return "triton"
            if "@flyc.kernel" in text or "flydsl.compiler" in text or "flydsl.expr" in text:
                return "flydsl"
        except OSError as exc:
            logger.debug("_infer_kernel_type: could not read %s: %s", kernel_path, exc)
        logger.debug("_infer_kernel_type: no triton markers in %s; returning 'unknown'.", kernel_path.name)
        return "unknown"
    if ext in (".cu", ".hip", ".hpp", ".cpp"):
        path_lower = str(kernel_path).lower()
        if "composable_kernel" in path_lower or "/ck_" in path_lower or "/ck/" in path_lower:
            logger.debug("_infer_kernel_type: CK path pattern in %s", kernel_path.name)
            return "ck"
        logger.debug("_infer_kernel_type: native extension %s → hip", ext)
        return "hip"
    logger.debug("_infer_kernel_type: unrecognised extension %s; returning 'unknown'.", ext)
    return "unknown"


def _check_imported_triton(content: str, file_path: Path, _depth: int = 0) -> bool:
    """Follow imports to check if any imported module contains @triton.jit."""
    if _depth > 2:
        return False

    import re
    import sys

    import_re = re.compile(r"^\s*from\s+([\w.]+)\s+import\s", re.MULTILINE)
    search_dirs = [file_path.parent]
    for sp in sys.path:
        p = Path(sp)
        if p.is_dir():
            search_dirs.append(p)

    for m in import_re.finditer(content):
        module_path = m.group(1).replace(".", "/")
        for base in search_dirs:
            candidate = base / f"{module_path}.py"
            if not candidate.is_file():
                candidate = base / module_path / "__init__.py"
            if not candidate.is_file():
                continue
            try:
                imported = candidate.read_text(errors="ignore")[:8192]
            except OSError:
                continue
            if "@triton.jit" in imported or "@triton.autotune" in imported or "@gluon.jit" in imported:
                return True
            if _depth < 2 and ("import triton" in imported or "triton.experimental.gluon" in imported):
                if _check_imported_triton(imported, candidate, _depth + 1):
                    return True
            break
    return False


def _extract_kernel_meta(
    discovery: dict | None,
    kernel_path: str,
) -> dict[str, Any]:
    """Build flat kernel metadata from a discovery.json dict and kernel path.

    When discovery.json is available, reads kernel_type from it directly.
    When absent, infers kernel_type from file extension and content.
    Other fields use simple defaults -- the LLM reads CODEBASE_CONTEXT.md
    for the full dependency tree, function names, and import relationships.
    """
    kp = Path(kernel_path) if kernel_path else Path("unknown.py")
    kernel_info = (discovery or {}).get("kernel") or {}
    raw_type = str(kernel_info.get("type") or "").strip().lower()
    ktype = _infer_kernel_type(kp) if raw_type in {"", "unknown"} else raw_type
    feature_meta = build_gluon_feature_metadata(
        kp,
        ktype,
        input_dialect=kernel_info.get("input_dialect") or os.getenv("GEAK_INPUT_DIALECT"),
        gluon_feature_mode=kernel_info.get("gluon_feature_mode") or os.getenv("GEAK_GLUON_FEATURE_MODE"),
        gluon_baseline_profile=kernel_info.get("gluon_baseline_profile")
        or os.getenv("GEAK_GLUON_BASELINE_PROFILE"),
        allowed_output_dialects=kernel_info.get("allowed_output_dialects"),
        preferred_output_dialects=kernel_info.get("preferred_output_dialects"),
        output_dialect_search_policy=kernel_info.get("output_dialect_search_policy"),
        target_backend=kernel_info.get("target_backend") or os.getenv("GEAK_TARGET_BACKEND"),
        # Preserve shape coverage signals when the preprocessor mirrored them
        # into discovery["kernel"]. Without this the CLI path (no baseline
        # metrics) silently drops shape_coverage_profile / benchmark_shape_count.
        benchmark_shape_count=kernel_info.get("benchmark_shape_count"),
        benchmark_test_cases=kernel_info.get("benchmark_test_cases"),
        shape_coverage_profile=kernel_info.get("shape_coverage_profile"),
    )

    return {
        "kernel_path": str(kp),
        "kernel_name": kernel_info.get("name", kp.stem),
        "kernel_type": ktype,
        "kernel_language": _infer_kernel_language(kp, ktype),
        "function_names": kernel_info.get("functions", []),
        "workspace_path": (discovery or {}).get("workspace", str(kp.parent)),
        **feature_meta,
    }


# ============================================================================
# Public API
# ============================================================================


def generate_tasks(
    base_task_context: str,
    agent_class: type,
    model: Any,
    *,
    kernel_path: str = "",
    kernel_name: str = "",
    kernel_type: str = "unknown",
    kernel_language: str = "python",
    function_names: list[str] | None = None,
    workspace_path: str = "",
    input_dialect: str = "plain_triton",
    gluon_feature_mode: str | None = None,
    gluon_baseline_profile: str = "raw",
    allowed_output_dialects: list[str] | None = None,
    preferred_output_dialects: list[str] | None = None,
    output_dialect_search_policy: str | None = None,
    target_backend: str = "",
    benchmark_shape_count: int | None = None,
    benchmark_test_cases: list[dict[str, Any]] | None = None,
    shape_coverage_profile: str | None = None,
    profiling_path: Path | None = None,
    commandment_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
    deep_search_path: Path | None = None,
    previous_results_dir: Path | None = None,
    discovery_path: Path | None = None,
    codebase_context_path: Path | None = None,
    previous_tasks_dir: Path | None = None,
    round_evaluations: list[dict[str, Any]] | None = None,
    current_round: int = 1,
    num_gpus: int = 1,
    output_dir: Path | None = None,
    rag_enabled: bool | None = None,
) -> list[AgentTask]:
    """Generate optimization tasks using an LLM planning agent.

    Args:
        base_task_context: Common context prepended to each task prompt.
        agent_class: Default agent class for tasks (typically StrategyAgent).
        model: LLM model instance (required).
        kernel_path: Absolute path to the kernel file.
        kernel_name: Human-readable kernel name.
        kernel_type: Backend type (triton, hip, cuda, ck, asm, unknown).
        kernel_language: Source language (python, cpp, asm).
        function_names: Key function names within the kernel file.
        workspace_path: Working directory for the planning agent.
        profiling_path: Path to kernel-profile JSON output.
        commandment_path: Path to COMMANDMENT.md.
        baseline_metrics_path: Path to baseline_metrics.json.
        deep_search_path: Path to deep search findings file.
        previous_results_dir: Path to previous round results directory.
        discovery_path: Path to the discovery.json file.
        codebase_context_path: Path to CODEBASE_CONTEXT.md file.
        previous_tasks_dir: Path to the parent tasks/ directory.
        round_evaluations: List of orchestrator round evaluation dicts.
        current_round: Current round number (for scanning prior tasks).
        rag_enabled: Whether RAG tools (query/optimize) are enabled. When
            False, RAG tools are excluded even if the MCP server is available.

    Returns:
        List of AgentTask sorted by priority.

    Raises:
        RuntimeError: If the agent fails to submit results.
    """
    if not kernel_path:
        logger.warning("generate_tasks: kernel_path is empty; returning no tasks.")
        return []

    submitted_text = _run_task_agent(
        kernel_path=kernel_path,
        kernel_name=kernel_name,
        kernel_type=kernel_type,
        kernel_language=kernel_language,
        function_names=function_names or [],
        workspace_path=workspace_path,
        input_dialect=input_dialect,
        gluon_feature_mode=gluon_feature_mode,
        gluon_baseline_profile=gluon_baseline_profile,
        allowed_output_dialects=allowed_output_dialects,
        preferred_output_dialects=preferred_output_dialects,
        output_dialect_search_policy=output_dialect_search_policy,
        target_backend=target_backend,
        benchmark_shape_count=benchmark_shape_count,
        benchmark_test_cases=benchmark_test_cases,
        shape_coverage_profile=shape_coverage_profile,
        base_task_context=base_task_context,
        model=model,
        profiling_path=profiling_path,
        commandment_path=commandment_path,
        baseline_metrics_path=baseline_metrics_path,
        deep_search_path=deep_search_path,
        previous_results_dir=previous_results_dir,
        discovery_path=discovery_path,
        codebase_context_path=codebase_context_path,
        previous_tasks_dir=previous_tasks_dir,
        round_evaluations=round_evaluations,
        current_round=current_round,
        num_gpus=num_gpus,
        output_dir=output_dir,
        rag_enabled=rag_enabled,
    )

    return _parse_llm_response(
        submitted_text,
        agent_class,
        kernel_path=kernel_path,
        commandment_path=str(commandment_path) if commandment_path else None,
        baseline_metrics_path=str(baseline_metrics_path) if baseline_metrics_path else None,
    )


def generate_tasks_from_content(
    base_task_context: str,
    agent_class: type,
    model: Any,
    *,
    kernel_path: str = "",
    kernel_name: str = "",
    kernel_type: str = "unknown",
    kernel_language: str = "python",
    function_names: list[str] | None = None,
    workspace_path: str = "",
    input_dialect: str = "plain_triton",
    gluon_feature_mode: str | None = None,
    gluon_baseline_profile: str = "raw",
    allowed_output_dialects: list[str] | None = None,
    preferred_output_dialects: list[str] | None = None,
    output_dialect_search_policy: str | None = None,
    target_backend: str = "",
    benchmark_shape_count: int | None = None,
    benchmark_test_cases: list[dict[str, Any]] | None = None,
    shape_coverage_profile: str | None = None,
    profiling_result: dict | None = None,
    commandment_content: str | None = None,
    baseline_metrics: dict | None = None,
    deep_search_content: str | None = None,
    previous_results_dir: Path | None = None,
    discovery_path: Path | None = None,
    codebase_context_path: Path | None = None,
    previous_tasks_dir: Path | None = None,
    round_evaluations: list[dict[str, Any]] | None = None,
    current_round: int = 1,
    num_gpus: int = 1,
    output_dir: Path | None = None,
    rag_enabled: bool | None = None,
) -> list[AgentTask]:
    """Convenience wrapper that materializes in-memory content to temp files.

    Use this when the caller has data in memory (dicts/strings) rather than
    on disk.  Each non-None content argument is written to a temporary file
    whose path is then forwarded to :func:`generate_tasks`.
    """
    tmp_files: list[Path] = []
    try:
        profiling_path = _write_temp(json.dumps(profiling_result, indent=2), ".json") if profiling_result else None
        if profiling_path:
            tmp_files.append(profiling_path)

        commandment_path = _write_temp(commandment_content, ".md") if commandment_content else None
        if commandment_path:
            tmp_files.append(commandment_path)

        baseline_metrics_path = (
            _write_temp(json.dumps(baseline_metrics, indent=2), ".json") if baseline_metrics else None
        )
        if baseline_metrics_path:
            tmp_files.append(baseline_metrics_path)

        deep_search_path = _write_temp(deep_search_content, ".md") if deep_search_content else None
        if deep_search_path:
            tmp_files.append(deep_search_path)

        return generate_tasks(
            base_task_context=base_task_context,
            agent_class=agent_class,
            model=model,
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            kernel_type=kernel_type,
            kernel_language=kernel_language,
            function_names=function_names,
            workspace_path=workspace_path,
            input_dialect=input_dialect,
            gluon_feature_mode=gluon_feature_mode,
            gluon_baseline_profile=gluon_baseline_profile,
            allowed_output_dialects=allowed_output_dialects,
            preferred_output_dialects=preferred_output_dialects,
            output_dialect_search_policy=output_dialect_search_policy,
            target_backend=target_backend,
            benchmark_shape_count=benchmark_shape_count,
            benchmark_test_cases=benchmark_test_cases,
            shape_coverage_profile=shape_coverage_profile,
            profiling_path=profiling_path,
            commandment_path=commandment_path,
            baseline_metrics_path=baseline_metrics_path,
            deep_search_path=deep_search_path,
            previous_results_dir=previous_results_dir,
            discovery_path=discovery_path,
            codebase_context_path=codebase_context_path,
            previous_tasks_dir=previous_tasks_dir,
            round_evaluations=round_evaluations,
            current_round=current_round,
            num_gpus=num_gpus,
            output_dir=output_dir,
            rag_enabled=rag_enabled,
        )
    finally:
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temp file %s", f)


def write_task_files(
    tasks: list[AgentTask],
    output_dir: Path,
    *,
    kernel_path: str = "",
    kernel_type: str = "",
    input_dialect: str = "plain_triton",
    gluon_feature_mode: str | None = None,
    gluon_baseline_profile: str = "raw",
    allowed_output_dialects: list[str] | None = None,
    preferred_output_dialects: list[str] | None = None,
    output_dialect_search_policy: str | None = None,
    target_backend: str = "",
    benchmark_shape_count: int | None = None,
    benchmark_test_cases: list[dict[str, Any]] | None = None,
    benchmark_test_cases_path: str | None = None,
    shape_coverage_profile: str | None = None,
    repo_root: str = "",
    commandment: str = "",
    baseline_metrics: str = "",
    profiling: str = "",
    codebase_context: str = "",
    benchmark_baseline: str = "",
    test_command: str = "",
    starting_patch: str = "",
    harness_path: str = "",
    round_num: int = 1,
) -> list[Path]:
    """Write AgentTask objects to .md task files on disk.

    Returns the list of written file paths.  Used by both the orchestrator
    tool (``tool_generate_tasks``) and the CLI (``main``).
    """
    from minisweagent.agents.agent_spec import _agent_class_to_type
    from minisweagent.run.task_file import write_task_file

    class_to_type = _agent_class_to_type()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    feature_meta = build_gluon_feature_metadata(
        Path(kernel_path) if kernel_path else Path("unknown.py"),
        kernel_type,
        input_dialect=input_dialect,
        gluon_feature_mode=gluon_feature_mode,
        gluon_baseline_profile=gluon_baseline_profile,
        allowed_output_dialects=allowed_output_dialects,
        preferred_output_dialects=preferred_output_dialects,
        output_dialect_search_policy=output_dialect_search_policy,
        target_backend=target_backend,
        benchmark_shape_count=benchmark_shape_count,
        benchmark_test_cases=benchmark_test_cases,
        shape_coverage_profile=shape_coverage_profile,
    )
    allowed_skill_tiers = allowed_skill_tiers_for_feature(
        kernel_type,
        gluon_feature_mode=feature_meta["gluon_feature_mode"],
        gluon_baseline_profile=feature_meta["gluon_baseline_profile"],
    )
    task_knowledge_paths = _resolve_task_knowledge_paths(
        Path(repo_root) if repo_root else (Path(kernel_path).parent if kernel_path else output_dir),
        kernel_type=kernel_type,
        feature_meta=feature_meta,
    )

    for t in tasks:
        filename = f"{t.priority:02d}_{t.label}.md"
        task_path = output_dir / filename
        metadata = {
            "label": t.label,
            "priority": t.priority,
            "agent_type": class_to_type.get(t.agent_class, "strategy_agent"),
            "kernel_language": t.kernel_language,
            "kernel_path": kernel_path,
            "kernel_type": kernel_type,
            "input_dialect": feature_meta["input_dialect"],
            "gluon_feature_mode": feature_meta["gluon_feature_mode"],
            "gluon_baseline_profile": feature_meta["gluon_baseline_profile"],
            "allowed_output_dialects": list(feature_meta["allowed_output_dialects"]),
            "preferred_output_dialects": list(feature_meta["preferred_output_dialects"]),
            "output_dialect_search_policy": feature_meta["output_dialect_search_policy"],
            "target_backend": feature_meta["target_backend"],
            "shape_coverage_profile": feature_meta.get("shape_coverage_profile"),
            "benchmark_shape_count": feature_meta.get("benchmark_shape_count"),
            "benchmark_test_cases": list(feature_meta.get("benchmark_test_cases") or []),
            "benchmark_test_cases_path": benchmark_test_cases_path or "",
            "repo_root": repo_root,
            "commandment": commandment,
            "baseline_metrics": baseline_metrics,
            "profiling": profiling,
            "codebase_context": codebase_context,
            "benchmark_baseline": benchmark_baseline,
            "starting_patch": starting_patch,
            "harness_path": harness_path,
            "num_gpus": t.num_gpus,
            "test_command": test_command,
            "round": round_num,
            "use_skills": _should_enable_skills(kernel_type),
            "allowed_skill_tiers": list(allowed_skill_tiers),
            **task_knowledge_paths,
        }
        body = f"# {t.label}\n\n{t.task}\n"
        write_task_file(task_path, metadata, body)
        paths.append(task_path)

    return paths


# ============================================================================
# Agent execution
# ============================================================================


def _write_temp(content: str, suffix: str) -> Path:
    """Write content to a temporary file and return its path."""
    fd, name = tempfile.mkstemp(suffix=suffix, prefix=".task_gen_")
    os.close(fd)
    Path(name).write_text(content)
    return Path(name)


def _find_repo_relative_file(workspace: Path, relative_path: str) -> Path | None:
    """Search for a repo-relative context file near the workspace."""
    for root in [workspace, workspace.parent, workspace.parent.parent]:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def _find_knowledge_base(workspace: Path) -> Path | None:
    """Locate the optimization strategies knowledge base file."""
    return _find_repo_relative_file(workspace, _KNOWLEDGE_BASE_REL)


def _resolve_task_knowledge_paths(
    workspace: Path,
    *,
    kernel_type: str = "",
    feature_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return the generic knowledge paths for a task run."""
    knowledge_base_path = _find_knowledge_base(workspace)
    feature_meta = feature_meta or {}

    uses_gluon_guidance = feature_uses_gluon_guidance(
        kernel_type,
        input_dialect=feature_meta.get("input_dialect"),
        gluon_feature_mode=feature_meta.get("gluon_feature_mode"),
        allowed_output_dialects=feature_meta.get("allowed_output_dialects"),
    )

    gluon_guide_path = (_GEAK_REPO_ROOT / _GLUON_GUIDE_REL).resolve() if uses_gluon_guidance else None
    gluon_kb_path = (_GEAK_REPO_ROOT / _GLUON_KB_REL).resolve() if uses_gluon_guidance else None
    gluon_examples_path = (_GEAK_REPO_ROOT / _GLUON_EXAMPLES_REL).resolve() if uses_gluon_guidance else None
    if gluon_guide_path and not gluon_guide_path.exists():
        gluon_guide_path = None
    if gluon_kb_path and not gluon_kb_path.exists():
        gluon_kb_path = None
    if gluon_examples_path and not gluon_examples_path.exists():
        gluon_examples_path = None

    primary_knowledge_path = knowledge_base_path
    if primary_knowledge_path is None and uses_gluon_guidance:
        primary_knowledge_path = gluon_kb_path or gluon_guide_path

    return {
        "knowledge_base_path": str(primary_knowledge_path) if primary_knowledge_path else "",
        "gluon_guide_path": str(gluon_guide_path) if gluon_guide_path else "",
        "gluon_kb_path": str(gluon_kb_path) if gluon_kb_path else "",
        "gluon_examples_path": str(gluon_examples_path) if gluon_examples_path else "",
    }


def _read_kernel_signal_text(
    *,
    kernel_path: str | Path,
    kernel_name: str = "",
    function_names: list[str] | None = None,
    baseline_metrics: dict[str, Any] | None = None,
    max_chars: int = 50000,
) -> str:
    """Collect lightweight planning signals from source and profiling metadata."""
    chunks: list[str] = [
        str(kernel_path),
        str(kernel_name),
        " ".join(function_names or []),
    ]
    try:
        chunks.append(Path(kernel_path).read_text(errors="ignore")[:max_chars])
    except OSError:
        logger.debug("Could not read kernel source for Gluon trait inference: %s", kernel_path)

    metrics = baseline_metrics or {}
    chunks.append(str(metrics.get("kernel_name", "")))
    for top in metrics.get("top_kernels", []) or []:
        if isinstance(top, dict):
            chunks.extend(str(top.get(key, "")) for key in ("name", "bottleneck"))

    return "\n".join(part for part in chunks if part).lower()


def _infer_gluon_planning_traits(
    feature_meta: dict[str, Any],
    signal_text: str,
) -> list[str]:
    """Infer composable traits that should shape Gluon task planning."""
    if not feature_uses_gluon_guidance_from_meta(feature_meta):
        return []

    traits: list[str] = ["semantics_contract"]
    input_dialect = str(feature_meta.get("input_dialect") or "").strip().lower()
    if input_dialect == "nv_gluon":
        traits.append("dialect_nv_gluon")
    elif input_dialect == "amd_gluon":
        traits.append("dialect_amd_gluon")
    else:
        traits.append("dialect_plain_triton")

    text = signal_text.lower()
    target_backend = str(feature_meta.get("target_backend") or "").lower()

    traits.append("layout_basic")
    if any(marker in text for marker in ("slicelayout", "expand_dims", "broadcast", "[:, none", "none, :", "mask")):
        traits.append("layout_slice_broadcast")
    if any(
        marker in text
        for marker in (
            "distributedlinearlayout",
            "partitionedsharedlayout",
            "tensordescriptor",
            "tensor_descriptor",
            "reshape",
            "permute",
            ".trans(",
            "prebuilt",
            "aot",
        )
    ):
        traits.append("layout_source_first_required")

    traits.append("memory_generic")
    if any(marker in text for marker in ("buffer_load", "buffer_store", ".amd.", "cdna3", "cdna4")) or any(
        arch in target_backend for arch in ("gfx942", "gfx950")
    ):
        traits.append("memory_amd_buffer")
    if any(
        marker in text
        for marker in (
            "allocate_shared_memory",
            "swizzledsharedlayout",
            "paddedsharedlayout",
            "async_copy",
            "async_load",
            "tdm",
            "descriptor",
            "tma",
        )
    ):
        traits.append("memory_shared_async_descriptor")

    has_scaled_marker = any(
        marker in text
        for marker in ("dot_scaled", "mfma_scaled", "wmma_scaled", "fp8", "e4m3", "e5m2", "e2m1", "scale")
    )
    has_dot_marker = any(marker in text for marker in ("tl.dot", "ttgl.dot", "dotoperandlayout", "mfma", "wmma"))
    has_wmma_descriptor = any(marker in text for marker in ("wmma", "amdwmmalayout", "gfx1250", "tdm", "descriptor"))
    if has_wmma_descriptor:
        traits.append("matrix_wmma_descriptor")
    if has_scaled_marker:
        traits.append("matrix_scaled_dot")
    if has_dot_marker:
        traits.append("matrix_dot")
    if not any(trait.startswith("matrix_") for trait in traits):
        traits.append("matrix_none")

    if any(marker in text for marker in ("gluon_jit_kernel_enabled", "prebuilt", "aot", "compile_gluon")):
        traits.append("execution_jit_aot_sensitive")
    if any(marker in text for marker in ("triton_version", "instr_shape", "amdmfmalayout", "triton.__version__")):
        traits.append("version_sensitive")
    if any(marker in text for marker in ("get_arch", "gfx942", "gfx950", "gfx1250", "is_hip", "gluon available")):
        traits.append("operator_support_sensitive")

    shape_profile = str(
        feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE
    ).lower()
    if shape_profile == SHAPE_COVERAGE_BUCKETED:
        traits.append("shape_coverage_bucketed")
    elif shape_profile == SHAPE_COVERAGE_MULTI:
        traits.append("shape_coverage_multi")
    elif shape_profile == SHAPE_COVERAGE_SINGLE:
        traits.append("shape_coverage_single")
    else:
        traits.append("shape_coverage_unknown")

    multi_shape = shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED)
    if multi_shape and any(
        marker in text
        for marker in (
            "block_m",
            "block_n",
            "block_k",
            "block_size",
            "num_warps",
            "instr_shape",
            "warps_per_cta",
            "threads_per_warp",
        )
    ):
        traits.append("shape_layout_constexpr_risk")
    if shape_profile == SHAPE_COVERAGE_BUCKETED and any(
        t in traits for t in ("matrix_dot", "matrix_scaled_dot", "matrix_wmma_descriptor")
    ):
        traits.append("shape_dispatch_required")

    deduped: list[str] = []
    seen: set[str] = set()
    for trait in traits:
        if trait not in seen:
            deduped.append(trait)
            seen.add(trait)
    return deduped


def _gluon_trait_headings(traits: list[str]) -> list[str]:
    """Return stable guide headings for targeted trait reading."""
    return [f"### Trait: {trait}" for trait in traits]


def _gluon_extension_strength(traits: list[str]) -> str:
    """Classify how strongly the current signals justify extra Gluon slots."""
    trait_set = set(traits)
    strong_traits = {
        "dialect_nv_gluon",
        "dialect_amd_gluon",
        "matrix_dot",
        "matrix_scaled_dot",
        "matrix_wmma_descriptor",
        "memory_amd_buffer",
        "layout_source_first_required",
        "shape_dispatch_required",
    }
    normal_traits = {
        "layout_slice_broadcast",
        "memory_shared_async_descriptor",
        "execution_jit_aot_sensitive",
        "version_sensitive",
        "operator_support_sensitive",
        "shape_layout_constexpr_risk",
        "shape_coverage_bucketed",
    }
    if trait_set & strong_traits:
        return "strong"
    if trait_set & normal_traits:
        return "normal"
    return "weak"


_PREVIOUS_GLUON_SPEEDUP_RE = re.compile(
    r"(?:verified_speedup|best[_\s-]*patch[_\s-]*speedup|speedup)\s*[=:]\s*([0-9]+\.?[0-9]*)\s*x?",
    re.IGNORECASE,
)


# Keywords that mark a previous-round task as Gluon-related when explicit
# dialect / policy frontmatter is missing. These cover label and body
# text patterns that show up in real Gluon-related rewrites
# (planner-emitted prompts and worker-produced patches).
_PREV_GLUON_KEYWORD_MARKERS = (
    "gluon",
    "amd_gluon",
    "nv_gluon",
    "mfma",
    "wmma_descriptor",
    "wmma_scaled",
    "buffer_load",
    "buffer_store",
    "amdmfmalayout",
    "amdwmmalayout",
    "dotoperandlayout",
    "gl.amd",
    "ttgl.",
    "@gluon.jit",
)


def _looks_gluon_related(block_text: str) -> bool:
    """Return whether a previous-round result/task block looks Gluon-related.

    Precision is the goal here: planner allocates an Extension Set slot to
    Gluon, and the previous-round signal is supposed to summarise *Gluon*
    outcomes only. A Base Set ``plain_triton`` win must not bleed into
    the Gluon quota math.

    First-pass uses explicit task frontmatter that ``scan_previous_tasks``
    now embeds (``input_dialect=`` / ``policy=``). Second-pass falls back
    to keyword markers so result-only sections still classify correctly.
    """
    if not block_text:
        return False
    lower = block_text.lower()
    if "input_dialect=amd_gluon" in lower or "input_dialect=nv_gluon" in lower:
        return True
    # Run-level metadata such as
    # ``policy=prefer_amd_gluon_if_viable_else_plain_triton`` appears on every
    # task in a Gluon-enabled run, including Base Set plain-Triton tasks. Strip
    # those fields before keyword matching so the policy itself does not make a
    # plain Triton result look like a Gluon Extension result.
    signal_lower = re.sub(
        r"\b(?:input_dialect|policy|output_dialect_search_policy)=[^,)\s]+",
        "",
        lower,
    )
    has_gluon_keyword = any(marker in signal_lower for marker in _PREV_GLUON_KEYWORD_MARKERS)
    if "input_dialect=plain_triton" in lower and not has_gluon_keyword:
        # Explicitly plain-Triton with no Gluon keywords: definitely not Gluon.
        return False
    if "policy=require_amd_gluon" in lower or "policy=prefer_amd_gluon" in lower:
        return "input_dialect=plain_triton" not in lower or has_gluon_keyword
    return has_gluon_keyword


_PREV_RESULT_SECTION_RE = re.compile(r"^### .*$", re.MULTILINE)
_PREV_TASK_BULLET_RE = re.compile(r"^- \*\*[^*]+\*\*.*$", re.MULTILINE)


def _filter_gluon_relevant_text(combined_text: str) -> str:
    """Strip Base Set plain-Triton sections from prior-round summary text.

    ``_scan_previous_results`` produces ``### <label>`` sections; the
    enriched ``_scan_previous_tasks`` produces ``- **<label>** (...)``
    bullets carrying frontmatter dialect/policy. Returns the concatenation
    of just the Gluon-related sections + bullets.

    When the input is free text without either structure (e.g. callers
    that pre-summarized the round in prose), fall back to whole-text
    classification so the legacy "single string" contract still works.
    """
    if not combined_text:
        return ""
    pieces: list[str] = []
    matched_any_structure = False

    section_starts = [m.start() for m in _PREV_RESULT_SECTION_RE.finditer(combined_text)]
    if section_starts:
        matched_any_structure = True
        boundaries = section_starts + [len(combined_text)]
        for i in range(len(section_starts)):
            section = combined_text[boundaries[i] : boundaries[i + 1]]
            if _looks_gluon_related(section):
                pieces.append(section)

    for line in combined_text.splitlines():
        if _PREV_TASK_BULLET_RE.match(line):
            matched_any_structure = True
            if _looks_gluon_related(line):
                pieces.append(line)

    if not matched_any_structure:
        # Free-text fallback: treat the whole blob as one block. This
        # preserves the legacy contract where callers passed pre-summarized
        # prose to ``_previous_gluon_signal`` directly.
        if _looks_gluon_related(combined_text):
            return combined_text
        return ""

    return "\n".join(pieces)


def _previous_gluon_signal(previous_text: str) -> str:
    """Infer a coarse previous-round Gluon signal from planner/result summaries.

    Classification priority is **numeric > textual**:

    1. The text is scanned for ``verified_speedup`` / ``best_patch_speedup``
       / ``speedup=N`` numbers. ``speedup > 1.0`` is the only signal that
       returns ``won``. ``speedup < 1.0`` (including ``0.0``) returns
       ``slower``.
    2. With no positive speedup, explicit terminal-failure markers
       (``traceback``, ``compile failed``, ``correctness failed``, etc.)
       return ``failed``.
    3. Slower-phrasing ("slower", "performance regression", "regressed",
       "worse") returns ``slower`` even without numeric backing.
    4. Otherwise the round is ``attempted`` -- bare markers like ``passed``
       or ``[best]`` without a positive structured speedup are treated as
       attempted, not won, because real benchmark output puts the speedup
       number alongside ``[BEST]`` and ``passed`` alone is consistent with
       both winning and slower-than-baseline outcomes.

    This guards the Base/Shared/Extension quota table against two
    specific over-corrections:

    - "Patch 0 failed, patch 5 [BEST] verified_speedup=1.18x" used to be
      mis-classified as ``failed`` (the Finding-2 mixed-failure case).
    - "Gluon best patch speedup=0.85 passed" used to be mis-classified
      as ``won`` once we added win markers, which would have wrongly
      expanded the next-round Extension Set.

    The caller is responsible for restricting this signal to Gluon
    contexts via ``feature_uses_gluon_guidance``; this function does not
    re-check the literal string ``"gluon"`` in the text because real
    task labels often omit it (e.g. ``mfma-rewrite``,
    ``layout-constexpr-fix``) and we would otherwise miss legitimate
    failed/slower/won signals on those rounds.
    """
    text = previous_text.lower()
    if not text:
        return "none"

    has_speedup_above_one = False
    has_speedup_below_one = False
    has_zero_speedup = False
    for match in _PREVIOUS_GLUON_SPEEDUP_RE.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value > 1.0:
            has_speedup_above_one = True
        elif value == 0.0:
            has_zero_speedup = True
        elif 0 < value < 1.0:
            has_speedup_below_one = True

    if has_speedup_above_one:
        return "won"

    slower_markers = ("slower", "performance regression", "regressed", "worse")
    if has_speedup_below_one or has_zero_speedup or any(
        marker in text for marker in slower_markers
    ):
        return "slower"

    failure_markers = (
        "compile failed",
        "compile error",
        "correctness failed",
        "all patches failed",
        "no patches passed",
        "traceback",
        "invalid",
        "failure",
    )
    if any(marker in text for marker in failure_markers):
        return "failed"
    # Standalone "failed" / "error" without explicit win evidence means the
    # prior round produced no usable best patch yet.
    if " failed" in text or "\nfailed" in text or "error:" in text:
        return "failed"

    # ``passed`` / bare ``[best]`` without numeric speedup evidence is
    # ambiguous -- fall through to ``attempted`` instead of guessing.
    return "attempted"


def _base_extension_quotas(
    num_gpus: int,
    strength: str,
    previous_signal: str,
    shape_profile: str = SHAPE_COVERAGE_UNKNOWN,
) -> tuple[int, int, int]:
    """Return recommended base/shared/extension slot counts.

    The shape coverage profile biases the allocation so that multi-shape
    benchmarks always retain at least one shape-robust Base slot and so
    that bucketed benchmarks can afford a paired Shared comparison plus
    an extra trait-specific Extension slot when the budget allows it.
    """
    budget = max(int(num_gpus or 1), 1)
    if budget <= 1:
        return 1, 0, 0
    if budget == 2:
        base, shared, extension = 1, 0, 1
    elif budget <= 4:
        base, shared, extension = 2, 1 if strength == "strong" and budget >= 4 else 0, 1
    else:
        base = max((budget + 1) // 2, 3)
        extension = 2 if strength == "strong" else 1
        shared = max(budget - base - extension, 0)

    if previous_signal == "failed":
        extension = min(extension, 1)
        shared = min(shared, 1)
    elif previous_signal == "slower":
        extension = min(extension, 1)
    elif previous_signal == "won" and budget >= 3:
        extension = min(extension + 1, max(budget - base, 1))
        shared = max(budget - base - extension, 0)

    if shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        # A shape-robust Base slot is mandatory whenever multiple shapes are
        # benchmarked together; Shared paired comparisons become valuable from
        # 4 GPUs onward; bucketed runs justify a second trait-specific
        # Extension slot once we have at least 5 GPUs.
        base = max(base, 1)
        if budget >= 4:
            shared = max(shared, 1)
        if shape_profile == SHAPE_COVERAGE_BUCKETED and budget >= 5:
            extension = max(extension, 2)

    if budget >= 2 and extension == 0:
        extension = 1
        if base + shared + extension > budget:
            shared = max(shared - 1, 0)
    while base + shared + extension > budget:
        if shared > 0:
            shared -= 1
        elif extension > 1:
            extension -= 1
        else:
            base -= 1
    return base, shared, extension


def _build_search_space_allocation_guidance(
    feature_meta: dict[str, Any],
    *,
    traits: list[str],
    num_gpus: int,
    previous_results_text: str = "",
    previous_tasks_text: str = "",
) -> str:
    """Return Base/Shared/Extension search-space allocation guidance."""
    if not traits or not feature_uses_gluon_guidance_from_meta(feature_meta):
        return ""
    strength = _gluon_extension_strength(traits)
    # Filter prior-round summary text to Gluon-related sections / bullets
    # only. Without this the Gluon quota responds to Base Set plain-Triton
    # outcomes (e.g. a Triton split-K winning at 1.5x would falsely
    # expand the next-round Extension Set even if Gluon never compiled).
    gluon_relevant_prior = _filter_gluon_relevant_text(
        "\n".join((previous_results_text, previous_tasks_text))
    )
    previous_signal = _previous_gluon_signal(gluon_relevant_prior)
    shape_profile = str(
        feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE
    ).lower()
    base_slots, shared_slots, extension_slots = _base_extension_quotas(
        num_gpus, strength, previous_signal, shape_profile=shape_profile,
    )

    lines = [
        "## Search Space Allocation",
        "- Treat AMD Gluon as an additive extension to the existing Triton search, not as a replacement.",
        f"- GPU/task budget: {max(int(num_gpus or 1), 1)}",
        f"- Gluon extension strength: {strength}",
        f"- Previous Gluon signal: {previous_signal}",
        f"- Shape coverage profile: {shape_profile}",
        "Recommended candidate allocation:",
        f"- Base Set (plain Triton): at least {base_slots} task(s). Preserve the main Triton planner path: algorithmic rewrite, memory/layout cleanup, fusion, shape-specialized variants, and low-priority autotune/launch work.",
        f"- Shared Set (Triton/Gluon common strategies): {shared_slots} task(s) when budget allows. Shared tasks must state whether they are a `plain_triton variant`, an `amd_gluon variant`, or a paired comparison.",
        f"- Extension Set (AMD Gluon): {extension_slots} task(s). Start with minimal viability, then trait-specific lowering only when traits justify it.",
        "Rules:",
        "- Do not replace all Base Set tasks with Gluon tasks.",
        "- Keep the same correctness and benchmark contract for all sets.",
        "- If a plain Triton candidate wins, accept it as the best result rather than forcing more Gluon work.",
        "- If a Triton strategy wins and maps cleanly to Gluon traits, a later round may create an AMD Gluon variant of that winning strategy.",
    ]
    if shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        lines.append(
            "- Multi-shape rule: at least one Base Set candidate MUST be `shape_robust`; "
            "Extension Set candidates MUST self-classify as `shape_robust` or `shape_bucketed` "
            "(not `single_shape_viability` after round 1)."
        )
    if shape_profile == SHAPE_COVERAGE_BUCKETED:
        lines.append(
            "- Bucketed rule: Shared Set tasks must be paired comparisons that report per-shape numbers; "
            "trait-specific Extension Set tasks must spell out their shape bucket and host-side dispatch."
        )
    if previous_signal == "failed":
        lines.append("- Because prior Gluon work appears to have failed, keep the next Extension Set to layout-only, translation-only, or memory-only work.")
    elif previous_signal == "slower":
        lines.append("- Because prior Gluon work appears slower, keep one targeted Gluon refinement focused on memory or matrix lowering; do not escalate to scheduler/persistent/async work.")
    elif previous_signal == "won":
        lines.append("- Because prior Gluon work appears promising, Gluon-specific refinement may expand, but keep a Base or Shared competitor.")
    return "\n".join(lines)


def _build_gluon_planning_traits_guidance(
    feature_meta: dict[str, Any],
    *,
    kernel_path: str | Path,
    kernel_name: str = "",
    function_names: list[str] | None = None,
    baseline_metrics: dict[str, Any] | None = None,
) -> str:
    """Return traits-based Gluon candidate guidance for the task planner."""
    signal_text = _read_kernel_signal_text(
        kernel_path=kernel_path,
        kernel_name=kernel_name,
        function_names=function_names,
        baseline_metrics=baseline_metrics,
    )
    traits = _infer_gluon_planning_traits(feature_meta, signal_text)
    if not traits:
        return ""

    trait_set = set(traits)
    lines = [
        "## Gluon Planning Traits",
        f"- Detected traits: {', '.join(f'`{trait}`' for trait in traits)}",
        "- Use these traits to allocate candidate task slots. They are planning constraints, not implementation templates.",
        "- Read guidance: do not read the whole Gluon guide first. In `docs/triton_gluon.md`, search for `## Quick section map for agents`, then search for these stable headings and view only those snippets:",
        *[f"  - `{heading}`" for heading in _gluon_trait_headings(traits)],
        "",
        "Prefer First:",
        "- Semantics-preserving candidate: preserve launcher shape, indexing, masks/boundaries, correctness oracle, and benchmark intent; do not claim success from compile-only validation.",
    ]

    if "dialect_nv_gluon" in trait_set:
        lines.append("- Dialect candidate: translate NVIDIA-facing Gluon assumptions to AMD-facing Gluon; do not rename APIs mechanically.")
    elif "dialect_amd_gluon" in trait_set:
        lines.append("- Dialect candidate: keep the existing AMD Gluon structure and optimize in-dialect before considering a plain Triton comparison.")
    else:
        lines.append("- Dialect candidate: create a minimal AMD Gluon viability rewrite with recognizable launcher, explicit layout, and correctness-preserving behavior.")

    lines.append("- Layout candidate: recover the base `BlockedLayout` from `tl.arange`, tile shape, `num_warps`, target family, and coalesced dimension; build host-side layouts as `constexpr` when they depend on launch choices.")

    if "layout_slice_broadcast" in trait_set:
        lines.append("- Layout candidate: preserve mask/broadcast/slice semantics with compatible `SliceLayout` or layout conversions instead of treating them as cleanup.")
    if "layout_source_first_required" in trait_set:
        lines.append("- Source-first candidate: read operator-local layout code before rewriting; preserve `DistributedLinearLayout`, descriptor, reshape/permute/trans, JIT/AOT, or prebuilt-kernel structure.")

    lines.append("- Fallback candidate: keep a plain Triton competitor when it remains an allowed output and compare by the same benchmark contract.")

    lines.extend(["", "Consider Next:"])
    if "memory_generic" in trait_set:
        lines.append("- Memory lowering: start with generic `gl.load` / `gl.store` for scalar or simple vector paths.")
    if "memory_amd_buffer" in trait_set:
        lines.append("- AMD memory lowering: evaluate `buffer_load` / `buffer_store` only when the target family or existing AMD structure makes it useful.")
    if "matrix_dot" in trait_set:
        lines.append("- Matrix lowering: plan result layout -> `DotOperandLayout` -> `convert_layout` -> target op (`mfma` / `wmma`) instead of direct `tl.dot` renaming.")
    if "matrix_scaled_dot" in trait_set:
        lines.append("- Scaled-matrix lowering: check dtype, scale layout, target arch, and CDNA4/gfx1250 constraints before assigning `mfma_scaled` or `wmma_scaled` work.")
    if "matrix_wmma_descriptor" in trait_set:
        lines.append("- WMMA/descriptor path: treat gfx1250 `wmma`, `tdm`, descriptor, cluster, and shared-layout constraints as a separate family from CDNA MFMA.")
    if any(trait in trait_set for trait in ("execution_jit_aot_sensitive", "version_sensitive", "operator_support_sensitive")):
        lines.append("- Runtime contract: verify JIT vs AOT, Triton minor version, `instr_shape` form, target backend, and operator-local arch guards before committing to a Gluon path.")

    lines.extend(
        [
            "",
            "Deprioritize Until Later:",
            "- Shared-memory swizzle, async copy, descriptor/tdm, scheduler hints, persistent kernels, atomics, and work-stealing until a simpler AMD Gluon candidate passes correctness.",
            "- Autotune-only or launch-only changes before at least one semantics-preserving Gluon candidate and one plain Triton competitor have been planned.",
            "- Tasks that mix NVIDIA and AMD layout families, introduce a top-level `gluon` kernel type, or produce an optimized `nv_gluon` output.",
            "",
            "Escalation rules:",
            "- Round 1 should include a minimal AMD Gluon viability task, one trait-specific AMD Gluon task, and a plain Triton fallback/competitor when allowed.",
            "- In later rounds, if Gluon failed, shrink the next attempt to layout-only, translation-only, or memory-only work; if correctness passed but performance regressed, escalate memory/matrix lowering before scheduler or persistent work.",
        ]
    )

    return "\n".join(lines)


def _build_gluon_task_generation_guidance(feature_meta: dict[str, Any]) -> str:
    """Return planner guidance for staging Gluon work by difficulty.

    This block must stay generic across Triton-family inputs and kernel families.
    It should shape task generation without hardcoding any sample-specific logic.
    """
    if not feature_uses_gluon_guidance_from_meta(feature_meta):
        return ""

    input_dialect = str(feature_meta.get("input_dialect") or "").strip().lower()
    search_policy = str(feature_meta.get("output_dialect_search_policy") or "").strip().lower()

    lines = [
        "## Gluon Task Staging Policy",
        "- Follow the standard GEAK progression for Gluon tasks:",
        "  1. minimal compileable AMD Gluon rewrite",
        "  2. semantics-preserving structural step",
        "  3. safer parallelization or decomposition",
        "  4. more aggressive multi-stage decomposition only after a passing AMD Gluon baseline exists",
        "  5. persistent scheduling, atomics, or work-stealing last",
        "- Favor early tasks that keep the original algorithm recognizable, make layout decisions explicit, and preserve correctness with the smallest possible semantic delta.",
        "- Treat the first passing AMD Gluon candidate as a platform for later aggressive optimizations, not as a final answer.",
    ]

    if input_dialect == "plain_triton":
        lines.extend(
            [
                "- For `plain_triton -> amd_gluon`, at least one early priority-0 task must target a minimal compileable amd_gluon rewrite with recognizable launcher, layout, and correctness semantics.",
                "- Do not spend the whole first batch on only the hardest persistent or work-stealing designs.",
            ]
        )
    elif input_dialect == "nv_gluon":
        lines.extend(
            [
                "- For `nv_gluon` inputs, early tasks should translate NVIDIA-facing semantics into AMD-facing Gluon first.",
                "- Defer aggressive algorithmic changes until at least one translation-style task preserves semantics on AMD.",
            ]
        )
    elif input_dialect == "amd_gluon":
        lines.extend(
            [
                "- For `amd_gluon` inputs, preserve the existing AMD Gluon structure first.",
                "- More aggressive in-dialect rewrites are allowed earlier because the baseline is already in the target dialect.",
            ]
        )

    if search_policy == REQUIRE_AMD_GLUON_POLICY:
        lines.append("- AMD Gluon is required for this run, but the first goal is still a passing baseline candidate before the most ambitious rewrites.")
    elif search_policy == PREFER_AMD_GLUON_IF_VIABLE_POLICY:
        lines.append("- Prefer AMD Gluon first, but keep one safer path alive before filling the batch with only ambitious Gluon strategies.")

    return "\n".join(lines)


def _build_gluon_planning_contract(feature_meta: dict[str, Any]) -> str:
    """Return planner-safe Gluon constraints.

    This block is intentionally about task decomposition and viability, not
    worker-level implementation detail.
    """
    if not feature_uses_gluon_guidance_from_meta(feature_meta):
        return ""

    lines = [
        "## Gluon Planning Contract",
        "- Your job is to generate valid optimization tasks, not to fully implement the kernel yourself.",
        "- Use Gluon knowledge as a planning constraint: decide what classes of tasks are viable, what order to try them in, and what failure modes to avoid.",
        "- Do not turn the task list into an implementation tutorial or a prose design document.",
        "- Do not let Gluon-specific guidance change the required output format: your `submit` payload must still be a JSON array of task objects and nothing else.",
        "- Preserve source semantics first when planning Gluon work:",
        "  - launcher shape",
        "  - indexing",
        "  - masks and boundary behavior",
        "  - correctness oracle",
        "  - benchmark intent",
        "- Treat Gluon as a Triton-family extension, not as a new top-level kernel type.",
        "- Prefer tasks that establish a correctness-passing AMD Gluon baseline before tasks that combine multiple difficult changes at once.",
    ]
    return "\n".join(lines)


def _build_gluon_failure_guardrails(feature_meta: dict[str, Any]) -> str:
    """Return high-level Gluon failure modes for planner use.

    This block should help the planner avoid assigning high-probability dead-end
    tasks without turning into a full execution guide.
    """
    if not feature_uses_gluon_guidance_from_meta(feature_meta):
        return ""

    lines = [
        "## Gluon Failure Guardrails",
        "- Plan around these high-frequency Gluon constraints:",
        "  - `zeros` and `full` often require explicit layout in AMD Gluon paths.",
        "  - `expand_dims` and `[:, None]` require compatible sliced layout context.",
        "  - `dot_fma` requires explicit accumulator and operand layout compatibility.",
        "  - layout warp geometry must agree with the runtime warp-size contract.",
        "  - cross-layout conversion is high-risk when parent distributed layouts do not match.",
        "- When the kernel structure includes any of the following, prefer source-first reading and safer tasks:",
        "  - `DistributedLinearLayout`",
        "  - `PartitionedSharedLayout`",
        "  - host `TensorDescriptor`",
        "  - `reshape`, `permute`, or `trans` used to unshuffle tiles",
        "  - nested 3D or 5D layout trees",
        "  - JIT/AOT gating or prebuilt-kernel loading",
        "  - scheduler, barrier, or priority hints that may affect correctness",
        "- Avoid task prompts that assume NVIDIA-facing Gluon APIs translate by name, or that compile-only success is enough.",
    ]
    return "\n".join(lines)


def _aggregate_prior_per_shape(
    round_evaluations: list[dict[str, Any]] | None,
) -> dict[str, dict[str, float]]:
    """Aggregate the most recent ``per_shape_speedups`` dict from round evals.

    The orchestrator already writes ``per_shape_speedups`` into each round
    evaluation; planner only needs the latest signal because the round-1
    decisions stay valid until contradicted by round-N data.
    """
    if not round_evaluations:
        return {}
    for rev in reversed(round_evaluations):
        if not isinstance(rev, dict):
            continue
        per_shape = rev.get("per_shape_speedups") or {}
        if isinstance(per_shape, dict) and per_shape:
            cleaned: dict[str, dict[str, float]] = {}
            for shape, info in per_shape.items():
                if not isinstance(info, dict):
                    continue
                speedup = info.get("speedup")
                if isinstance(speedup, (int, float)):
                    cleaned[str(shape)] = {
                        "speedup": float(speedup),
                        "baseline_ms": float(info.get("baseline_ms") or 0.0),
                        "candidate_ms": float(info.get("candidate_ms") or 0.0),
                    }
            if cleaned:
                return cleaned
    return {}


def _build_shape_coverage_guidance(
    feature_meta: dict[str, Any],
    *,
    traits: list[str],
    prior_per_shape: dict[str, dict[str, float]] | None = None,
) -> str:
    """Return planner guidance for multi-shape benchmarks.

    Generic by design: this block applies to any kernel_type whose
    benchmark exposes more than one shape via baseline_metrics. It does
    not depend on Gluon-specific traits, but tightens the Gluon path
    further when ``shape_layout_constexpr_risk`` /
    ``shape_dispatch_required`` are present.
    """
    profile = str(
        feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE
    ).lower()
    if profile not in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        return ""
    test_cases = feature_meta.get("benchmark_test_cases") or []
    shape_count = feature_meta.get("benchmark_shape_count") or len(test_cases) or "?"
    trait_set = set(traits)

    lines = [
        "## Shape Coverage Policy",
        f"- Detected shape coverage profile: `{profile}` ({shape_count} cases).",
        "- Correctness AND performance share the SAME multi-shape case set. "
        "A patch that fails any shape's correctness is invalid, even if it is faster on others.",
        "- Treat overall speedup as the geomean / mean across all cases. "
        "Single-shape wins do not justify accepting per-shape regressions.",
        "- Each task_prompt MUST self-classify as one of:",
        "  - `single_shape_viability` (early correctness baseline only; not promotable without a multi-shape variant)",
        "  - `shape_robust` (one parameterization that should hold across all observed shapes)",
        "  - `shape_bucketed` (multiple internal variants + an explicit, documented dispatch condition)",
        "- Do NOT hardcode `M`, `N`, `K`, `seq_len`, batch, or hidden size literals "
        "unless the task is explicitly `shape_bucketed` AND the dispatch condition is documented.",
    ]
    if test_cases:
        lines.append("- Observed cases (from baseline benchmark):")
        for case in list(test_cases)[:8]:
            params = case.get("params") if isinstance(case, dict) else {}
            cid = case.get("case_id") if isinstance(case, dict) else None
            baseline_ms = (
                case.get("baseline_ms") if isinstance(case, dict) else None
            )
            lines.append(
                f"  - `{cid or '?'}` params={params or {}} "
                + (f"baseline_ms={baseline_ms:.5f}" if isinstance(baseline_ms, (int, float)) else "baseline_ms=?")
            )
        if len(test_cases) > 8:
            lines.append(f"  - ... ({len(test_cases) - 8} more cases)")
    if profile == SHAPE_COVERAGE_BUCKETED:
        lines.append(
            "- Because shapes span multiple buckets (small / medium / large), plan for "
            "shape-bucketed dispatch backed by autotune or shape-specialized kernels gated "
            "by host-side selection."
        )
    if "shape_layout_constexpr_risk" in trait_set:
        lines.append(
            "- Gluon `BLOCK_*` / `num_warps` / layout `instr_shape` are `constexpr`. "
            "Every Gluon candidate must state how it stays valid across all observed shapes "
            "(parametric layout, host-built layout passed as `constexpr`, or shape-bucketed dispatch)."
        )
    if "shape_dispatch_required" in trait_set:
        lines.append(
            "- For matrix paths (MFMA / WMMA / scaled-dot), `instr_shape` and operand layouts "
            "depend on `BLOCK_M/N/K` and target arch. Plan paired candidates per bucket; "
            "a single MFMA tile size rarely wins across all buckets."
        )
    if prior_per_shape:
        slow = sorted(
            shape
            for shape, info in prior_per_shape.items()
            if (info.get("speedup") or 1.0) < 0.95
        )
        fast = sorted(
            shape
            for shape, info in prior_per_shape.items()
            if (info.get("speedup") or 1.0) > 1.10
        )
        if slow:
            lines.append(
                f"- Prior round per-shape regressions on shapes {slow}. "
                "Next round MUST include at least one task that explicitly addresses these regressed shapes."
            )
        if fast and slow:
            lines.append(
                f"- Prior round wins on {fast} but lost on {slow}. "
                "Treat this as evidence the candidate is shape-overfit; require a shape-robust "
                "or shape-bucketed competitor in the new batch."
            )
    return "\n".join(lines)


def _should_enable_skills(kernel_type: str) -> bool:
    """Enable skills only on the Triton planning/execution path."""
    return kernel_type.strip().lower() == "triton"


def _build_output_dialect_guidance(feature_meta: dict[str, Any]) -> str:
    """Return planner guidance for how to prioritize output dialects."""
    policy = str(feature_meta.get("output_dialect_search_policy") or "").strip().lower()
    input_dialect = str(feature_meta.get("input_dialect") or "").strip().lower()
    preferred_outputs = feature_meta.get("preferred_output_dialects") or []
    preferred_text = ", ".join(str(item) for item in preferred_outputs)
    if policy == PREFER_AMD_GLUON_IF_VIABLE_POLICY:
        lines = [
            "## Output Dialect Planning Policy",
            f"- Preferred output dialect order: {preferred_text}",
            "- Treat output dialect selection as a planning constraint, not as a post-hoc preference.",
            "- Prefer an AMD Gluon candidate first for this Triton-family input whenever the structure and target backend make that path viable.",
            "- Do not spend the whole first batch on fallback-only tuning when AMD Gluon remains viable.",
        ]
        if input_dialect == "nv_gluon":
            lines.append("- This input starts as NVIDIA-facing Gluon. Generate at least one early task that translates vendor-specific APIs, layout assumptions, or memory paths into AMD-facing Gluon before tuning.")
        elif input_dialect == "amd_gluon":
            lines.append("- This input is already AMD Gluon. Keep the main optimization path inside AMD Gluon rather than drifting back to plain Triton unless benchmark evidence forces a fallback.")
        else:
            lines.append("- Generate at least one early task that explicitly evaluates an AMD Gluon rewrite before filling the batch with only plain Triton tuning.")
        lines.append("- Keep a plain Triton fallback path alive when it remains an allowed output. If the kernel structure or benchmark evidence suggests Gluon is unlikely to help, say so and focus the remaining tasks on the fallback path.")
        return "\n".join(lines)
    if policy == REQUIRE_AMD_GLUON_POLICY:
        return "\n".join(
            [
                "## Output Dialect Planning Policy",
                f"- Preferred output dialect order: {preferred_text}",
                "- Treat output dialect selection as a planning constraint, not as a post-hoc preference.",
                "- AMD Gluon output is required for this run. Do not plan plain Triton-only fallback tasks as the main path.",
                "- At least one earliest-priority task must target a compileable AMD Gluon path.",
            ]
        )
    return ""


def _run_task_agent(
    *,
    kernel_path: str,
    kernel_name: str,
    kernel_type: str,
    kernel_language: str,
    function_names: list[str],
    workspace_path: str,
    input_dialect: str,
    gluon_feature_mode: str | None,
    gluon_baseline_profile: str,
    allowed_output_dialects: list[str] | None,
    target_backend: str,
    base_task_context: str,
    model: Any,
    profiling_path: Path | None,
    commandment_path: Path | None,
    baseline_metrics_path: Path | None,
    deep_search_path: Path | None,
    previous_results_dir: Path | None,
    discovery_path: Path | None,
    codebase_context_path: Path | None = None,
    previous_tasks_dir: Path | None = None,
    round_evaluations: list[dict[str, Any]] | None = None,
    current_round: int = 1,
    num_gpus: int = 1,
    output_dir: Path | None = None,
    rag_enabled: bool | None = None,
    preferred_output_dialects: list[str] | None = None,
    output_dialect_search_policy: str | None = None,
    benchmark_shape_count: int | None = None,
    benchmark_test_cases: list[dict[str, Any]] | None = None,
    shape_coverage_profile: str | None = None,
) -> str:
    """Run a read-only planning agent and return the submitted JSON text."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.tools.tools_runtime import ToolRuntime

    workspace = Path(workspace_path) if workspace_path else Path(kernel_path).parent
    feature_meta = build_gluon_feature_metadata(
        Path(kernel_path),
        kernel_type,
        input_dialect=input_dialect,
        gluon_feature_mode=gluon_feature_mode,
        gluon_baseline_profile=gluon_baseline_profile,
        allowed_output_dialects=allowed_output_dialects,
        preferred_output_dialects=preferred_output_dialects,
        output_dialect_search_policy=output_dialect_search_policy,
        target_backend=target_backend,
        benchmark_shape_count=benchmark_shape_count,
        benchmark_test_cases=benchmark_test_cases,
        shape_coverage_profile=shape_coverage_profile,
    )

    _allowed_names = {"str_replace_editor", "submit"}
    if rag_enabled is not False:
        _allowed_names |= {"query", "optimize"}
    read_only_tools = [t for t in ToolRuntime.fetch_tools_list() if t["name"] in _allowed_names]
    # AmdLlmModel forwards set_tools() to its _impl; snapshot the actual target.
    _model_target = getattr(model, "_impl", model)
    original_tools = list(_model_target.tools) if hasattr(_model_target, "tools") else None
    # region agent log
    emit_debug_log(
        "task_generator.py:_run_task_agent:before_override",
        "Replacing model tools for task-planning sub-agent",
        {
            "workspace": str(workspace),
            "read_only_tools": tool_names(read_only_tools),
            "original_tools": tool_names(original_tools),
            "model_target_type": type(_model_target).__name__,
            "model_target_id": id(_model_target),
            "model_before": model_tools_snapshot(model),
        },
        hypothesis_id="H1",
    )
    # endregion
    if hasattr(model, "set_tools"):
        model.set_tools(read_only_tools)
    else:
        _model_target.tools = read_only_tools

    tmp_files: list[Path] = []
    try:
        env = LocalEnvironment(cwd=str(workspace))
        knowledge_paths = _resolve_task_knowledge_paths(
            workspace,
            kernel_type=kernel_type,
            feature_meta=feature_meta,
        )

        prev_results_path: Path | None = None
        prev_results_summary = ""
        if previous_results_dir and Path(previous_results_dir).is_dir():
            summary = _scan_previous_results(Path(previous_results_dir))
            if summary:
                prev_results_summary = summary
                prev_results_path = _write_temp(summary, "_prev_results.md")
                tmp_files.append(prev_results_path)

        prev_tasks_path: Path | None = None
        prev_tasks_summary = ""
        if previous_tasks_dir and Path(previous_tasks_dir).is_dir() and current_round > 1:
            tasks_summary = _scan_previous_tasks(Path(previous_tasks_dir), current_round)
            if tasks_summary:
                prev_tasks_summary = tasks_summary
                prev_tasks_path = _write_temp(tasks_summary, "_prev_tasks.md")
                tmp_files.append(prev_tasks_path)

        round_evals_path: Path | None = None
        if round_evaluations:
            evals_text = "## Orchestrator Round Evaluations\n\n"
            for rev in round_evaluations:
                r_num = rev.get("round", "?")
                evals_text += f"### Round {r_num}\n"
                evals_text += f"- Best task: {rev.get('best_task', 'N/A')}\n"
                fb = rev.get("full_benchmark", {})
                canonical_speedup = (
                    fb.get("verified_speedup", "N/A")
                    if isinstance(fb, dict) and fb
                    else rev.get("benchmark_speedup", "N/A")
                )
                evals_text += f"- Canonical benchmark speedup: {canonical_speedup}x\n"
                if fb:
                    evals_text += f"- Verified kernel time: {fb.get('kernel_time_ms', 'N/A')}ms\n"
                profile = rev.get("profile_comparison", {})
                if profile:
                    evals_text += f"- Profile comparison: {json.dumps(profile, default=str)[:500]}\n"
                per_shape = rev.get("per_shape_speedups") or {}
                if isinstance(per_shape, dict) and per_shape:
                    regressed: list[str] = []
                    evals_text += "- Per-shape speedups:\n"
                    for shape, info in per_shape.items():
                        if not isinstance(info, dict):
                            continue
                        sp = info.get("speedup")
                        if isinstance(sp, (int, float)):
                            evals_text += f"  - {shape}: {sp:.3f}x\n"
                            if sp < 0.95:
                                regressed.append(str(shape))
                    if regressed:
                        evals_text += f"- Per-shape regressions on: {regressed}\n"
                evals_text += f"- Best patch: {rev.get('best_patch', 'N/A')}\n\n"
            round_evals_path = _write_temp(evals_text, "_round_evals.md")
            tmp_files.append(round_evals_path)

        # Pre-load baseline metrics so we can enrich feature_meta with shape
        # information before building any guidance blocks.
        #
        # baseline_metrics.json is the authoritative source for shape
        # signals because the preprocessor writes it AFTER actually
        # running ``perf_cmd`` on the kernel. discovery.json may carry an
        # older / inaccurate profile (e.g. ``multi`` recorded before the
        # bucketed Arena harness was wired up). When baseline_metrics
        # has explicit ``benchmark_test_cases``, treat it as canonical
        # and let it overwrite stale discovery values.
        _bm_dict: dict[str, Any] = {}
        if baseline_metrics_path and Path(baseline_metrics_path).exists():
            try:
                _bm_dict = json.loads(Path(baseline_metrics_path).read_text())
            except (OSError, ValueError) as exc:
                logger.debug("Failed to read baseline metrics for shape coverage: %s", exc)
                _bm_dict = {}
        _bm_cases = (_bm_dict or {}).get("benchmark_test_cases") if isinstance(_bm_dict, dict) else None
        _bm_count = (_bm_dict or {}).get("benchmark_shape_count") if isinstance(_bm_dict, dict) else None
        baseline_has_authoritative_cases = bool(_bm_cases)

        if baseline_has_authoritative_cases:
            feature_meta["benchmark_test_cases"] = list(_bm_cases or [])
            if _bm_count is not None:
                feature_meta["benchmark_shape_count"] = _bm_count
            # Re-derive shape_coverage_profile from authoritative baseline
            # data, replacing any value that came in via discovery / kwargs.
            feature_meta["shape_coverage_profile"] = derive_shape_coverage_profile(
                benchmark_shape_count=feature_meta.get("benchmark_shape_count"),
                benchmark_test_cases=feature_meta.get("benchmark_test_cases"),
            )
        else:
            if not feature_meta.get("benchmark_shape_count") and _bm_count is not None:
                feature_meta["benchmark_shape_count"] = _bm_count
            if not feature_meta.get("benchmark_test_cases"):
                feature_meta["benchmark_test_cases"] = list(_bm_cases or [])
            if (
                str(feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE).lower()
                == DEFAULT_SHAPE_COVERAGE_PROFILE
            ):
                feature_meta["shape_coverage_profile"] = derive_shape_coverage_profile(
                    benchmark_shape_count=feature_meta.get("benchmark_shape_count"),
                    benchmark_test_cases=feature_meta.get("benchmark_test_cases"),
                )

        template_vars = {
            "kernel_path": kernel_path,
            "kernel_name": kernel_name,
            "kernel_type": kernel_type,
            "kernel_language": kernel_language,
            "function_names": ", ".join(function_names) if function_names else "",
            "gluon_feature_context": build_gluon_feature_prompt_block(
                feature_meta,
                heading="## Gluon Feature Context",
            ),
            "output_dialect_guidance": _build_output_dialect_guidance(feature_meta),
            "gluon_planning_contract": _build_gluon_planning_contract(feature_meta),
            "codebase_context_path": str(codebase_context_path) if codebase_context_path else "",
            "discovery_path": str(discovery_path) if discovery_path else "",
            "profiling_path": str(profiling_path) if profiling_path else "",
            "commandment_path": str(commandment_path) if commandment_path else "",
            "baseline_metrics_path": str(baseline_metrics_path) if baseline_metrics_path else "",
            **knowledge_paths,
            "deep_search_path": str(deep_search_path) if deep_search_path else "",
            "previous_results_path": str(prev_results_path) if prev_results_path else "",
            "previous_tasks_path": str(prev_tasks_path) if prev_tasks_path else "",
            "round_evaluations_path": str(round_evals_path) if round_evals_path else "",
            "base_task_context": base_task_context,
            "num_gpus": num_gpus,
            "memory_context": "",
            "workload_guidance": "",
            "gluon_planning_traits_guidance": "",
            "search_space_allocation_guidance": "",
            "shape_coverage_guidance": "",
            "gluon_task_generation_guidance": _build_gluon_task_generation_guidance(feature_meta),
            "gluon_failure_guardrails": _build_gluon_failure_guardrails(feature_meta),
        }

        try:
            from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
                assemble_memory_context,
            )
            from minisweagent.memory.working_notebook import (  # pylint: disable=import-error,no-name-in-module
                summarize_working_notebook,
            )

            if not _bm_dict and baseline_metrics_path and Path(baseline_metrics_path).exists():
                _bm_dict = json.loads(Path(baseline_metrics_path).read_text())
            _notebook_dir = None
            if baseline_metrics_path and Path(baseline_metrics_path).exists():
                _notebook_dir = Path(baseline_metrics_path).resolve().parent / "_working_memory"
            elif previous_results_dir and Path(previous_results_dir).is_dir():
                _notebook_dir = Path(previous_results_dir).resolve().parent.parent / "_working_memory"
            _wm_ctx = summarize_working_notebook(_notebook_dir)
            _mem = assemble_memory_context(
                kernel_path=kernel_path,
                bottleneck_type=_bm_dict.get("bottleneck"),
                profiling_metrics=_bm_dict,
            )
            combined_memory = "\n\n".join(part.strip() for part in (_wm_ctx, _mem or "") if part and str(part).strip())
            if combined_memory:
                template_vars["memory_context"] = combined_memory
        except Exception as exc:
            logger.warning("Memory assembly failed in task generator: %s", exc)

        _kernel_meta = {
            "file_path": kernel_path,
            "kernel_name": kernel_name,
            "kernel_type": kernel_type,
        }
        signal_text = _read_kernel_signal_text(
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            function_names=function_names,
            baseline_metrics=_bm_dict,
        )
        gluon_traits = _infer_gluon_planning_traits(feature_meta, signal_text)
        template_vars["gluon_planning_traits_guidance"] = _build_gluon_planning_traits_guidance(
            feature_meta,
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            function_names=function_names,
            baseline_metrics=_bm_dict,
        )
        template_vars["search_space_allocation_guidance"] = _build_search_space_allocation_guidance(
            feature_meta,
            traits=gluon_traits,
            num_gpus=num_gpus,
            previous_results_text=prev_results_summary,
            previous_tasks_text=prev_tasks_summary,
        )
        template_vars["shape_coverage_guidance"] = _build_shape_coverage_guidance(
            feature_meta,
            traits=gluon_traits,
            prior_per_shape=_aggregate_prior_per_shape(round_evaluations),
        )
        template_vars["workload_guidance"] = _build_workload_guidance(_kernel_meta, _bm_dict)

        tg_step_limit = int(os.getenv("GEAK_TASKGEN_STEP_LIMIT", "200"))
        tg_cost_limit = float(os.getenv("GEAK_TASKGEN_COST_LIMIT", "50.0"))

        # Inject RAG tools section if query/optimize tools are available
        _has_rag = any(t["name"] in ("query", "optimize") for t in read_only_tools)
        if _has_rag:
            _rag_section = (
                "### **Knowledge Base Lookup** (Recommended)\n\n"
                "- Use `query` tool to search for optimization techniques, "
                "hardware-specific tips, and code patterns relevant to this kernel\n"
                "- Use `optimize` tool to get targeted optimization suggestions "
                "based on your kernel type and bottleneck analysis\n"
                "- Integrate retrieved knowledge into your strategy planning\n\n"
            )
        else:
            _rag_section = ""
        system_prompt = _SYSTEM_PROMPT.replace("__RAG_TOOLS_SECTION__", _rag_section)
        system_prompt = system_prompt + _build_agent_restriction_addendum()
        use_skills = _should_enable_skills(kernel_type) and not feature_uses_gluon_guidance(
            kernel_type,
            input_dialect=feature_meta["input_dialect"],
            gluon_feature_mode=feature_meta["gluon_feature_mode"],
            allowed_output_dialects=feature_meta["allowed_output_dialects"],
        )
        allowed_skill_tiers = allowed_skill_tiers_for_feature(
            kernel_type,
            gluon_feature_mode=feature_meta["gluon_feature_mode"],
            gluon_baseline_profile=feature_meta["gluon_baseline_profile"],
        )
        agent = DefaultAgent(
            model,
            env,
            system_template=system_prompt,
            instance_template=_INSTANCE_TEMPLATE,
            step_limit=tg_step_limit,
            cost_limit=tg_cost_limit,
            use_skills=use_skills,
            allowed_skill_tiers=allowed_skill_tiers,
        )

        # Write per-turn conversation log for debugging
        if output_dir:
            _log_dir = Path(output_dir)
            _log_dir.mkdir(parents=True, exist_ok=True)
            agent.log_file = _log_dir / "task_generator.log"

        _context_files = [
            k
            for k in (
                "profiling_path",
                "commandment_path",
                "baseline_metrics_path",
                "codebase_context_path",
                "previous_results_path",
                "round_evaluations_path",
            )
            if template_vars.get(k)
        ]
        logger.info(
            "[bold yellow]Starting task-generation agent[/bold yellow] "
            "(step_limit=%d, cost=%.1f, context=%s) — this may take a few minutes",
            tg_step_limit,
            tg_cost_limit,
            ", ".join(k.replace("_path", "") for k in _context_files) or "minimal",
        )

        _t0 = time.monotonic()
        exit_type, exit_msg = agent.run(
            task="generate optimization tasks",
            **template_vars,
        )
        _elapsed = time.monotonic() - _t0

        if exit_type == "Submitted":
            logger.info(
                "[bold green]Task-generation agent completed[/bold green] in %.1fs (%d chars).",
                _elapsed,
                len(exit_msg),
            )
            return exit_msg

        logger.warning("Task-generation agent did not submit (exit_type=%s).", exit_type)
        raise RuntimeError(f"Task-generation agent did not submit results (exit: {exit_type}): {exit_msg[:500]}")
    finally:
        if original_tools is not None:
            if hasattr(model, "set_tools"):
                model.set_tools(original_tools)
            else:
                _model_target.tools = original_tools
        # region agent log
        emit_debug_log(
            "task_generator.py:_run_task_agent:after_restore",
            "Finished task-planning tool restore",
            {
                "restored_tools": tool_names(original_tools),
                "model_target_type": type(_model_target).__name__,
                "model_target_id": id(_model_target),
                "model_after": model_tools_snapshot(model),
            },
            hypothesis_id="H1",
        )
        # endregion
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temp file %s", f)


def _parse_llm_response(
    content: str,
    agent_class: type,
    *,
    kernel_path: str | None = None,
    commandment_path: str | None = None,
    baseline_metrics_path: str | None = None,
) -> list[AgentTask]:
    """Parse JSON response into AgentTask objects."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    raw_tasks = json.loads(content)
    if not isinstance(raw_tasks, list):
        raise TypeError(f"Expected JSON array, got {type(raw_tasks).__name__}")

    from minisweagent.agents.agent_spec import _agent_type_to_class, filter_agent_type

    type_to_class = _agent_type_to_class()

    tasks: list[AgentTask] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            logger.debug("_parse_llm_response: skipping non-dict item: %s", type(item).__name__)
            continue

        label = str(item.get("label", "unknown"))
        try:
            priority = int(item.get("priority", 10))
        except (ValueError, TypeError):
            logger.debug(
                "_parse_llm_response: invalid priority %r for '%s'; defaulting to 10.", item.get("priority"), label
            )
            priority = 10
        priority = max(0, min(15, priority))
        agent_type = filter_agent_type(str(item.get("agent_type", "strategy_agent")))
        kernel_language = str(item.get("kernel_language", "python"))
        task_prompt = str(item.get("task_prompt", ""))
        try:
            task_num_gpus = max(1, int(item.get("num_gpus", 1)))
        except (ValueError, TypeError):
            task_num_gpus = 1

        if not task_prompt:
            logger.debug("_parse_llm_response: skipping task '%s' with empty prompt.", label)
            continue

        resolved_class = type_to_class.get(agent_type, agent_class)
        if agent_type not in type_to_class:
            logger.debug("_parse_llm_response: unknown agent_type %r for '%s'; using default class.", agent_type, label)

        cfg: dict[str, Any] = {}

        tasks.append(
            AgentTask(
                agent_class=resolved_class,
                task=task_prompt,
                label=label,
                priority=priority,
                kernel_language=kernel_language,
                config=cfg,
                num_gpus=task_num_gpus,
            )
        )

    if not tasks:
        raise ValueError("LLM response contained no valid tasks")

    return sorted(tasks, key=lambda t: t.priority)


# ============================================================================
# CLI helpers
# ============================================================================


# ============================================================================
# CLI
# ============================================================================


def main():
    """Generate optimization tasks from the command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate optimization tasks using an LLM planning agent",
    )
    parser.add_argument("--kernel-path", default=None, help="Path to the kernel file")
    parser.add_argument(
        "--from-discovery",
        default=None,
        metavar="FILE",
        help="Read discovery.json and extract kernel-path and repo-root",
    )
    parser.add_argument("--profiling", default=None, help="Path to kernel-profile JSON output")
    parser.add_argument("--commandment", default=None, help="Path to COMMANDMENT.md")
    parser.add_argument("--baseline-metrics", default=None, help="Path to baseline_metrics.json")
    parser.add_argument("--model", default=None, help="Model name (default: from config/env)")
    parser.add_argument("--repo-root", default=None, help="Repository root (for discovery)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="DIR",
        help="Write task files to this directory (one .md per task) instead of JSON to stdout",
    )
    parser.add_argument(
        "--from-results",
        default=None,
        metavar="DIR",
        help="Previous round results directory (for iterative refinement)",
    )
    parser.add_argument(
        "--deep-search",
        default=None,
        metavar="FILE",
        help="Path to deep search findings (JSON or Markdown file)",
    )
    parser.add_argument(
        "--codebase-context",
        default=None,
        metavar="FILE",
        help="Path to CODEBASE_CONTEXT.md (auto-detected from --from-discovery directory if not set)",
    )
    parser.add_argument(
        "--benchmark-baseline",
        default=None,
        metavar="FILE",
        help="Path to benchmark_baseline.txt (canonical benchmark output from preprocessing)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Round number for task file frontmatter (default: 1)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of available GPUs (guides task count and GPU allocation, default: 1)",
    )
    from minisweagent.run.pipeline_helpers import add_agent_filter_args, apply_agent_filter_env

    add_agent_filter_args(parser)

    args = parser.parse_args()
    apply_agent_filter_env(args)

    # Populate from discovery JSON if provided (explicit flags override)
    disc_json = None
    test_command = None
    if args.from_discovery:
        disc_json = json.loads(Path(args.from_discovery).read_text())
        if not args.kernel_path:
            args.kernel_path = (disc_json.get("kernel") or {}).get("file")
        if not args.repo_root:
            args.repo_root = disc_json.get("workspace")
        focused = disc_json.get("focused_test") or {}
        if focused.get("focused_command"):
            test_command = focused["focused_command"]
        else:
            for t in disc_json.get("tests") or []:
                if t.get("command"):
                    test_command = t["command"]
                    break

    # Auto-detect codebase context from --from-discovery directory
    if not args.codebase_context and args.from_discovery:
        _ctx_sibling = Path(args.from_discovery).parent / "CODEBASE_CONTEXT.md"
        if _ctx_sibling.exists():
            args.codebase_context = str(_ctx_sibling)

    if not args.kernel_path:
        parser.error("--kernel-path is required (or provide --from-discovery)")

    kernel_path = Path(args.kernel_path).resolve()
    if not kernel_path.exists():
        print(f"ERROR: kernel path not found: {args.kernel_path}", file=sys.stderr)
        sys.exit(1)

    if not disc_json:
        # No pre-computed discovery JSON -- run automated-test-discovery
        print(f"[task-generator] Running discovery on {kernel_path}...", file=sys.stderr)
        try:
            from automated_test_discovery.server import discover as atd_discover

            _discover_fn = getattr(atd_discover, "fn", atd_discover)
            disc_json = _discover_fn(
                kernel_path=str(kernel_path),
                output_dir=str(kernel_path.parent),
            )
        except Exception as e:
            print(f"ERROR: discovery failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[task-generator] Loading discovery from {args.from_discovery}...", file=sys.stderr)

    kernel_meta = _extract_kernel_meta(disc_json, str(kernel_path))

    if not kernel_meta["kernel_path"] or kernel_meta["kernel_path"] == "unknown.py":
        print("ERROR: no kernel found in discovery", file=sys.stderr)
        sys.exit(1)

    # Create model (REQUIRED)
    try:
        from minisweagent.run.pipeline_helpers import load_geak_model

        model = load_geak_model(args.model or os.environ.get("GEAK_MODEL"))
        print(f"[task-generator] Using model: {model.config.model_name}", file=sys.stderr)
    except Exception as e:
        print(
            f"ERROR: task-generator requires an LLM model. Set GEAK_MODEL or use --model. ({e})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Placeholder agent class for CLI output
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    agent_class = StrategyInteractiveAgent

    base_task_context = f"Optimize the kernel at {kernel_path} for maximum performance."

    # Resolve file paths (pass through to the agent, not loaded into memory)
    profiling_path = Path(args.profiling).resolve() if args.profiling else None
    commandment_path = Path(args.commandment).resolve() if args.commandment else None
    baseline_metrics_path = Path(args.baseline_metrics).resolve() if args.baseline_metrics else None
    deep_search_path = Path(args.deep_search).resolve() if args.deep_search else None
    previous_results_dir = Path(args.from_results).resolve() if args.from_results else None
    discovery_path = Path(args.from_discovery).resolve() if args.from_discovery else None
    codebase_context_path = Path(args.codebase_context).resolve() if args.codebase_context else None

    # Generate tasks
    tasks = generate_tasks(
        base_task_context=base_task_context,
        agent_class=agent_class,
        model=model,
        kernel_path=kernel_meta["kernel_path"],
        kernel_name=kernel_meta["kernel_name"],
        kernel_type=kernel_meta["kernel_type"],
        kernel_language=kernel_meta["kernel_language"],
        function_names=kernel_meta["function_names"],
        workspace_path=kernel_meta["workspace_path"],
        input_dialect=kernel_meta["input_dialect"],
        gluon_feature_mode=kernel_meta["gluon_feature_mode"],
        gluon_baseline_profile=kernel_meta["gluon_baseline_profile"],
        allowed_output_dialects=kernel_meta["allowed_output_dialects"],
        preferred_output_dialects=kernel_meta["preferred_output_dialects"],
        output_dialect_search_policy=kernel_meta["output_dialect_search_policy"],
        target_backend=kernel_meta["target_backend"],
        profiling_path=profiling_path,
        commandment_path=commandment_path,
        baseline_metrics_path=baseline_metrics_path,
        deep_search_path=deep_search_path,
        previous_results_dir=previous_results_dir,
        discovery_path=discovery_path,
        codebase_context_path=codebase_context_path,
        num_gpus=args.num_gpus,
    )

    # Print summary to stderr
    print(f"\n[task-generator] Generated {len(tasks)} task(s):\n", file=sys.stderr)
    for t in tasks:
        print(f"  [{t.priority:2d}] {t.label} ({t.kernel_language})", file=sys.stderr)

    # Output: directory of task files or JSON to stdout
    if args.output:
        out_dir = Path(args.output)
        task_paths = write_task_files(
            tasks,
            out_dir,
            kernel_path=str(kernel_path),
            kernel_type=kernel_meta["kernel_type"],
            input_dialect=kernel_meta["input_dialect"],
            gluon_feature_mode=kernel_meta["gluon_feature_mode"],
            gluon_baseline_profile=kernel_meta["gluon_baseline_profile"],
            allowed_output_dialects=kernel_meta["allowed_output_dialects"],
            preferred_output_dialects=kernel_meta["preferred_output_dialects"],
            output_dialect_search_policy=kernel_meta["output_dialect_search_policy"],
            target_backend=kernel_meta["target_backend"],
            repo_root=args.repo_root or "",
            commandment=args.commandment or "",
            baseline_metrics=args.baseline_metrics or "",
            profiling=args.profiling or "",
            codebase_context=args.codebase_context or "",
            benchmark_baseline=args.benchmark_baseline or "",
            test_command=test_command or "",
            round_num=args.round,
        )

        manifest = [
            {
                "index": i,
                "label": tasks[i].label,
                "priority": tasks[i].priority,
                "kernel_language": tasks[i].kernel_language,
                "file": str(f),
            }
            for i, f in enumerate(task_paths)
        ]

        print(f"\n[task-generator] Wrote {len(tasks)} task file(s) to {out_dir}/", file=sys.stderr)
        print(json.dumps(manifest, indent=2))
    else:
        output = []
        for i, t in enumerate(tasks):
            output.append(
                {
                    "index": i,
                    "label": t.label,
                    "priority": t.priority,
                    "kernel_language": t.kernel_language,
                    "task_prompt_preview": t.task[:300] + ("..." if len(t.task) > 300 else ""),
                }
            )
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
