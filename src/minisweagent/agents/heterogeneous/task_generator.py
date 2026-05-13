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
    GLUON_FEATURE_MODE_FORCE,
    GLUON_FEATURE_MODE_FORCE_L0_ANCHOR,
    GLUON_FEATURE_MODE_OFF,
    GLUON_FEATURE_MODE_REQUIRE_VIABLE,
    PREFER_AMD_GLUON_IF_VIABLE_POLICY,
    REQUIRE_AMD_GLUON_POLICY,
    SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR,
    SOURCE_ORIGIN_GENERATED_OVERLAY,
    SOURCE_ORIGIN_NV_GLUON_TRANSLATION,
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
from minisweagent.run.resource_paths import resolve_project_resource
from minisweagent.run.gluon_doc_profiles import (
    add_unique_doc_key,
    required_doc_keys_for_profile,
    task_requires_gluon_worker_docs,
)
from minisweagent.run.target_contracts import (
    do_not_clauses,
    filter_abstract_target_symbols,
    forbidden_symbols_from_scoped_text,
    split_patch_and_route_symbols,
    target_symbols_from_actionable_prose,
    target_symbols_from_scoped_text,
)

logger = logging.getLogger(__name__)
_GEAK_REPO_ROOT = get_repo_root()
_LAST_GLUON_TASK_GENERATION_DIAGNOSTICS: dict[str, Any] = {}

_KNOWLEDGE_BASE_REL = "knowledge_base/optimization_strategies.py"
_PLAIN_TRITON_KB_FALLBACK_RELS = (
    "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-on-rocm.md",
    "knowledge-base/amd-knowledge-base/layer-5-llm/05-advanced/custom-kernels/triton-kernels.md",
    "knowledge-base/INDEX.md",
)
_GLUON_SKILL_REL = "skills/triton-gluon/SKILL.md"
_GLUON_KB_REL = "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-gluon-on-rocm.md"
_GLUON_EXAMPLES_REL = "examples/triton_gluon_inputs/README.md"
_GLUON_SPLIT_DOC_RELS = {
    "gluon_always_read_path": "skills/triton-gluon/docs/00_always_read.md",
    "gluon_search_policies_path": "skills/triton-gluon/docs/10_search_policies.md",
    "gluon_component_traits_path": "skills/triton-gluon/docs/20_component_traits.md",
    "gluon_architecture_notes_path": "skills/triton-gluon/docs/30_architecture_notes.md",
    "gluon_examples_doc_path": "skills/triton-gluon/docs/40_examples.md",
    "gluon_api_reference_path": "skills/triton-gluon/docs/50_api_reference.md",
    "gluon_real_patterns_path": "skills/triton-gluon/docs/60_real_patterns.md",
    "gluon_backup_details_path": "skills/triton-gluon/docs/70_backup_details.md",
}
_TRITON_FAMILY_MARKERS = (
    "@triton",
    "tl.",
    "@gluon.jit",
    "triton.experimental.gluon",
    "from triton.experimental import gluon",
)


def get_last_gluon_task_generation_diagnostics() -> dict[str, Any]:
    """Return diagnostics from the most recent task-generation parse."""
    return json.loads(json.dumps(_LAST_GLUON_TASK_GENERATION_DIAGNOSTICS))


def _set_last_gluon_task_generation_diagnostics(diagnostics: dict[str, Any]) -> None:
    _LAST_GLUON_TASK_GENERATION_DIAGNOSTICS.clear()
    _LAST_GLUON_TASK_GENERATION_DIAGNOSTICS.update(diagnostics)

_BASE_FAMILY_SWIZZLE_TILE = "base_swizzle_and_tile_schedule"
_BASE_FAMILY_SPLIT_K = "base_split_k_or_multipass_reduce"
_BASE_FAMILY_SCALED_FUSION = "base_scaled_dot_fusion"
_BASE_FAMILY_STREAMLINE = "base_hot_path_streamline"
_BASE_FAMILY_PERSISTENT = "base_small_matrix_persistent_or_launch_amortization"
_BASE_FAMILY_GENERIC_ALGO = "base_algorithmic_rewrite"
_BASE_FAMILY_GENERIC_FUSION = "base_fusion_or_launch_reduction"
_BASE_FAMILY_GENERIC_MEMORY = "base_memory_layout_cleanup"
_BASE_FAMILY_CONFIG_DISPATCH = "base_config_or_shape_dispatch"
_BASE_FAMILY_PIPELINE_STAGE = "base_pipeline_stage_boundary"
_BASE_FAMILY_AOT_JIT = "base_aot_jit_integration"
_BASE_FAMILY_PRESHUFFLE_DESCRIPTOR = "base_preshuffle_or_descriptor_layout"
_BASE_FAMILY_SPLITK_REDUCE_PIPELINE = "base_splitk_reduce_pipeline"

_BASE_FAMILY_DETAILS: dict[str, str] = {
    _BASE_FAMILY_SWIZZLE_TILE: (
        "tile schedule, swizzle/traversal, BLOCK_M/N/K, GROUP_SIZE_M, num_warps, and launch mapping"
    ),
    _BASE_FAMILY_SPLIT_K: "split-K, partial accumulation, separate reduction, or multi-pass reduce",
    _BASE_FAMILY_SCALED_FUSION: "scale_a/scale_b fusion into tl.dot or the epilogue without extra launches",
    _BASE_FAMILY_STREAMLINE: (
        "redundant casts, mask/broadcast cleanup, pointer CSE, live-range and register-pressure reduction"
    ),
    _BASE_FAMILY_PERSISTENT: "small-matrix persistent, multi-tile per program, workqueue, or launch amortization",
    _BASE_FAMILY_GENERIC_ALGO: "algorithmic kernel-body rewrite",
    _BASE_FAMILY_GENERIC_FUSION: "operation fusion or launch-count reduction",
    _BASE_FAMILY_GENERIC_MEMORY: "memory/layout cleanup on the hottest path",
    _BASE_FAMILY_CONFIG_DISPATCH: "shape/config dispatch, checked-in config, or explicit bucket selection",
    _BASE_FAMILY_PIPELINE_STAGE: "multi-stage operator boundary, helper/reduce stage, or measured-output dependency",
    _BASE_FAMILY_AOT_JIT: "JIT/AOT/prebuilt artifact, signature, target triple, or fallback integration",
    _BASE_FAMILY_PRESHUFFLE_DESCRIPTOR: "preshuffle/unshuffle, descriptor, tensor descriptor, or cache-layout contract",
    _BASE_FAMILY_SPLITK_REDUCE_PIPELINE: "split-K partial output plus reduce pipeline or multi-kernel reduction",
}

_BASE_FAMILY_GLUON_OVERLAY_REASONS: dict[str, str] = {
    _BASE_FAMILY_SWIZZLE_TILE: "explicit_layout or shape_bucket when layout control is the bottleneck",
    _BASE_FAMILY_SPLIT_K: "matrix_lowering only when the split/decomposition keeps the same algorithmic shape",
    _BASE_FAMILY_SCALED_FUSION: "matrix_lowering or dialect_specific_memory for scaled dot/epilogue paths",
    _BASE_FAMILY_STREAMLINE: "explicit_layout or buffer_path for a measured memory/index hot path",
    _BASE_FAMILY_PERSISTENT: "local_subpath_win only; prefer plain Triton/HIP launch amortization first",
    _BASE_FAMILY_GENERIC_ALGO: "local_subpath_win or explicit_layout tied to the same algorithmic rewrite",
    _BASE_FAMILY_GENERIC_FUSION: "same fused operation expressed in AMD Gluon with a same-ABI comparison",
    _BASE_FAMILY_GENERIC_MEMORY: "explicit_layout or buffer_path for the measured memory/layout cleanup",
    _BASE_FAMILY_CONFIG_DISPATCH: "shape_bucket or config-driven dispatch only with visible host-side selection",
    _BASE_FAMILY_PIPELINE_STAGE: "local_subpath_win only when the Gluon stage feeds measured output",
    _BASE_FAMILY_AOT_JIT: "jit_aot_sensitive or operator_artifact_integration; preserve product fallback",
    _BASE_FAMILY_PRESHUFFLE_DESCRIPTOR: "tdm_descriptor_path or source-first layout extraction after operator-local evidence",
    _BASE_FAMILY_SPLITK_REDUCE_PIPELINE: "split/reduce Gluon variant only when partial/reduce boundaries remain auditable",
}

_BASE_FAMILY_LABEL_MARKERS: dict[str, tuple[str, ...]] = {
    _BASE_FAMILY_SWIZZLE_TILE: ("swizzle", "tile", "tiling", "schedule", "group_size"),
    _BASE_FAMILY_SPLIT_K: ("split-k", "split_k", "split k", "multipass", "multi-pass", "reduction"),
    _BASE_FAMILY_SCALED_FUSION: ("scale-fusion", "scale fusion", "scaled_dot", "scale", "fused-scale"),
    _BASE_FAMILY_STREAMLINE: ("streamline", "redundant", "simplify", "cleanup", "eliminate"),
    _BASE_FAMILY_PERSISTENT: ("persistent", "launch-amortization", "launch amortization", "workqueue"),
    _BASE_FAMILY_GENERIC_ALGO: ("algorithm", "rewrite"),
    _BASE_FAMILY_GENERIC_FUSION: ("fusion", "fuse"),
    _BASE_FAMILY_GENERIC_MEMORY: ("memory", "coalesc", "layout"),
    _BASE_FAMILY_CONFIG_DISPATCH: ("config", "shape dispatch", "bucket", "json", "heuristic"),
    _BASE_FAMILY_PIPELINE_STAGE: ("pipeline", "stage", "helper", "measured output", "reduce stage"),
    _BASE_FAMILY_AOT_JIT: ("aot", "jit", "prebuilt", "artifact", "signature"),
    _BASE_FAMILY_PRESHUFFLE_DESCRIPTOR: ("preshuffle", "unshuffle", "descriptor", "tensordescriptor", "tdm"),
    _BASE_FAMILY_SPLITK_REDUCE_PIPELINE: ("split-k", "splitk", "partial output", "reduce pipeline"),
}

_OPTIONAL_GLUON_TASK_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("source_origin", "Source origin"),
    ("gluon_tl_policy", "Gluon TL policy"),
    ("allowed_tl_symbols", "Allowed TL symbols"),
    ("forbidden_tl_symbols", "Forbidden TL symbols"),
    ("layout_construction_policy", "Layout construction policy"),
    ("execution_mode", "Execution mode"),
    ("aot_signature_contract", "AOT signature contract"),
    ("target_triple", "Target triple"),
    ("divisibility_hints", "Divisibility hints"),
    ("scratch_requirement_check", "Scratch requirement check"),
    ("prebuilt_artifact_contract", "Prebuilt artifact contract"),
    ("jit_aot_fallback_preservation", "JIT AOT fallback preservation"),
    ("target_stage", "Target stage"),
    ("target_kernel_role", "Target kernel role"),
    ("upstream_stage", "Upstream stage"),
    ("downstream_stage", "Downstream stage"),
    ("measured_output_dependency", "Measured output dependency"),
    ("integration_boundary", "Integration boundary"),
    ("task_type", "Task type"),
    ("primary_component", "Primary component"),
    ("primary_atomic_component", "Primary atomic component"),
    ("extension_intent", "Extension intent"),
    ("expected_outcome", "Expected outcome"),
    ("not_viable_for_l1_if_slower_than_base", "Not viable for L1 if slower than Base"),
    ("overhead_source_to_record", "Overhead source to record"),
    ("l0_scope_classification", "L0 scope classification"),
    ("l0_coupling_reasons", "L0 coupling reasons"),
    ("expected_failure_layers", "Expected failure layers"),
    ("first_patch_compile_goal", "First patch compile goal"),
    ("do_not_optimize_before_compile", "Do not optimize before compile"),
    ("matrix_lowering_required", "Matrix lowering required"),
    ("declared_failure_layer", "Declared failure layer"),
    ("changed_failure_layer", "Changed failure layer"),
    ("task_signals", "Task signals"),
    ("routed_doc_reasons", "Routed doc reasons"),
    ("kernel_family_signal", "Kernel family signal"),
    ("failure_layers", "Failure layers"),
    ("minimum_executable_unit", "Minimum executable unit"),
    ("executed_route_symbols", "Executed route symbols"),
    ("target_symbol", "Target symbol"),
    ("target_component", "Target component"),
    ("forbidden_change", "Forbidden change"),
    ("allowed_execution_path", "Allowed execution path"),
    ("scope_infeasible_policy", "Scope infeasible policy"),
    ("whole_kernel_required_reason", "Whole kernel required reason"),
)

_VALID_SEARCH_SETS = {"base", "shared", "extension"}
_VALID_REQUIRED_OUTPUT_DIALECTS = {"plain_triton", "amd_gluon", "mixed", "any"}


def _task_config_requires_gluon_worker_docs(
    *,
    kernel_type: str,
    cfg: dict[str, Any],
    task_body: str,
    label: str,
) -> bool:
    return task_requires_gluon_worker_docs(
        kernel_type=kernel_type,
        required_output=str(cfg.get("required_output_dialect") or ""),
        implementation_layer=str(cfg.get("implementation_layer") or ""),
        extension_layer=str(cfg.get("extension_layer") or ""),
        doc_profile=str(cfg.get("gluon_doc_profile") or ""),
        task_body=task_body,
        label=label,
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
        source_origin=kernel_info.get("source_origin"),
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
    audit_diagnostics_dir: Path | None = None,
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

    required_base_families, expected_extension_slots = _task_generation_audit_inputs(
        kernel_path=kernel_path,
        kernel_name=kernel_name,
        kernel_type=kernel_type,
        function_names=function_names or [],
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
        baseline_metrics_path=baseline_metrics_path,
        previous_results_dir=previous_results_dir,
        previous_tasks_dir=previous_tasks_dir,
        current_round=current_round,
        num_gpus=num_gpus,
    )
    parse_feature_meta = build_gluon_feature_metadata(
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
    _patch_feature_meta_from_baseline_metrics(parse_feature_meta, baseline_metrics_path)

    return _parse_llm_response(
        submitted_text,
        agent_class,
        kernel_path=kernel_path,
        commandment_path=str(commandment_path) if commandment_path else None,
        baseline_metrics_path=str(baseline_metrics_path) if baseline_metrics_path else None,
        required_base_families=required_base_families,
        expected_extension_slots=expected_extension_slots,
        audit_diagnostics_dir=audit_diagnostics_dir,
        gluon_feature_mode=gluon_feature_mode,
        feature_meta=parse_feature_meta,
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


def _patch_feature_meta_from_baseline_metrics(
    feature_meta: dict[str, Any],
    baseline_metrics_path: str | Path | None,
) -> str | None:
    """Patch feature metadata with authoritative shape data from baseline metrics.

    ``_run_task_agent`` uses this information to render planner guidance. Task
    files need the same patch before frontmatter is written so workers that read
    metadata, rather than planner prose, receive the multi-shape contract too.
    """
    if not baseline_metrics_path:
        return None
    path = Path(baseline_metrics_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.debug("Failed to read baseline metrics for shape metadata: %s", exc)
        return None
    if not isinstance(payload, dict):
        return None

    cases = payload.get("benchmark_test_cases")
    count = payload.get("benchmark_shape_count")
    if cases:
        feature_meta["benchmark_test_cases"] = list(cases or [])
        if count is not None:
            feature_meta["benchmark_shape_count"] = count
        feature_meta["shape_coverage_profile"] = derive_shape_coverage_profile(
            benchmark_shape_count=feature_meta.get("benchmark_shape_count"),
            benchmark_test_cases=feature_meta.get("benchmark_test_cases"),
        )
    else:
        if not feature_meta.get("benchmark_shape_count") and count is not None:
            feature_meta["benchmark_shape_count"] = count
        if not feature_meta.get("benchmark_test_cases"):
            feature_meta["benchmark_test_cases"] = list(cases or [])
        if (
            str(feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE).lower()
            == DEFAULT_SHAPE_COVERAGE_PROFILE
        ):
            feature_meta["shape_coverage_profile"] = derive_shape_coverage_profile(
                benchmark_shape_count=feature_meta.get("benchmark_shape_count"),
                benchmark_test_cases=feature_meta.get("benchmark_test_cases"),
            )

    cases_path = payload.get("benchmark_test_cases_path")
    return str(cases_path) if cases_path else None


def _load_baseline_metrics(path: str | Path | None) -> dict[str, Any]:
    """Load baseline metrics JSON if available."""
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text())
    except (OSError, ValueError) as exc:
        logger.debug("Failed to read baseline metrics: %s", exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_generation_audit_inputs(
    *,
    kernel_path: str,
    kernel_name: str,
    kernel_type: str,
    function_names: list[str],
    input_dialect: str,
    gluon_feature_mode: str | None,
    gluon_baseline_profile: str,
    allowed_output_dialects: list[str] | None,
    preferred_output_dialects: list[str] | None,
    output_dialect_search_policy: str | None,
    target_backend: str,
    benchmark_shape_count: int | None,
    benchmark_test_cases: list[dict[str, Any]] | None,
    shape_coverage_profile: str | None,
    baseline_metrics_path: Path | None,
    previous_results_dir: Path | None,
    previous_tasks_dir: Path | None,
    current_round: int,
    num_gpus: int,
) -> tuple[list[str], int]:
    """Return Base family and Extension layer expectations for submitted tasks."""
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
    _patch_feature_meta_from_baseline_metrics(feature_meta, baseline_metrics_path)
    baseline_metrics = _load_baseline_metrics(baseline_metrics_path)
    if not baseline_metrics and not feature_meta.get("benchmark_test_cases"):
        return [], 0
    signal_text = _read_kernel_signal_text(
        kernel_path=kernel_path,
        kernel_name=kernel_name,
        function_names=function_names,
        baseline_metrics=baseline_metrics,
    )
    traits = _infer_gluon_planning_traits(feature_meta, signal_text)
    required_families = _base_triton_mandatory_families(feature_meta, traits, baseline_metrics)

    extension_slots = 0
    if feature_uses_gluon_guidance_from_meta(feature_meta):
        prior_text = ""
        if previous_results_dir and Path(previous_results_dir).is_dir():
            prior_text += _scan_previous_results(Path(previous_results_dir))
        if previous_tasks_dir and Path(previous_tasks_dir).is_dir():
            prior_text += "\n" + _scan_previous_tasks(Path(previous_tasks_dir), current_round)
        previous_signal = _previous_gluon_signal(_filter_gluon_relevant_text(prior_text))
        _, _, extension_slots = _base_extension_quotas(
            num_gpus,
            _gluon_extension_strength(traits),
            previous_signal,
            shape_profile=str(feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE).lower(),
            current_round=current_round,
            input_dialect=str(feature_meta.get("input_dialect") or ""),
            source_origin=str(feature_meta.get("source_origin") or ""),
        )
    return required_families, extension_slots


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
    baseline_cases_path = _patch_feature_meta_from_baseline_metrics(feature_meta, baseline_metrics)
    effective_benchmark_test_cases_path = benchmark_test_cases_path or baseline_cases_path or ""
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
        task_needs_gluon_docs = _task_config_requires_gluon_worker_docs(
            kernel_type=kernel_type,
            cfg=t.config,
            task_body=t.task,
            label=t.label,
        )
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
            "benchmark_test_cases_path": effective_benchmark_test_cases_path,
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
            "use_skills": _task_config_uses_skills(
                kernel_type=kernel_type,
                cfg=t.config,
                task_body=t.task,
                label=t.label,
                requires_gluon_docs=task_needs_gluon_docs,
            ),
            "allowed_skill_tiers": list(allowed_skill_tiers),
        }
        if task_knowledge_paths.get("knowledge_base_path"):
            metadata["knowledge_base_path"] = task_knowledge_paths["knowledge_base_path"]
        if str(kernel_type or "").strip().lower() == "triton":
            if t.config.get("search_set"):
                metadata["search_set"] = t.config["search_set"]
            if t.config.get("required_output_dialect"):
                metadata["required_output_dialect"] = t.config["required_output_dialect"]
            if task_needs_gluon_docs:
                metadata.update({key: value for key, value in task_knowledge_paths.items() if value})
                if t.config.get("gluon_doc_profile"):
                    metadata["gluon_doc_profile"] = t.config["gluon_doc_profile"]
                if t.config.get("required_gluon_docs"):
                    metadata["required_gluon_docs"] = list(t.config["required_gluon_docs"])
            if t.config.get("source_base_family"):
                metadata["source_base_family"] = t.config["source_base_family"]
            if t.config.get("plain_competitor"):
                metadata["plain_competitor"] = t.config["plain_competitor"]
            if t.config.get("implementation_layer"):
                metadata["implementation_layer"] = t.config["implementation_layer"]
            if t.config.get("extension_layer"):
                metadata["extension_layer"] = t.config["extension_layer"]
            if t.config.get("required_patch_target_symbols"):
                metadata["required_patch_target_symbols"] = list(t.config["required_patch_target_symbols"])
            if t.config.get("executed_route_symbols"):
                metadata["executed_route_symbols"] = list(t.config["executed_route_symbols"])
            for key, _tag in _OPTIONAL_GLUON_TASK_METADATA_FIELDS:
                if key in t.config and t.config.get(key) not in (None, ""):
                    metadata[key] = t.config[key]
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
    knowledge_base = _find_repo_relative_file(workspace, _KNOWLEDGE_BASE_REL)
    if knowledge_base:
        return knowledge_base
    for rel in _PLAIN_TRITON_KB_FALLBACK_RELS:
        candidate = resolve_project_resource(rel, workspace=workspace)
        if candidate:
            return candidate
    return None


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

    gluon_skill_path = resolve_project_resource(_GLUON_SKILL_REL, workspace=workspace) if uses_gluon_guidance else None
    gluon_kb_path = resolve_project_resource(_GLUON_KB_REL, workspace=workspace) if uses_gluon_guidance else None
    gluon_examples_path = resolve_project_resource(_GLUON_EXAMPLES_REL, workspace=workspace) if uses_gluon_guidance else None
    split_doc_paths = {
        key: path
        for key, rel in _GLUON_SPLIT_DOC_RELS.items()
        if (path := resolve_project_resource(rel, workspace=workspace))
    } if uses_gluon_guidance else {}

    resolved = {
        "knowledge_base_path": str(knowledge_base_path) if knowledge_base_path else "",
        "gluon_skill_path": str(gluon_skill_path) if gluon_skill_path else "",
        "gluon_kb_path": str(gluon_kb_path) if gluon_kb_path else "",
        "gluon_examples_path": str(gluon_examples_path) if gluon_examples_path else "",
    }
    resolved.update({key: str(path) for key, path in split_doc_paths.items()})
    return resolved


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


def _baseline_bottleneck(baseline_metrics: dict[str, Any] | None) -> str:
    """Return a normalized bottleneck string from baseline metrics."""
    if not isinstance(baseline_metrics, dict):
        return ""
    for key in ("bottleneck", "bottleneck_type", "primary_bottleneck"):
        value = baseline_metrics.get(key)
        if value:
            return str(value).strip().lower()
    top = baseline_metrics.get("top_kernels") or []
    if isinstance(top, list):
        for item in top:
            if isinstance(item, dict) and item.get("bottleneck"):
                return str(item.get("bottleneck")).strip().lower()
    return ""


def _base_triton_mandatory_families(
    feature_meta: dict[str, Any],
    traits: list[str],
    baseline_metrics: dict[str, Any] | None = None,
) -> list[str]:
    """Return Base Triton families that must be covered before Gluon can add width."""
    if str(feature_meta.get("kernel_type") or "").strip().lower() != "triton":
        return []
    if str(feature_meta.get("source_origin") or "").strip().lower() == "existing_amd_gluon_operator":
        return []

    trait_set = set(traits)
    shape_profile = str(
        feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE
    ).lower()
    bottleneck = _baseline_bottleneck(baseline_metrics)
    duration = 0.0
    if isinstance(baseline_metrics, dict):
        try:
            duration = float(baseline_metrics.get("duration_us") or baseline_metrics.get("benchmark_duration_us") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0

    matrix_like = bool(
        trait_set.intersection({"matrix_dot", "matrix_scaled_dot", "matrix_wmma_descriptor"})
    )
    shape_sensitive = shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED)
    latency_like = "latency" in bottleneck or (duration > 0 and duration <= 250.0)
    scaled_like = "matrix_scaled_dot" in trait_set
    if matrix_like or scaled_like or (shape_sensitive and latency_like):
        return [
            _BASE_FAMILY_SWIZZLE_TILE,
            _BASE_FAMILY_SPLIT_K,
            _BASE_FAMILY_SCALED_FUSION,
            _BASE_FAMILY_STREAMLINE,
            _BASE_FAMILY_PERSISTENT,
        ]

    return [
        _BASE_FAMILY_GENERIC_ALGO,
        _BASE_FAMILY_GENERIC_FUSION,
        _BASE_FAMILY_GENERIC_MEMORY,
    ]


def _render_base_family_checklist(required_families: list[str]) -> list[str]:
    """Render mandatory Base family guidance lines."""
    if not required_families:
        return []
    lines = [
        "Mandatory plain Triton family checklist:",
        "- Every plain Triton competitor task must include `Base family: <family_id>` in task_prompt for compatibility audit.",
        "- Every high-value optimization direction needs a plain Triton competitor before a Gluon overlay can count as coverage.",
        "- Fill these families before creating extra plain Triton variants:",
    ]
    lines.extend(
        f"  - `{family}`: {_BASE_FAMILY_DETAILS.get(family, 'planner-required plain Triton direction')}"
        for family in required_families
    )
    lines.append("Gluon overlay reasons by family, when a same-direction AMD Gluon branch is justified:")
    lines.extend(
        f"  - `{family}`: {_BASE_FAMILY_GLUON_OVERLAY_REASONS.get(family, 'requires a concrete measured Gluon mechanism')}"
        for family in required_families
    )
    lines.append(
        "- Autotune-only or generic memory-coalescing tasks do not satisfy a missing mandatory family unless they are attached to one of the family IDs above."
    )
    return lines


def _build_evidence_anchored_composition_guidance(has_prior_evidence: bool) -> str:
    """Render generic Base/Gluon composition rules for later-round planning."""
    lines = [
        "## Evidence-Anchored Composition",
        "- Do not treat plain Triton and AMD Gluon as a binary choice. Use prior evidence to compose around the current safe anchor.",
        "- Safe anchor selection: prefer the best verified plain Triton competitor patch; if no plain Triton patch is usable, use the best verified non-regressing patch; otherwise use the original baseline.",
        "- Portable component types: `algorithm_decomposition`, `tiling_or_blocking`, `memory_access_policy`, `layout_or_indexing`, `matrix_lowering`, `mask_or_boundary_simplification`, `accumulator_representation`, `launch_or_dispatch_policy`, `scheduler_or_persistent_policy`, `dtype_or_precision_policy`.",
        "- Usually portable components include memory/load policy, mask or boundary simplification, indexing/pointer cleanup, dtype/cast cleanup, and shape dispatch evidence.",
        "- Usually mutually exclusive components include two tile schemes, 1D vs 2D accumulator, split-K vs persistent scheduling, full Triton loop structure vs full Gluon layout rewrite, autotune-key dispatch vs manual host dispatch, and two launcher/constexpr contract changes.",
        "- Composition task types:",
        "  - `base_refine`: continue optimizing the safe anchor.",
        "  - `shared_transplant`: preserve the safe anchor algorithm and transplant one portable component from Shared/Extension evidence; output may remain plain Triton.",
        "  - `gluon_variant`: re-express the safe anchor algorithm in AMD Gluon only when the useful component requires explicit layout, AMD buffer paths, matrix lowering, or dialect-specific machinery.",
        "  - `hybrid_dispatch`: use host-side shape/feature dispatch only when per-shape or sub-operation evidence shows different winners.",
        "- Every composition task must include `Composition type: ...`, `Safe anchor: ...`, `Source component: ...`, and `Comparison target: safe_anchor` in task_prompt.",
        "- Composition candidates must compare against the safe anchor, not only the original baseline, and must reject any per-shape regression relative to that safe anchor.",
    ]
    if has_prior_evidence:
        lines.append(
            "- Prior-round evidence is available: generate at least one `shared_transplant` or `gluon_variant` task that preserves the safe anchor and applies one portable component from Shared/Extension evidence when such a component is present."
        )
    else:
        lines.append(
            "- No prior-round evidence yet: do not force composition in round 1; generate Base anchors and Gluon/Shared evidence so later rounds can compose safely."
        )
    return "\n".join(lines)


def _parse_tagged_value(text: str, tag: str) -> str | None:
    match = re.search(rf"{re.escape(tag)}\s*:\s*`?([A-Za-z0-9_/-]+)`?", text, re.IGNORECASE)
    return match.group(1).strip().lower() if match else None


def _strip_enclosing_backticks(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1].strip()
    return value


def _parse_prompt_field_value(text: str, field: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = _strip_enclosing_backticks(match.group(1))
    return value or None


def _parse_prompt_key_value(text: str, key: str) -> str | None:
    """Parse exact snake_case/kebab-case task metadata from task_prompt lines."""
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = _strip_enclosing_backticks(match.group(1))
    return value or None


def _normalize_optional_task_metadata_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value or "").strip().strip("`")
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    return text


def _optional_task_metadata_from_item_or_prompt(
    item: dict[str, Any],
    task_prompt: str,
    key: str,
    tag: str,
) -> Any:
    if key in item:
        return _normalize_optional_task_metadata_value(item.get(key))
    tagged = _parse_prompt_field_value(task_prompt, tag)
    if tagged in (None, ""):
        tagged = _parse_prompt_key_value(task_prompt, key)
    return _normalize_optional_task_metadata_value(tagged)


def _is_shared_like_task_text(label: str, task_prompt: str) -> bool:
    text = f"{label}\n{task_prompt}".lower()
    return "shared set" in text or label.lower().startswith(("shared-", "compose-", "composition-"))


def _is_extension_like_task_text(label: str, task_prompt: str) -> bool:
    text = f"{label}\n{task_prompt}".lower()
    return (
        "extension set" in text
        or "extension layer:" in text
        or label.lower().startswith(("ext-", "extension-"))
    )


def _normalize_search_output_contract(
    search_set: str,
    required_output: str,
    *,
    hybrid_like: bool,
    extension_like: bool,
) -> tuple[str, str]:
    """Derive legacy search-set metadata from the output/layer contract.

    ``required_output_dialect`` and the prompt's implementation/extension layer
    are the source of truth. ``search_set`` is kept as a compatibility bucket for
    older dispatch/reporting paths.
    """
    if hybrid_like or required_output == "mixed":
        return "extension", "mixed"
    if extension_like and required_output in {"", "plain_triton", "any"}:
        return "extension", "amd_gluon"
    if search_set == "base" and required_output in {"amd_gluon"}:
        return "extension", required_output
    if required_output == "amd_gluon":
        return "extension", required_output
    if required_output == "plain_triton" and not extension_like:
        return "base", required_output
    return search_set, required_output


def _infer_required_patch_target_symbols(task_prompt: str, item: dict[str, Any] | None = None) -> list[str]:
    item = item or {}
    raw = item.get("required_patch_target_symbols")
    if isinstance(raw, str):
        symbols = [part.strip().strip("`") for part in raw.split(",")]
    elif isinstance(raw, list):
        symbols = [str(part).strip().strip("`") for part in raw]
    else:
        symbols = []
    for key in ("target_symbol", "target_component"):
        item_value = item.get(key)
        if isinstance(item_value, str) and item_value.strip():
            symbols.extend(part.strip().strip("`") for part in item_value.split(","))
            symbols.extend(target_symbols_from_scoped_text(item_value))

    for match in re.finditer(
        r"^\s*(?:Target symbol|Target component|Required patch target symbols?|required_patch_target_symbols)\s*:\s*(.+?)\s*$",
        task_prompt,
        re.IGNORECASE | re.MULTILINE,
    ):
        value = match.group(1)
        symbols.extend(part.strip().strip("`") for part in value.split(","))
        symbols.extend(target_symbols_from_scoped_text(value))
    for field in ("Allowed change", "Target component"):
        tagged = _parse_prompt_field_value(task_prompt, field)
        if tagged:
            symbols.extend(target_symbols_from_scoped_text(tagged))
    if not filter_abstract_target_symbols(
        [symbol for symbol in symbols if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol)]
    ):
        symbols.extend(target_symbols_from_actionable_prose(task_prompt))

    unique: list[str] = []
    for symbol in symbols:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol) and symbol not in unique:
            unique.append(symbol)
    target_component = str(
        item.get("target_component")
        or _parse_prompt_field_value(task_prompt, "Target component")
        or ""
    )
    patch_targets, _route_symbols = split_patch_and_route_symbols(unique, target_component=target_component)
    return patch_targets


def _infer_executed_route_symbols(
    task_prompt: str,
    item: dict[str, Any] | None = None,
    *,
    required_patch_target_symbols: list[str] | None = None,
) -> list[str]:
    item = item or {}
    raw = item.get("executed_route_symbols")
    if isinstance(raw, str):
        symbols = [part.strip().strip("`") for part in raw.split(",")]
    elif isinstance(raw, list):
        symbols = [str(part).strip().strip("`") for part in raw]
    else:
        symbols = []
    for match in re.finditer(
        r"^\s*(?:Executed route symbols?|executed_route_symbols|Target symbol|required_patch_target_symbols|Required patch target symbols?)\s*:\s*(.+?)\s*$",
        task_prompt,
        re.IGNORECASE | re.MULTILINE,
    ):
        value = match.group(1)
        symbols.extend(part.strip().strip("`") for part in value.split(","))
        symbols.extend(target_symbols_from_scoped_text(value))
    target_component = str(
        item.get("target_component")
        or _parse_prompt_field_value(task_prompt, "Target component")
        or ""
    )
    _patch_targets, route_symbols = split_patch_and_route_symbols(
        [*symbols, *(required_patch_target_symbols or [])],
        target_component=target_component,
    )
    unique: list[str] = []
    for symbol in route_symbols:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol) and symbol not in unique:
            unique.append(symbol)
    return unique


def _infer_search_set_and_required_output(label: str, task_prompt: str, item: dict[str, Any] | None = None) -> tuple[str, str]:
    """Infer output contract plus legacy search metadata for task frontmatter."""
    item = item or {}
    text = f"{label}\n{task_prompt}".lower()
    hybrid_like = "extension layer: hybrid" in text or "hybrid_dispatch" in text or "hybrid dispatch" in text
    shared_like = _is_shared_like_task_text(label, task_prompt)
    extension_like = _is_extension_like_task_text(label, task_prompt)
    search_set = str(item.get("search_set") or "").strip().lower()
    required_output = str(item.get("required_output_dialect") or "").strip().lower()

    tagged_search = _parse_tagged_value(task_prompt, "Search set")
    tagged_output = _parse_tagged_value(task_prompt, "Required output dialect")
    if not tagged_search:
        tagged_search = _parse_prompt_key_value(task_prompt, "search_set")
    if not tagged_output:
        tagged_output = _parse_prompt_key_value(task_prompt, "required_output_dialect")
    if tagged_search:
        search_set = tagged_search
    if tagged_output:
        required_output = tagged_output

    if search_set not in _VALID_SEARCH_SETS:
        if shared_like:
            search_set = "shared"
        elif extension_like:
            search_set = "extension"
        else:
            search_set = "base"

    if required_output not in _VALID_REQUIRED_OUTPUT_DIALECTS:
        if hybrid_like:
            required_output = "mixed"
        elif search_set == "base":
            required_output = "plain_triton"
        elif search_set == "extension":
            required_output = "amd_gluon"
        else:
            required_output = "any"

    search_set, required_output = _normalize_search_output_contract(
        search_set,
        required_output,
        hybrid_like=hybrid_like,
        extension_like=extension_like,
    )
    return search_set, required_output


def _infer_gluon_doc_profile(label: str, task_prompt: str, search_set: str, required_output: str) -> str:
    text = f"{label}\n{task_prompt}\n{search_set}\n{required_output}".lower()
    if "hybrid_dispatch" in text or "extension layer: hybrid" in text or required_output == "mixed":
        if "evidence" in text or "safe anchor" in text or "per-shape" in text or "sub-operation" in text:
            return "hybrid_dispatch_from_evidence"
        return "hybrid_dispatch"
    if "gluon_variant" in text or "gluon variant" in text:
        return "gluon_variant_from_anchor"
    if "shared_transplant" in text or search_set == "shared":
        return "shared_transplant"
    if "nv_gluon" in text or "nvidia" in text or "translation" in text:
        return "nv_to_amd_translation"
    if any(marker in text for marker in ("shape_bucketed", "shape_dispatch", "bucketed")):
        return "shape_bucketed_dispatch"
    if any(marker in text for marker in ("jit", "aot", "prebuilt", "compile_gluon", "signature", "waves_per_eu", "scratch")):
        return "jit_aot_sensitive"
    if any(marker in text for marker in ("buffer", "buffer_load", "buffer_store", "memory", "load", "store", "kv", "cache")):
        return "memory_lowering"
    if any(marker in text for marker in ("matrix", "dot", "mfma", "wmma", "scaled", "gemm", "fp8", "fp4")):
        return "matrix_lowering"
    if search_set == "extension" and required_output == "amd_gluon":
        return "extension_l0_minimal"
    return "base_or_shared_gluon"


def _infer_required_gluon_doc_keys(
    label: str,
    task_prompt: str,
    search_set: str,
    required_output: str,
    profile: str,
) -> list[str]:
    text = f"{label}\n{task_prompt}\n{search_set}\n{required_output}\n{profile}".lower()
    docs = required_doc_keys_for_profile(profile)

    def add(key: str) -> None:
        add_unique_doc_key(docs, key)

    if any(
        marker in text
        for marker in (
            "gluon",
            "amd_gluon",
            "nv_gluon",
            "extension",
            "shared",
            "matrix",
            "memory",
            "layout",
            "blockedlayout",
            "slicelayout",
            "dotoperandlayout",
            "convert_layout",
            "amdmfmalayout",
            "amdwmmalayout",
            "threads_per_warp",
            "warps_per_cta",
            "wave64",
            "wave32",
        )
    ):
        add("gluon_component_traits_path")
    if (
        required_output == "amd_gluon"
        or profile
        in {
            "extension_l0_minimal",
            "memory_lowering",
            "matrix_lowering",
            "jit_aot_sensitive",
            "gluon_variant_from_anchor",
        }
        or any(
            marker in text
            for marker in (
                "convert_layout",
                "dotoperandlayout",
                "amdmfmalayout",
                "amdwmmalayout",
                "buffer_load",
                "buffer_store",
                "mfma_scaled",
                "wmma_scaled",
                "get_mfma_scale_layout",
                "get_wmma_scale_layout",
                "k_width",
            )
        )
    ):
        add("gluon_api_reference_path")
    if (
        profile
        in {
            "nv_to_amd_translation",
            "matrix_lowering",
            "jit_aot_sensitive",
            "gluon_variant_from_anchor",
            "hybrid_dispatch",
            "hybrid_dispatch_from_evidence",
        }
        or any(
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
                "signature",
                "waves_per_eu",
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
                "tdm",
                "current_target",
                "translator",
            )
        )
    ):
        add("gluon_architecture_notes_path")
    if (
        profile
        in {
            "nv_to_amd_translation",
            "shape_bucketed_dispatch",
            "shared_transplant",
            "gluon_variant_from_anchor",
            "hybrid_dispatch",
            "hybrid_dispatch_from_evidence",
        }
        or any(
            marker in text
            for marker in (
                "aiter",
                "attention",
                "decode",
                "gemm",
                "kv",
                "cache",
                "mqa",
                "blockscale",
                "afp4",
                "wfp4",
                "fp8",
                "fp4",
                "preshuffle",
                "benchmark",
                "per-shape",
                "shape regression",
                "source-first",
                "translator",
                "current_target",
                "artifact",
                "zip",
                "json",
                "config",
                "env",
            )
        )
    ):
        add("gluon_real_patterns_path")
    if "example" in text:
        add("gluon_examples_doc_path")
    return docs


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
_GLUON_STRONG_WIN_SPEEDUP = 1.02


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
       / ``speedup=N`` numbers. Only speedups above the strong-win threshold
       return ``won``. Noise-level wins just above parity are ``attempted`` so
       they do not expand the next-round L1/Shared/Hybrid Gluon budget.
       ``speedup < 1.0`` (including ``0.0``) returns ``slower``.
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

    has_strong_speedup = False
    has_weak_positive_speedup = False
    has_speedup_below_one = False
    has_zero_speedup = False
    for match in _PREVIOUS_GLUON_SPEEDUP_RE.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value > _GLUON_STRONG_WIN_SPEEDUP:
            has_strong_speedup = True
        elif value > 1.0:
            has_weak_positive_speedup = True
        elif value == 0.0:
            has_zero_speedup = True
        elif 0 < value < 1.0:
            has_speedup_below_one = True

    if has_strong_speedup:
        return "won"

    slower_markers = ("slower", "performance regression", "regressed", "worse")
    if has_speedup_below_one or has_zero_speedup or any(
        marker in text for marker in slower_markers
    ):
        return "slower"
    if has_weak_positive_speedup:
        return "attempted"

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
    *,
    current_round: int = 1,
    input_dialect: str = "plain_triton",
    source_origin: str = "",
) -> tuple[int, int, int]:
    """Return recommended optimization-direction / dialect-overlay counts.

    ``num_gpus`` is the parallel execution budget, not the total number of
    candidates the planner may create. The pool dispatcher queues overflow tasks.
    Base/Shared/Extension remain metadata, but task allocation is direction-first:
    choose Triton optimization directions, then add a Gluon overlay only when
    a concrete dialect mechanism exists.

    Extension slots are optional AMD Gluon overlays. Round 1 for plain/nv Triton
    inputs gets at most one L0 overlay, and only with strong trait evidence.
    Additional slots are later-round trait-specific refinements and require
    prior benchmark evidence. Shared slots are paired same-direction mappings,
    not generic Gluon rewrites.
    """
    budget = max(int(num_gpus or 1), 1)
    input_dialect = str(input_dialect or "").strip().lower()
    source_origin = str(source_origin or "").strip().lower()
    existing_amd_gluon = (
        input_dialect == "amd_gluon"
        and source_origin == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
    )
    if budget <= 1:
        extension = 0 if existing_amd_gluon else 1 if strength == "strong" or input_dialect in {"amd_gluon", "nv_gluon"} else 0
        return 2, 0, extension

    if budget <= 2:
        base = 2
    elif budget <= 5:
        base = budget
    else:
        base = min(budget, 7)
    shared = 0
    extension = 0

    if shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        shared = max(shared, 1)
    if budget >= 4:
        shared = max(shared, 1)
    if budget >= 8 or shape_profile == SHAPE_COVERAGE_BUCKETED:
        shared = max(shared, 2)

    if existing_amd_gluon:
        base = min(max(1, budget - 1), 3)
        shared = 1 if budget >= 4 or shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED) else 0
        if budget >= 8 or shape_profile == SHAPE_COVERAGE_BUCKETED:
            shared = max(shared, 2)
        extension = 0
    elif input_dialect == "amd_gluon":
        extension = 1
    elif previous_signal == "won":
        extension = max(extension, 2)
        if budget >= 8 or shape_profile == SHAPE_COVERAGE_BUCKETED:
            extension = max(extension, 3)
    elif previous_signal in ("failed", "slower", "attempted"):
        extension = 1 if strength == "strong" else 0
        if previous_signal == "failed":
            shared = min(shared, 1)
    else:
        if strength == "strong" and budget >= 4:
            extension = max(extension, 1)
        if shape_profile == SHAPE_COVERAGE_BUCKETED and budget >= 5:
            extension = max(extension, 1 if strength in {"normal", "strong"} else 0)

    if int(current_round or 1) <= 1 and input_dialect != "amd_gluon":
        # Round 1 has no verified AMD Gluon anchor for plain/nv Triton inputs.
        # Keep Extension work optional and capped at one L0 overlay.
        extension = min(extension, 1)

    shared = min(shared, 2)
    extension = min(extension, 0 if existing_amd_gluon else 3)
    return base, shared, extension


def _dialect_interleave_strategy(
    *,
    num_gpus: int,
    base_slots: int,
    shared_slots: int,
    extension_slots: int,
    previous_signal: str,
    shape_profile: str,
    existing_amd_gluon: bool = False,
) -> str:
    """Render the scheduling strategy for the Base/Shared/Extension portfolio."""
    budget = max(int(num_gpus or 1), 1)
    total = base_slots + shared_slots + extension_slots
    if existing_amd_gluon:
        if budget <= 1:
            return "\n".join(
                [
                    "Scheduling mode: `serial_existing_amd_gluon_refinement`",
                    "- The measured baseline already executes AMD Gluon. Queue narrow in-dialect refinements sequentially.",
                    "- Plain Triton tasks are optional comparison/fallback work, not mandatory same-batch competitors for every Gluon component.",
                    f"- Candidate count: {total} task(s) for {budget} GPU. This is allowed only because the mode is serial interleave.",
                ]
            )
        return "\n".join(
            [
                "Scheduling mode: `parallel_existing_amd_gluon_refinement`",
                f"- {budget} GPU(s) are available; generate a component-first portfolio ({total} task(s) recommended).",
                "- This is a Gluon-first portfolio: prefer fewer tasks over filling unused GPU slots with generic plain Triton work.",
                "- Pick strategy directions from the standard Triton taxonomy, then map each dispatchable task to one AMD Gluon atomic component.",
                "- Keep optional plain/shared comparisons only for fallback evidence, portable components, mixed dispatch, or source subkernels that are still plain Triton; they must be typed as `plain_subkernel_refine` or `shared_or_plain_comparison`.",
                f"- Shape profile for scheduling: `{shape_profile}`.",
            ]
        )
    if budget <= 1:
        extension_line = (
            "- Add one L0 AMD Gluon overlay after the plain Triton directions only if the task names a concrete Gluon overlay reason."
            if extension_slots
            else "- No AMD Gluon overlay is required for this serial batch; keep the width on plain Triton directions."
        )
        return "\n".join(
            [
                "Scheduling mode: `serial_interleave`",
                "- One GPU is available, but planner should still plan multiple optimization directions; dispatch queues them sequentially.",
                "- Dispatch will queue all tasks and run them sequentially on the single GPU; this is intentional low-cost triage, not over-allocation.",
                "- Recommended order: run plain Triton competitors first, then any same-direction Gluon overlay, then compare by the same correctness and benchmark contract.",
                extension_line,
                f"- Candidate count: {total} task(s) for {budget} GPU. This is allowed only because the mode is serial interleave.",
            ]
        )

    if previous_signal == "won":
        followup = (
            "- Because a Gluon-related prior round appears promising, include at least one Base or Shared competitor "
            "and one follow-up Gluon/mixed refinement so the final patch can combine the winning pieces without regressing Triton buckets."
        )
    elif previous_signal == "failed":
        followup = (
            "- Because prior Gluon work appears failed, keep the Extension task narrow (layout-only, translation-only, or memory-only) "
            "while Base/Shared tasks continue the main Triton search."
        )
    elif previous_signal == "slower":
        followup = (
            "- Because prior Gluon work appears slower, keep the Extension task targeted at memory/matrix lowering and require a Base/Shared competitor."
        )
    else:
        followup = (
            "- No decisive Gluon prior signal yet; prioritize distinct optimization directions and add Gluon only as a same-direction overlay when a concrete mechanism is named."
        )

    return "\n".join(
        [
            "Scheduling mode: `parallel_mixed_portfolio`",
            f"- {budget} GPU(s) are available; generate a direction-first portfolio ({total} task(s) recommended). The GPU pool queues overflow tasks instead of trimming the search.",
            "- Choose optimization directions first. For each high-value direction, keep a plain Triton competitor; add Shared or AMD-Gluon overlays only as same-direction implementation choices.",
            "- Later rounds may create mixed/hybrid strategies that keep the best plain-Triton path for shapes or operations where it wins and use AMD Gluon only where it passes correctness and beats the Base competitor; mixed patches must preserve per-shape no-regression.",
            "- Mixed/hybrid patches must preserve the harness contract and must not drop the plain Triton no-regression path unless benchmark evidence across all shapes supports it.",
            f"- Shape profile for scheduling: `{shape_profile}`.",
            followup,
        ]
    )


def _existing_amd_gluon_planning_recommendation(
    *,
    traits: list[str],
    num_gpus: int,
    shape_profile: str,
    previous_signal: str,
) -> list[str]:
    """Render non-persistent planning guidance for production AMD Gluon inputs."""
    trait_set = set(traits)
    recommendation = [
        "Existing AMD Gluon lightweight planning recommendation:",
        "- This recommendation is prompt guidance only; do not add new task metadata such as `gluon_depth`, `complexity_level`, or `recommended_slots`.",
    ]
    if "layout_source_first_required" in trait_set or any(
        trait in trait_set for trait in ("execution_jit_aot_sensitive", "version_sensitive", "operator_support_sensitive")
    ):
        recommendation.append(
            "- Source-first / JIT-AOT-heavy signal: prioritize in-dialect refinement of the measured production path; use plain/shared only for fallback, plain subkernels, portable comparison, or mixed evidence."
        )
    if any(trait in trait_set for trait in ("matrix_dot", "matrix_scaled_dot", "matrix_wmma_descriptor")):
        recommendation.append(
            "- Matrix-like signal: pick at most one matrix/layout component in round 1 unless prior evidence proves a composed path; put softmax/reduction/wrapper dependencies in blockers."
        )
    if shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        recommendation.append(
            "- Multi-shape signal: prefer shape-robust refinements and keep optional plain/shared comparison for no-regression evidence; every refinement rejects per-shape regression."
        )
    if previous_signal in {"failed", "slower", "attempted"}:
        recommendation.append(
            "- Prior Gluon evidence is weak or regressing: narrow the next refinement to one failure layer or one removable overhead instead of expanding search width."
        )
    if max(int(num_gpus or 1), 1) <= 2:
        recommendation.append("- Small execution budget: start with one narrow in-dialect refinement before optional plain/shared comparison.")
    else:
        recommendation.append("- Larger GPU budget does not imply maximum width: keep 1-3 in-dialect refinements and 0-2 optional plain/shared comparisons.")
    recommendation.append(
        "- High-coupling combinations across wrapper dispatch, matrix layout, softmax/reduction state, partition policy, or scheduler policy should be `Task type: defer_composition`."
    )
    return recommendation


def _build_search_space_allocation_guidance(
    feature_meta: dict[str, Any],
    *,
    traits: list[str],
    num_gpus: int,
    baseline_metrics: dict[str, Any] | None = None,
    previous_results_text: str = "",
    previous_tasks_text: str = "",
    current_round: int = 1,
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
    input_dialect = str(feature_meta.get("input_dialect") or "").strip().lower()
    source_origin = str(feature_meta.get("source_origin") or "").strip().lower()
    existing_amd_gluon = (
        input_dialect == "amd_gluon"
        and source_origin == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
    )
    base_slots, shared_slots, extension_slots = _base_extension_quotas(
        num_gpus,
        strength,
        previous_signal,
        shape_profile=shape_profile,
        current_round=current_round,
        input_dialect=input_dialect,
        source_origin=source_origin,
    )
    required_base_families = _base_triton_mandatory_families(
        feature_meta,
        traits,
        baseline_metrics,
    )
    base_slots = max(base_slots, len(required_base_families))
    interleave_strategy = _dialect_interleave_strategy(
        num_gpus=num_gpus,
        base_slots=base_slots,
        shared_slots=shared_slots,
        extension_slots=extension_slots,
        previous_signal=previous_signal,
        shape_profile=shape_profile,
        existing_amd_gluon=existing_amd_gluon,
    )
    if existing_amd_gluon:
        extension_allocation = (
            "- Existing AMD Gluon in-dialect refinements: 2-3 narrow task(s) recommended for round 1 when distinct atomic components are viable; 1 high-confidence refinement is acceptable if no second safe component exists. "
            "These are not generated overlays and do not require `Plain competitor`; each must name one primary atomic component and compare against `true_baseline` or `safe_anchor`."
        )
    elif extension_slots <= 0:
        extension_allocation = (
            "- AMD Gluon overlay: 0 task(s) recommended. Do not create a Gluon task unless the prompt names a concrete same-direction Gluon overlay reason and a plain Triton competitor."
        )
    elif extension_slots == 1:
        extension_allocation = (
            f"- AMD Gluon overlay: {extension_slots} task(s). This slot is L0: "
            "minimal AMD Gluon overlay for one named optimization direction, with recognizable launcher, explicit layout, and the same correctness/benchmark contract. "
            "Include `Extension layer: L0` in task_prompt."
        )
    else:
        extension_allocation = (
            f"- AMD Gluon overlay: {extension_slots} task(s). Slot 1 is L0 minimal AMD Gluon viability; "
            "slots 2+ may be L1 trait-specific lowering only when traits or prior evidence justify it. "
            "Include `Extension layer: L0|L1|Hybrid` in task_prompt."
        )

    lines = [
        "## Search Space Allocation",
        (
            "- Existing AMD Gluon input: use Triton optimization families as performance taxonomy, then generate in-dialect component refinements."
            if existing_amd_gluon
            else "- Treat AMD Gluon as an implementation layer for a Triton optimization direction, not as a separate optimization direction or a replacement."
        ),
        f"- GPU/task budget: {max(int(num_gpus or 1), 1)}",
        f"- Gluon extension strength: {strength}",
        f"- Previous Gluon signal: {previous_signal}",
        f"- Shape coverage profile: {shape_profile}",
        interleave_strategy,
        "Recommended candidate allocation:",
        (
            f"- AMD Gluon in-dialect refinements: {base_slots} task(s) across distinct atomic components. Use `Task type: amd_gluon_in_dialect_refine|amd_gluon_layout_or_matrix_refine|amd_gluon_shape_dispatch_refine`."
            if existing_amd_gluon
            else f"- Plain Triton competitors: at least {base_slots} task(s) across distinct Triton optimization directions. Include `Base family: <family_id>` for compatibility audit."
        ),
        (
            f"- Optional plain/shared comparisons: at most {shared_slots} task(s) for fallback evidence, portable component transplant, mixed/hybrid dispatch, or source subkernels that are still plain Triton. "
            "Use explicit `Task type: plain_subkernel_refine` or `Task type: shared_or_plain_comparison`; do not emit generic plain/Base tasks to fill budget."
            if existing_amd_gluon
            else f"- Paired same-direction mappings: {shared_slots} task(s) when budget allows. Use `plain_triton variant`, `amd_gluon variant`, or paired comparison; include `Shared source family: <base_family_id>` when mapping a plain Triton strategy."
        ),
        extension_allocation,
        "Rules:",
        "- `required_output_dialect` and `Implementation layer` are the output contract; `search_set` is optional compatibility metadata and must not drive planning.",
        "- Keep `required_output_dialect` separate from the Gluon-internal API policy: legal `tl.range` / `tl.constexpr` inside an executed Gluon path does not make the output `mixed`.",
        "- Set `source_origin` when known: `generated_overlay` for new plain->Gluon overlays, `existing_amd_gluon_operator` only when the measured baseline already executes a production AMD Gluon operator, and `nv_gluon_translation` for NVIDIA-facing Gluon translation.",
        "- Generated overlays default to `gluon_tl_policy: strict_generated` and `layout_construction_policy: host_preferred`; production Gluon source may use `production_source_allowed` and `source_preserve` only when source evidence supports it.",
        (
            "- Existing AMD Gluon refinements compare to `true_baseline` or `safe_anchor`; keep plain Triton only for fallback, portable comparison, mixed dispatch, or plain source subkernels."
            if existing_amd_gluon
            else "- Keep a plain Triton competitor for every high-value direction unless AMD Gluon is explicitly required. Layering order: plain Triton -> optional L0/paired mapping -> L1 from viable/local-win evidence -> later Hybrid/Mixed."
        ),
        (
            "- Plain/shared tasks in this path are optional no-regression evidence, not synthetic anchors for existing Gluon refinements. If only one reliable Gluon refinement exists, emit fewer total tasks rather than padding the batch with untyped plain/Base tasks."
            if existing_amd_gluon
            else "- Base/plain Triton tasks are no-regression performance candidates, not synthetic Gluon anchors. Emit a Gluon L0 overlay only when the same direction has a plausible plain competitor and a concrete Gluon mechanism."
        ),
        "- If a Gluon L0 target has a clear local component, name it with `Target component:` or scoped `Allowed change:`. If the Base anchor is too broad or the component is unclear, keep the Base task and omit or downgrade the overlay instead of forcing a bookkeeping anchor.",
        "- Round 1 plain Triton input may have at most one L0 overlay. Generate it only when `overlay_priority_routing` is Prefer/high-confidence Consider; otherwise spend the slot on another plain Triton direction.",
        "- Before emitting any L0 overlay or later task derived from L0 evidence, perform `Gluon L0 scope classification`: is the candidate layout-heavy, does it include loop-carried state, online reductions, dot/matrix paths, multiple 2D parents, nested layouts, wrapper reroute, cross-stage ABI changes, boundary-specific branches, or other high-coupling logic; can one exact subpath execute without translating the whole algorithm; and is the expected outcome `execution_anchor` or `performance_candidate`?",
        "- L0 scope ladder is generic across kernels: prefer the smallest executable, attributable, low-coupling subpath (scalar/1D stage, one load/store, one index/mask layout smoke path). Consider matrix/MFMA subpaths only after an executed anchor or when the task explicitly proves the matrix subpath is the smallest viable component.",
        "- Round 1 L0 should prefer generic micro-anchor archetypes, not fixed task labels: layout/broadcast micro-anchor, index/mask layout anchor, load/store layout anchor, matrix operand layout probe, reduction/accumulator layout probe, or wrapper/integration anchor. Name the actual task from the optimization direction and target component.",
        "- Classify candidate signals by failure layer rather than operator name: attention-like composite path, matrix-like composite path, broadcast-heavy layout path, reduction/accumulator path, conditional/source-first path, or wrapper/integration boundary.",
        "- Route worker docs by `gluon_doc_profile` plus obvious task signals. Use `task_signals`, `routed_doc_reasons`, `kernel_family_signal`, and `failure_layers` when they are clear; they are routing aids, not a reason to invent a larger task.",
        "- AMD Gluon overlay prompts should include the core layer/direction/comparison fields from `10_search_policies.md`; do not expand API recipes or full failure maps in the task body.",
        "- For L0, state the executable boundary when you can infer it. Missing advisory routing fields can be inferred later, but do not emit a required AMD Gluon task if the scope is explicitly infeasible.",
        "- L0 execution path choices are: `inline_scoped_helper` for a local executable component, `separate_gluon_kernel` when a second launch/temp buffer is accepted as evidence, `whole_jit_kernel` only when the whole helper/kernel is truly the minimum executable unit, or no required Gluon task when the scope is infeasible.",
        "- `inline_scoped_helper` is invalid for high-coupling L0 targets such as online softmax accumulator loops, `tl.dot`/matrix paths, loop-carried reductions, cross-stage ABI changes, or wrapper reroutes. Shrink to a lower-coupling index/mask/load/store smoke path, use a justified `whole_jit_kernel`, or do not emit the Gluon task.",
        "- `whole_jit_kernel` is a compile-risk anchor, not the default L0 shape. If used, the task must include `expected_failure_layers`, `first_patch_compile_goal`, `do_not_optimize_before_compile: true`, and `matrix_lowering_required: true|false`. `patch_0` should aim for a compile anchor before performance tuning.",
        "- Before submitting a Gluon task, run a concise consistency pass: keep `Optimization direction` as the shared performance goal, pick one primary component for `patch_0`, and put Gluon-specific mechanisms in overlay fields.",
        "- Final L0 overlay pair check: bind to a same-batch plain competitor when one exists at the same direction/component. If the anchor is too broad or unclear, omit the overlay rather than failing the whole plan.",
        "- First-pass L0 branch rule: emit exactly one shape. If the target is local or ambiguous, default to a local single-component smoke/probe with `inline_scoped_helper`; use `whole_jit_kernel` only after retargeting both the target wording and Base/Gluon pair to a whole-helper skeleton with a concrete `whole_kernel_required_reason`.",
        "- Branch A signal partition: `task_signals` should name only the primary patch target component. Put matrix/reduction/wrapper/dot/dispatch context in `Secondary components / blockers`, `failure_layers`, or `expected_failure_layers`, not in the positive patch-target signals.",
        "- Required Gluon worker contract: ask for `Gluon knowledge lookup plan`, `Gluon implementation plan`, `Performance hypothesis:`, `Same ABI comparison:`, and `Patch evolution:` before editing. Keep API-level rewrite and pass/fail patch details in the routed skills/docs, not in the prompt body.",
        "- For an L0 `extension_intent=execution_anchor`, keep `Performance hypothesis:` to execution and attribution: verify the explicit layout/scoped helper can execute and record layout construction or conversion overhead. Do not claim MFMA utilization or broad throughput improvement unless the task is matrix-lowering/L1 or prior evidence supports that mechanism.",
        "- L0 overlay prompts must not ask the worker to convert an entire stage/helper/kernel just to prove Gluon. Scope L0 to one named subpath/component, such as a load/store, layout, mask, or matrix subpath; only use a whole helper as the target when the task explicitly explains why the helper is the smallest viable component.",
        "- `local target + whole_jit_kernel` is invalid unless the task is retargeted as a whole-helper skeleton. Unknown scope should become a local smoke/probe or Base/plain task, not a default whole-kernel rewrite.",
        "- Same-family Base tasks are not enough for a local L0 binding when the component differs. If you cannot produce a credible same-direction plain Base competitor, spend the slot on Base/plain Triton instead of emitting the overlay.",
        "- L1 tasks additionally require an executed Gluon/mixed anchor (`Anchor patch`, `Anchor speedup`, `Anchor execution: true`, `Comparison target: anchor_patch`) and stage-specific tasks require target metadata such as `Target symbol` / `Target component` plus top-level `required_patch_target_symbols` when available.",
        "- If a plain Triton candidate wins, accept it as the best result rather than forcing more Gluon work.",
        "- If a Triton strategy wins and maps cleanly to Gluon traits, a later round may create an AMD Gluon variant of that winning strategy only with a concrete performance hypothesis.",
        "- If an AMD Gluon candidate wins on only some shapes or sub-operations, a later round may create a `mixed/hybrid` candidate that dispatches between plain Triton and AMD Gluon by host-side shape/feature checks; the mixed path must keep per-shape no-regression and must be compared against a plain Triton or paired competitor.",
        "- Round progression: if prior Gluon compile failed, did not execute, or reported scope escalation, do not emit L1, `gluon_variant`, or `hybrid_dispatch`; shrink/retry L0 or spend the slot on Base/plain Triton. If prior Gluon passed but was slower, emit only a narrow overhead refinement tied to the recorded `overhead_source_to_record`. Only prior per-shape or sub-operation Gluon wins justify `gluon_variant` or `hybrid_dispatch`.",
    ]
    if existing_amd_gluon:
        lines.extend(
            [
                "Existing AMD Gluon refinement decomposition:",
                "- Pipeline: `kernel_family_signal -> atomic_component_graph -> coupling/execution-boundary decision -> task_type -> doc_profile + failure_layers`.",
                "- Kernel family signals are routing hints: `attention_decode_kv_cache`, `logits_or_small_reduction`, `gemm_or_scaled_dot`, `memory_or_elementwise`, `wrapper_shape_dispatch`, or `descriptor_or_source_first`.",
                "- Atomic components are dispatchable task scopes: `wrapper_shape_dispatch`, `layout_parent_slice`, `load_store_buffer`, `matrix_operand_mfma`, `reduction_accumulator`, `state_update_softmax`, `epilogue_output_store`, `scheduler_launch_runtime`, or `source_contract_integration`.",
                "- Low-coupling single-component tasks may dispatch in round 1; medium-coupling tasks must state measurement boundary, ABI risk, and rollback condition; high-coupling multi-component combinations become `Task type: defer_composition`.",
                "- Existing refinement task contract: `source_origin: existing_amd_gluon_operator`, `required_output_dialect: amd_gluon`, `Implementation layer: amd_gluon in-dialect refinement`, `Target component: ...`, `Allowed change: ...`, `Comparison target: true_baseline|safe_anchor`, `Reject if: correctness fails or any benchmark shape regresses`.",
                "- Preserve source contracts: public ABI, measured wrapper route, JIT/AOT guards, target guards, fallback/artifact selection, layout declarations, and output feeding path.",
                "- If the target symbol is a plain `@triton.jit` subkernel, emit `Task type: plain_subkernel_refine` with `required_output_dialect: plain_triton`; do not force it into AMD Gluon output.",
                "- Patch iteration: after pass+regression try at most one named overhead cleanup; after compile/layout failure fix only that failure layer; when multiple components are required, record `compose_later:<components>` instead of widening the current task.",
            ]
        )
        lines.extend(
            _existing_amd_gluon_planning_recommendation(
                traits=traits,
                num_gpus=num_gpus,
                shape_profile=shape_profile,
                previous_signal=previous_signal,
            )
        )
    lines.extend(_render_base_family_checklist(required_base_families))
    if str(feature_meta.get("source_origin") or "").strip().lower() == "existing_amd_gluon_operator":
        lines.append(
            "- Existing AMD Gluon production operator anchors (not mandatory Base audit): "
            f"`{_BASE_FAMILY_CONFIG_DISPATCH}`, `{_BASE_FAMILY_PIPELINE_STAGE}`, "
            f"`{_BASE_FAMILY_AOT_JIT}`, `{_BASE_FAMILY_PRESHUFFLE_DESCRIPTOR}`, "
            f"`{_BASE_FAMILY_SPLITK_REDUCE_PIPELINE}`. Use these as comparison/attribution families "
            "when the measured production operator path exposes config, pipeline, AOT/JIT, descriptor, or split-reduce boundaries."
        )
    if shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        lines.append(
            "- Multi-shape rule: at least one plain Triton competitor candidate MUST be `shape_robust`; "
            "Gluon overlay candidates MUST self-classify as `shape_robust` or `shape_bucketed` "
            "(not `single_shape_viability` after round 1)."
        )
    if shape_profile == SHAPE_COVERAGE_BUCKETED:
        lines.append(
            "- Bucketed rule: paired comparison tasks must report per-shape numbers; "
            "trait-specific Gluon overlay tasks must spell out their shape bucket and host-side dispatch."
        )
    if previous_signal == "failed":
        lines.append("- Because prior Gluon work appears to have failed, keep the next overlay attempt layout-only, translation-only, or memory-only.")
    elif previous_signal == "slower":
        lines.append("- Because prior Gluon work appears slower, keep one targeted Gluon refinement only if it names a concrete memory/matrix performance mechanism; do not escalate to scheduler/persistent/async work or hybrid dispatch.")
    elif previous_signal == "attempted":
        lines.append("- Prior Gluon work produced only weak or inconclusive evidence; keep Gluon at L0/narrow refinement and spend extra width on plain Triton no-regression tasks.")
    elif previous_signal == "won":
        lines.append("- Because prior Gluon work appears promising, Gluon-specific refinement may expand, but keep a plain Triton or paired competitor.")
    has_prior_evidence = bool((previous_results_text or "").strip() or (previous_tasks_text or "").strip())
    lines.append(_build_evidence_anchored_composition_guidance(has_prior_evidence))
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
    input_dialect = str(feature_meta.get("input_dialect") or "").strip().lower()
    source_origin = str(feature_meta.get("source_origin") or "").strip().lower()
    existing_amd_gluon = (
        input_dialect == "amd_gluon"
        and source_origin == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
    )
    lines = [
        "## Gluon Planning Traits",
        f"- Detected traits: {', '.join(f'`{trait}`' for trait in traits)}",
        "- Use these traits to decide whether a Triton optimization direction deserves a Gluon implementation layer. They are planning constraints, not implementation templates.",
        "- Read guidance: start with `skills/triton-gluon/docs/00_always_read.md`, use its `stable_split_doc_index`, then view only the split-doc files and stable headings relevant to this task:",
        *[f"  - `{heading}`" for heading in _gluon_trait_headings(traits)],
        "",
        "Prefer First:",
        "- Semantics-preserving candidate: preserve launcher shape, indexing, masks/boundaries, correctness oracle, and benchmark intent; do not claim success from compile-only validation.",
    ]

    if "dialect_nv_gluon" in trait_set:
        lines.append("- Dialect overlay: translate NVIDIA-facing Gluon assumptions to AMD-facing Gluon for the same optimization direction; do not rename APIs mechanically.")
    elif existing_amd_gluon:
        lines.append("- Existing AMD Gluon input: keep the production structure and optimize one in-dialect atomic component at a time; plain Triton is optional comparison or fallback evidence, not a required `Plain competitor` for each component.")
    elif "dialect_amd_gluon" in trait_set:
        lines.append("- Dialect overlay: keep the existing AMD Gluon structure and optimize in-dialect only when source origin proves the measured baseline is production AMD Gluon; otherwise keep generated-overlay audit strict.")
    else:
        lines.append("- Dialect overlay: create a minimal AMD Gluon L0 only when it implements a named optimization direction with a concrete Gluon overlay reason.")

    lines.append("- Layout candidate: recover the base `BlockedLayout` from `tl.arange`, tile shape, `num_warps`, target family, and coalesced dimension; build host-side layouts as `constexpr` when they depend on launch choices.")

    if "layout_slice_broadcast" in trait_set:
        lines.append("- Layout candidate: preserve mask/broadcast/slice semantics with compatible `SliceLayout` or layout conversions instead of treating them as cleanup.")
    if "layout_source_first_required" in trait_set:
        lines.append("- Source-first candidate: read operator-local layout code before rewriting; preserve `DistributedLinearLayout`, descriptor, reshape/permute/trans, JIT/AOT, or prebuilt-kernel structure.")

    if existing_amd_gluon:
        lines.append("- Comparison target: use `true_baseline` or `safe_anchor` for in-dialect refinements. Use `Plain competitor:` only for generated overlays; do not point it at another Gluon task.")
    else:
        lines.append("- Plain competitor: keep a plain Triton competitor for each high-value direction when it remains an allowed output; Round 1 L0 overlays must name the exact same-batch plain task label in `Plain competitor:` and compare by the same benchmark contract.")

    lines.extend(["", "Consider Next:"])
    if "memory_generic" in trait_set:
        lines.append("- Memory lowering: use the routed Gluon memory docs and keep the first patch scoped to the simplest viable memory path.")
    if "memory_amd_buffer" in trait_set:
        lines.append("- AMD memory lowering: use the routed AMD memory docs only when target-family or source evidence justifies that path.")
    if "matrix_dot" in trait_set:
        lines.append("- Matrix lowering: use the routed Gluon matrix docs; do not treat matrix paths as textual rewrites.")
    if "matrix_scaled_dot" in trait_set:
        lines.append("- Scaled-matrix lowering: use the routed scaled-matrix docs and require dtype, scale-layout, and target-architecture evidence before assigning work.")
    if "matrix_wmma_descriptor" in trait_set:
        lines.append("- WMMA/descriptor path: use the routed architecture docs and treat descriptor-style paths as a separate target-family decision.")
    if any(trait in trait_set for trait in ("execution_jit_aot_sensitive", "version_sensitive", "operator_support_sensitive")):
        lines.append("- Runtime contract: verify JIT vs AOT, Triton minor version, `instr_shape` form, target backend, and operator-local arch guards before committing to a Gluon path.")

    lines.extend(
        [
            "",
            "Deprioritize Until Later:",
            "- Shared-memory swizzle, async copy, descriptor/tdm, scheduler hints, persistent kernels, atomics, and work-stealing until a simpler AMD Gluon candidate passes correctness.",
            "- Autotune-only or launch-only changes before the direction has a real performance hypothesis and a plain Triton competitor.",
            "- Tasks that mix NVIDIA and AMD layout families, introduce a top-level `gluon` kernel type, or produce an optimized `nv_gluon` output.",
            "",
            "Escalation rules:",
            "- Round 1 may include at most one L0 minimal AMD Gluon overlay when the priority docs place it in the Prefer or high-confidence Consider bucket; do not create a Gluon task unless it includes the full planner-audited fields: `Optimization direction:`, `Source Base family:`, `Plain competitor:`, `Gluon overlay reason:`, `Overlay priority:`, `Implementation layer:`, `Performance hypothesis:`, `Measurement boundary:`, `Comparison target:`, `Allowed change:`, and `Reject if:`.",
            "- Round 1 L0 is an execution anchor or one-subpath probe. Classify the scope before emitting it: avoid full-stage rewrites when the candidate includes complex control flow, online reductions, multiple matrix paths, multiple parent layouts, nested layout transforms, boundary-specific logic, or a tiny sub-100us stage.",
            "- Prefer the smallest executable target component for L0: scalar/1D stage, one load/store, one index/mask layout smoke path, or another explicitly smaller subpath. Do not hardcode any sample-specific stage name as the default.",
            "- L0 prompts must name one target component and must not instruct the worker to convert a full multi-stage kernel/helper unless that whole helper is explicitly the smallest viable component.",
            "- Additional overlay layers may become L1 trait-specific AMD Gluon tasks only after the L0 path is executed and performance-viable, or after local shape/sub-operation evidence shows Gluon beats the safe anchor.",
            "- In later rounds, if Gluon failed, shrink the next attempt to layout-only, translation-only, or memory-only work; if correctness passed but performance regressed, escalate memory/matrix lowering before scheduler or persistent work.",
            "- In later rounds, if either the plain Triton competitor or AMD Gluon overlay wins on only part of the benchmark surface, plan a mixed/hybrid candidate only when it preserves the plain Triton no-regression path and uses host-side dispatch or explicit feature checks to choose between plain Triton and AMD Gluon.",
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
    source_origin = str(feature_meta.get("source_origin") or "").strip().lower()
    search_policy = str(feature_meta.get("output_dialect_search_policy") or "").strip().lower()
    feature_mode = str(feature_meta.get("gluon_feature_mode") or "").strip().lower()
    existing_amd_gluon = (
        input_dialect == "amd_gluon"
        and source_origin == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
    )

    if existing_amd_gluon:
        lines = [
            "## Gluon Task Staging Policy",
            "- Follow the existing AMD Gluon refinement progression:",
            "  1. Pick a Triton optimization direction as taxonomy from profiling and baseline evidence.",
            "  2. Map it to one AMD Gluon atomic component and one task type.",
            "  3. Preserve source contract before editing: public ABI, measured route, target/JIT/AOT guards, fallback/artifact, layouts, and output feeding.",
            "  4. Generate 1-3 narrow in-dialect refinements, plus optional plain/shared comparison only when it is real fallback, portable-component, mixed-dispatch, or plain-subkernel work.",
            "  5. Defer high-coupling component combinations until evidence shows which pieces compose safely.",
            "- This path is Gluon-first. For a multi-GPU first round, prefer 2-3 `amd_gluon_*_refine` tasks across distinct atomic components; if only one safe refinement exists, emit fewer tasks rather than padding with generic plain Triton/Base work.",
            "- Plain/shared tasks are capped optional evidence. They must declare `Task type: plain_subkernel_refine` or `Task type: shared_or_plain_comparison`; otherwise they are treated as generic plain overflow and may be dropped before dispatch.",
            "- Do not assign generic plain/Base tasks a better priority than the only existing AMD Gluon refinement in a Gluon-priority run.",
            "- Favor early tasks that keep the original algorithm recognizable, make layout decisions explicit, and preserve correctness with the smallest possible semantic delta.",
            "- Treat the first passing AMD Gluon candidate as a platform for later aggressive optimizations, not as a final answer.",
            "- For low-latency kernels or tiny stages, record overhead evidence and stop launch-constant sweeps unless the next task names a specific overhead to remove.",
            "- For emitted AMD Gluon L0/existing-refinement tasks with `extension_intent=execution_anchor` and `minimum_executable_unit=whole_jit_kernel`, especially tiny/1D/reduction/load-store helpers, ask the worker for patch evolution: `patch_0` establishes a real executed anchor; if it is correct but slower, `patch_1+` must try exactly one named removable overhead or record why no named overhead applies.",
            "- Named slow-L0 overheads are generic: `host_layout_construction`, `layout_padding`, `mask_path_overhead`, `typed_fallback_overhead`, `loop_invariant_overhead`, `small_stage_launch_params`, or `memory_path_overhead`. Treat `tiny_stage_overhead` only as an umbrella label after checking concrete sub-sources.",
            "- If a task declares `Measurement boundary: kernel_only` but the runner may enter through a wrapper/fair/full path, require worker notes to include `Measurement boundary reconciliation` before edit/save.",
            "- Do not inject this Gluon patch-evolution ladder into `required_output_dialect=plain_triton` Base/plain tasks, including `plain_subkernel_refine` work inside an AMD Gluon input.",
            "- Mixed/hybrid optimization is a later-round strategy: combine plain Triton and AMD Gluon only after both sides have benchmark evidence and a visible no-regression dispatch condition.",
        ]
    else:
        lines = [
            "## Gluon Task Staging Policy",
            "- Follow the standard GEAK progression by optimization direction, then implementation layer:",
            "  1. Pick a Triton optimization direction from profiling and baseline evidence.",
            "  2. Keep a plain Triton competitor for each high-value direction.",
            "  3. Add L0 only as a minimal compileable AMD Gluon overlay for that same direction, and only from the `overlay_priority_routing` Prefer/high-confidence Consider buckets unless `gluon_feature_mode=force_l0_anchor` explicitly asks for one diagnostic execution anchor.",
            "  4. Add paired mapping when the same idea can be tested in both dialects.",
            "  5. Add L1 trait-specific lowering only after executed viable/local-win Gluon evidence.",
            "  6. Add Hybrid/mixed host-side dispatch only in later rounds with per-shape or sub-operation evidence.",
            "  7. persistent scheduling, atomics, or work-stealing last.",
            "- Favor early tasks that keep the original algorithm recognizable, make layout decisions explicit, and preserve correctness with the smallest possible semantic delta.",
            "- Treat the first passing AMD Gluon candidate as a platform for later aggressive optimizations, not as a final answer.",
            "- For low-latency kernels or tiny stages, treat L0 as a smallest executed anchor. If it passes correctness but is slower than Base, record overhead evidence and stop launch-constant sweeps unless the next task names a specific overhead to remove.",
            "- For emitted AMD Gluon L0 tasks with `extension_intent=execution_anchor` and `minimum_executable_unit=whole_jit_kernel`, especially tiny/1D/reduction/load-store helpers, ask the worker for patch evolution: `patch_0` establishes a real executed anchor; if it is correct but slower, `patch_1+` must try exactly one named removable overhead or record why no named overhead applies.",
            "- Named slow-L0 overheads are generic: `host_layout_construction`, `layout_padding`, `mask_path_overhead`, `typed_fallback_overhead`, `loop_invariant_overhead`, `small_stage_launch_params`, or `memory_path_overhead`. Treat `tiny_stage_overhead` only as an umbrella label after checking concrete sub-sources.",
            "- If a task declares `Measurement boundary: kernel_only` but the runner may enter through a wrapper/fair/full path, require worker notes to include `Measurement boundary reconciliation` before edit/save.",
            "- Do not inject this Gluon patch-evolution ladder into `required_output_dialect=plain_triton` Base/plain tasks, including `plain_subkernel_refine` work inside an AMD Gluon input.",
            "- Treat correctness-passing but slower AMD Gluon as neutral/slower evidence: it can provide layout/source information but should not trigger `gluon_variant` or `hybrid_dispatch` without a shape/sub-operation where Gluon beats the safe anchor.",
            "- Mixed/hybrid optimization is a later-round strategy: combine plain Triton and AMD Gluon only after both sides have benchmark evidence, and keep a plain Triton or paired competitor alive for no-regression.",
        ]

    if input_dialect == "plain_triton":
        lines.extend(
            [
                "- For `plain_triton -> amd_gluon`, add an early L0 task only when it has a concrete `Gluon overlay reason` tied to a named optimization direction after preserving the plain Triton competitor.",
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
    elif existing_amd_gluon:
        lines.extend(
            [
                "- For `source_origin=existing_amd_gluon_operator`, do not force the plain overlay model. Generate `amd_gluon_in_dialect_refine`, `amd_gluon_layout_or_matrix_refine`, or `amd_gluon_shape_dispatch_refine` tasks.",
                "- Use plain Triton family names as performance taxonomy labels only; they do not require mandatory plain Base task coverage for existing production Gluon input.",
                "- Each existing refinement task must choose one primary atomic component, preserve source contracts, include `Comparison target: true_baseline|safe_anchor`, and reject correctness failure or per-shape regression.",
                "- Each dispatchable existing refinement task must provide exactly one `Allowed change` and one concrete patch target via `Target symbol`, backticked local names, or `components(...)` / `expressions(...)`; abstract atomic components are routing labels, not target symbols.",
                "- If the planner cannot produce multiple safe in-dialect refinements, it should submit the smaller Gluon-first portfolio instead of filling the batch with untyped plain Triton tasks.",
                "- Combined tasks that touch wrapper dispatch, matrix layout, softmax/reduction state, and partition/scheduler policy together should be emitted only as `Task type: defer_composition` or postponed to a later evidence-based round.",
            ]
        )
    elif input_dialect == "amd_gluon":
        lines.extend(
            [
                "- For `amd_gluon` inputs, preserve the existing AMD Gluon structure first.",
                "- Do not assume this is a measured production Gluon operator unless `source_origin=existing_amd_gluon_operator`; otherwise generated-overlay contracts still apply.",
            ]
        )

    if search_policy == REQUIRE_AMD_GLUON_POLICY:
        lines.append("- AMD Gluon is required for this run, but the first goal is still a passing baseline candidate before the most ambitious rewrites.")
    elif search_policy == PREFER_AMD_GLUON_IF_VIABLE_POLICY:
        lines.append("- Prefer AMD Gluon only as an early same-direction overlay after preserving the plain Triton competitor; the first Gluon task is L0 viability, not a reason to replace plain Triton tasks.")
    if feature_mode == GLUON_FEATURE_MODE_FORCE_L0_ANCHOR:
        lines.append(
            "- Forced L0 anchor mode: if no Prefer/high-confidence Consider overlay exists, emit at most one ordinary-Consider AMD Gluon L0 execution anchor, mark it as compile/execution evidence, and keep its same-direction plain Triton competitor."
        )
    elif feature_mode == GLUON_FEATURE_MODE_REQUIRE_VIABLE:
        lines.append(
            "- Require viable Gluon mode: do not substitute a Base-only plan for a missing Gluon task. Emit a dispatchable Prefer/high-confidence Consider Gluon task, or let task generation report no viable Gluon task for this round."
        )

    return "\n".join(lines)


def _build_gluon_planning_contract(feature_meta: dict[str, Any]) -> str:
    """Return planner-safe Gluon constraints.

    This block is intentionally about task decomposition and viability, not
    worker-level implementation detail.
    """
    if not feature_uses_gluon_guidance_from_meta(feature_meta):
        return ""

    input_dialect = str(feature_meta.get("input_dialect") or "").strip().lower()
    source_origin = str(feature_meta.get("source_origin") or "").strip().lower()
    existing_amd_gluon = (
        input_dialect == "amd_gluon"
        and source_origin == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
    )
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
        "- Choose the optimization direction from the standard Triton priority path first, then decide whether AMD Gluon is a useful implementation layer for that direction.",
        "- Do not use `input_dialect=amd_gluon` alone to bypass Base/no-regression audit; only measured production operators with `source_origin=existing_amd_gluon_operator` use the in-dialect refinement path.",
        "- For generated overlays, emit strict API/layout policy metadata when useful: `source_origin: generated_overlay`, `gluon_tl_policy: strict_generated`, and `layout_construction_policy: host_preferred`.",
        "- Prefer tasks that establish a correctness-passing AMD Gluon overlay for one named direction before tasks that combine multiple difficult changes at once.",
    ]
    if existing_amd_gluon:
        lines.extend(
            [
                "- Existing AMD Gluon task contract: use `source_origin: existing_amd_gluon_operator`, `gluon_tl_policy: production_source_allowed`, `layout_construction_policy: source_preserve`, `required_output_dialect: amd_gluon`, and `Implementation layer: amd_gluon in-dialect refinement`.",
                "- Existing refinement tasks must include `Task type`, `Kernel family signal`, `Target component`, `Allowed change`, `Failure layers`, `Measurement boundary`, `Comparison target`, and `Reject if`.",
                "- Pick exactly one primary atomic component for `patch_0`; put secondary components in blockers/failure layers or `compose_later`, not in the allowed change.",
            ]
        )
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
            "explicit host-side dispatch that selects shape-specialized kernels or launch "
            "parameters. Do not hide the bucket decision inside `@triton.heuristics`."
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


def _task_config_uses_skills(
    *,
    kernel_type: str,
    cfg: dict[str, Any],
    task_body: str = "",
    label: str = "",
    requires_gluon_docs: bool | None = None,
) -> bool:
    """Return whether a generated worker task should expose skills.

    The task generator runs once for the whole Triton-family feature, but worker
    skill visibility is per task: plain Base tasks should not see the
    triton-gluon skill unless their own contract requires Gluon docs/output.
    """
    if str(kernel_type or "").strip().lower() != "triton":
        return False
    required_output = str(cfg.get("required_output_dialect") or "").strip().lower()
    if required_output in {"amd_gluon", "mixed"}:
        return False
    if requires_gluon_docs is None:
        requires_gluon_docs = _task_config_requires_gluon_worker_docs(
            kernel_type=kernel_type,
            cfg=cfg,
            task_body=task_body,
            label=label,
        )
    if requires_gluon_docs:
        return False
    if "use_skills" in cfg:
        raw = cfg.get("use_skills")
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    return False


def _build_output_dialect_guidance(feature_meta: dict[str, Any]) -> str:
    """Return planner guidance for how to prioritize output dialects."""
    policy = str(feature_meta.get("output_dialect_search_policy") or "").strip().lower()
    input_dialect = str(feature_meta.get("input_dialect") or "").strip().lower()
    source_origin = str(feature_meta.get("source_origin") or "").strip().lower()
    existing_amd_gluon = (
        input_dialect == "amd_gluon"
        and source_origin == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
    )
    preferred_outputs = feature_meta.get("preferred_output_dialects") or []
    preferred_text = ", ".join(str(item) for item in preferred_outputs)
    if policy == PREFER_AMD_GLUON_IF_VIABLE_POLICY:
        lines = [
            "## Output Dialect Planning Policy",
            f"- Preferred output dialect order: {preferred_text}",
            "- Treat output dialect selection as the implementation layer for a chosen optimization direction, not as a post-hoc preference or independent search direction.",
            "- Prefer an AMD Gluon overlay early only when the structure and target backend give a concrete Gluon mechanism, after preserving the same-direction plain Triton competitor.",
            "- Do not spend the whole first batch on fallback-only tuning when AMD Gluon remains viable, but also do not let Gluon replace plain Triton competitors.",
        ]
        if input_dialect == "nv_gluon":
            lines.append("- This input starts as NVIDIA-facing Gluon. Generate at least one early task that translates vendor-specific APIs, layout assumptions, or memory paths into AMD-facing Gluon before tuning.")
        elif existing_amd_gluon:
            lines.append("- This input is a measured production AMD Gluon operator. Keep the main path as in-dialect refinement; plain Triton is optional comparison/fallback evidence, not a mandatory overlay competitor.")
        elif input_dialect == "amd_gluon":
            lines.append("- This input is AMD Gluon but lacks measured production source-origin proof. Keep generated-overlay audit strict unless `source_origin=existing_amd_gluon_operator` is available.")
        else:
            lines.append(
                "- Generate at most one early L0 AMD Gluon overlay task, and only as a same-direction overlay from the `overlay_priority_routing` Prefer/high-confidence Consider buckets with the full planner-audited Gluon fields: "
                "`Optimization direction:`, `Source Base family:`, `Plain competitor:`, `Gluon overlay reason:`, `Overlay priority:`, `Implementation layer:`, "
                "`Performance hypothesis:`, `Measurement boundary:`, `Comparison target:`, `Allowed change:`, and `Reject if:`."
            )
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
            baseline_metrics=_bm_dict,
            previous_results_text=prev_results_summary,
            previous_tasks_text=prev_tasks_summary,
            current_round=current_round,
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


def _strict_base_family_audit_enabled() -> bool:
    return os.getenv("GEAK_STRICT_BASE_FAMILY_AUDIT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _explicit_base_family(task_text: str) -> str | None:
    match = re.search(r"base\s+family\s*:\s*`?([a-z0-9_/-]+)`?", task_text, re.IGNORECASE)
    return match.group(1).strip().lower() if match else None


def _task_matches_base_family(task: AgentTask, family: str) -> bool:
    text = f"{task.label}\n{task.task}".lower()
    explicit = _explicit_base_family(text)
    if explicit:
        return explicit == family
    return any(marker in text for marker in _BASE_FAMILY_LABEL_MARKERS.get(family, ()))


def _is_plain_competitor_task(task: AgentTask) -> bool:
    """Return whether this task can cover a mandatory plain Triton direction."""
    if _is_gluon_extension_task(task):
        return False
    required_output = str(task.config.get("required_output_dialect") or "").strip().lower()
    search_set = str(task.config.get("search_set") or "").strip().lower()
    implementation_layer = str(task.config.get("implementation_layer") or "").strip().lower()
    extension_layer = str(task.config.get("extension_layer") or "").strip().lower()
    if required_output != "plain_triton":
        return False
    has_gluon_signal = _task_has_explicit_gluon_signal(task)
    if "amd_gluon" in implementation_layer or "mixed" in implementation_layer or (
        extension_layer in {"l0", "l1", "hybrid"} and has_gluon_signal
    ):
        return False
    text = f"{task.label}\n{task.task}".lower()
    if "implementation layer: amd_gluon" in text or has_gluon_signal:
        return False
    return search_set == "base" or "base family:" in text


_EXISTING_AMD_GLUON_REFINEMENT_TASK_TYPES = {
    "amd_gluon_in_dialect_refine",
    "amd_gluon_layout_or_matrix_refine",
    "amd_gluon_shape_dispatch_refine",
}
_NON_REFINEMENT_TASK_TYPES = {
    "plain_subkernel_refine",
    "shared_or_plain_comparison",
    "defer_composition",
}
_TYPED_EXISTING_AMD_GLUON_PLAIN_TASK_TYPES = {
    "plain_subkernel_refine",
    "shared_or_plain_comparison",
}


def _task_source_origin(task: AgentTask) -> str:
    return _task_contract_value(task, "source_origin", "Source origin")


def _task_type(task: AgentTask) -> str:
    return _task_contract_value(task, "task_type", "Task type")


def _is_existing_amd_gluon_refinement_task(task: AgentTask) -> bool:
    """Return whether a task refines a measured production AMD Gluon source path."""
    if _task_source_origin(task) != SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR:
        return False
    task_type = _task_type(task)
    if task_type in _NON_REFINEMENT_TASK_TYPES:
        return False
    required_output = str(task.config.get("required_output_dialect") or "").strip().lower()
    implementation_layer = str(task.config.get("implementation_layer") or "").strip().lower()
    text = f"{task.label}\n{task.task}".lower()
    if task_type in _EXISTING_AMD_GLUON_REFINEMENT_TASK_TYPES:
        return True
    if "in-dialect refinement" in implementation_layer or "in dialect refinement" in implementation_layer:
        return required_output == "amd_gluon" or "amd_gluon" in implementation_layer
    return required_output == "amd_gluon" and (
        "amd_gluon" in implementation_layer
        or "existing amd gluon" in text
        or "production_source_allowed" in text
        or "source_preserve" in text
    )


def _plain_family_label(family: str, existing_labels: set[str]) -> str:
    stem = "triton-" + family.removeprefix("base_").replace("_", "-")
    label = stem
    idx = 2
    while label in existing_labels:
        label = f"{stem}-{idx}"
        idx += 1
    return label


def _plain_family_prompt(family: str) -> str:
    detail = _BASE_FAMILY_DETAILS.get(family, "plain Triton optimization direction")
    return "\n".join(
        [
            "Plain Triton competitor task",
            f"Base family: {family}",
            "Implementation layer: plain_triton",
            "Required output dialect: plain_triton",
            f"Optimization direction: {detail}",
            "Measurement boundary: kernel_only",
            "Comparison target: true_baseline",
            "Allowed change: one focused plain Triton optimization for this family",
            "Reject if: correctness fails, output falls back to AMD Gluon, or any benchmark shape has a significant regression",
        ]
    )


def _ensure_mandatory_plain_families(
    tasks: list[AgentTask],
    required_families: list[str],
    *,
    agent_class: type,
) -> list[AgentTask]:
    if not required_families:
        return tasks

    repaired = list(tasks)
    existing_labels = {task.label for task in repaired}
    for family in required_families:
        if any(_is_plain_competitor_task(task) and _task_matches_base_family(task, family) for task in repaired):
            continue
        label = _plain_family_label(family, existing_labels)
        existing_labels.add(label)
        repaired.append(
            AgentTask(
                agent_class=agent_class,
                task=_plain_family_prompt(family),
                label=label,
                priority=0,
                kernel_language="python",
                config={
                    "search_set": "base",
                    "required_output_dialect": "plain_triton",
                    "implementation_layer": "plain_triton",
                },
            )
        )
        logger.warning("Planner omitted mandatory plain Triton family %s; added fallback task %s.", family, label)
    return repaired


def _is_gluon_extension_task(task: AgentTask) -> bool:
    if _is_existing_amd_gluon_refinement_task(task):
        return False
    if (
        _task_source_origin(task) == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
        and _task_type(task) in _NON_REFINEMENT_TASK_TYPES
        and str(task.config.get("required_output_dialect") or "").strip().lower() == "plain_triton"
    ):
        return False
    required_output = str(task.config.get("required_output_dialect") or "").strip().lower()
    if required_output in {"amd_gluon", "mixed"}:
        return True
    implementation_layer = str(task.config.get("implementation_layer") or "").strip().lower()
    if "amd_gluon" in implementation_layer:
        return True
    text = f"{task.label}\n{task.task}".lower()
    search_set = str(task.config.get("search_set") or "").strip().lower()
    if search_set == "shared" or "shared set task" in text or "shared source family:" in text:
        return False
    doc_profile = str(task.config.get("gluon_doc_profile") or "").strip().lower()
    if doc_profile and doc_profile != "base_or_shared_gluon":
        return True
    if task.config.get("required_gluon_docs"):
        return True
    return any(
        marker in text
        for marker in (
            "@gluon.jit",
            "amd_gluon",
            "nv_gluon",
            "amd-gluon",
            "triton-gluon",
            "gluon overlay",
            "gluon variant",
            "gluon_doc_profile",
            "required_gluon_docs",
        )
    ) or (
        "extension set" in text
        or task.label.lower().startswith(("ext-", "extension-"))
    ) and "gluon" in text


def _task_has_explicit_gluon_signal(task: AgentTask) -> bool:
    return _is_gluon_extension_task(task) or _is_existing_amd_gluon_refinement_task(task)


def _has_l0_extension_tag(task: AgentTask) -> bool:
    text = f"{task.label}\n{task.task}".lower()
    return "extension layer: l0" in text or " l0 " in f" {text} " or "-l0-" in text


def _is_l0_gluon_overlay_task(task: AgentTask) -> bool:
    if not _is_gluon_extension_task(task):
        return False
    extension_layer = str(task.config.get("extension_layer") or "").strip().lower()
    implementation_layer = str(task.config.get("implementation_layer") or "").strip().lower()
    required_output = str(task.config.get("required_output_dialect") or "").strip().lower()
    if extension_layer == "l0":
        return True
    if extension_layer in {"l1", "hybrid"}:
        return False
    if _has_l0_extension_tag(task):
        return True
    return required_output == "amd_gluon" and "amd_gluon" in implementation_layer and "overlay" in implementation_layer


def _has_l1_extension_tag(task: AgentTask) -> bool:
    text = f"{task.label}\n{task.task}".lower()
    return "extension layer: l1" in text or " l1 " in f" {text} " or "-l1-" in text


def _l1_anchor_contract_errors(task: AgentTask) -> list[str]:
    text = f"{task.label}\n{task.task}"
    lower = text.lower()
    if not _is_gluon_extension_task(task) or not _has_l1_extension_tag(task):
        return []

    missing: list[str] = []
    for field in (
        "Anchor patch:",
        "Anchor speedup:",
        "Anchor execution:",
        "Comparison target:",
        "Allowed change:",
        "Reject if:",
    ):
        if field.lower() not in lower:
            missing.append(field)
    if "comparison target: anchor_patch" not in lower:
        missing.append("Comparison target: anchor_patch")
    if "anchor execution: true" not in lower:
        missing.append("Anchor execution: true")
    if missing:
        return [f"{task.label} missing L1 anchor contract fields: {', '.join(missing)}"]
    return []


def _has_prompt_field(text: str, field: str) -> bool:
    return re.search(rf"^\s*{re.escape(field)}\s*:", text, re.IGNORECASE | re.MULTILINE) is not None


def _gluon_overlay_contract_errors(task: AgentTask) -> list[str]:
    if not _is_gluon_extension_task(task):
        return []
    text = f"{task.label}\n{task.task}"
    missing = [
        field
        for field in (
            "Optimization direction",
            "Source Base family",
            "Plain competitor",
            "Gluon overlay reason",
            "Overlay priority",
            "Implementation layer",
            "Performance hypothesis",
            "Measurement boundary",
            "Comparison target",
            "Allowed change",
            "Reject if",
        )
        if not _has_prompt_field(text, field)
    ]
    if missing:
        return [f"{task.label} missing Gluon overlay contract fields: {', '.join(missing)}"]
    return []


def _gluon_overlay_priority_errors(task: AgentTask) -> list[str]:
    if not _is_l0_gluon_overlay_task(task):
        return []
    text = f"{task.label}\n{task.task}"
    priority = (_parse_prompt_field_value(text, "Overlay priority") or "").strip().lower()
    if not priority:
        return [f"{task.label} missing Overlay priority"]
    if priority.startswith("prefer"):
        return []
    if "consider" in priority and any(
        marker in priority for marker in ("high", "strong", "confidence", "confident")
    ):
        return []
    return [f"{task.label} has invalid L0 Overlay priority `{priority}`; expected Prefer or high-confidence Consider"]


def _has_valid_l0_overlay_priority(task: AgentTask) -> bool:
    return not _gluon_overlay_priority_errors(task)


_VALID_MINIMUM_EXECUTABLE_UNITS = {"inline_scoped_helper", "separate_gluon_kernel", "whole_jit_kernel", "infeasible"}
_VALID_L0_ALLOWED_EXECUTION_PATHS = {"inline_scoped_helper", "separate_gluon_kernel", "whole_jit_kernel"}
_VALID_SCOPE_INFEASIBLE_POLICIES = {"do_not_emit", "shrink_or_report", "separate_kernel_if_allowed"}
_VALID_L0_SCOPE_CLASSIFICATIONS = {"low_coupling", "high_coupling", "infeasible"}
_WHOLE_JIT_COMPILE_RISK_FIELDS = (
    "expected_failure_layers",
    "first_patch_compile_goal",
    "do_not_optimize_before_compile",
    "matrix_lowering_required",
)
_HIGH_COUPLING_L0_MARKERS = (
    "tl.dot",
    "gl.dot",
    "dot product",
    "matrix",
    "mfma",
    "wmma",
    "online softmax",
    "softmax accumulator",
    "accumulator loop",
    "stage1/stage2",
    "stage1 stage2",
    "cross-stage",
    "wrapper reroute",
    "wrapper route",
    "whole kernel",
    "whole-kernel",
    "entire kernel",
    "full kernel",
)

_NEGATIVE_TASK_LINE_PREFIXES = (
    "reject if",
    "forbidden change",
    "do not",
    "don't",
)
_NEGATIVE_TASK_LINE_MARKERS = (
    "must not",
    "should not",
    "not introduce",
    "do not introduce",
    "do not add",
)
_CONTEXT_ONLY_TASK_FIELDS = (
    "secondary components",
    "secondary component",
    "blockers",
    "blocked by",
    "failure layers",
    "expected failure layers",
    "routed doc reasons",
)
_PATCH_TARGET_TASK_FIELDS = (
    "target component",
    "allowed change",
    "primary atomic component",
    "primary component",
    "target symbol",
)
_ATOMIC_COMPONENT_MARKERS = (
    "index_map",
    "mask_boundary",
    "load_store",
    "layout_broadcast",
    "matrix_operand",
    "scale_dtype",
    "reduction_accumulator",
    "selection_update",
    "state_update",
    "epilogue_fusion",
    "shape_dispatch",
    "wrapper_integration",
    "scheduler_launch",
)


def _normalize_audit_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _task_contract_value(task: AgentTask, key: str, field: str) -> str:
    value = task.config.get(key)
    if value in (None, ""):
        value = _parse_prompt_field_value(task.task, field)
    return str(value or "").strip().strip("`").lower()


def _task_contract_raw_value(task: AgentTask, key: str, field: str) -> str:
    value = task.config.get(key)
    if value in (None, ""):
        value = _parse_prompt_field_value(task.task, field)
    return str(value).strip().strip("`") if value not in (None, "") else ""


def _is_negative_task_line(line: str) -> bool:
    lowered = str(line or "").strip().lower()
    return lowered.startswith(_NEGATIVE_TASK_LINE_PREFIXES) or any(
        marker in lowered for marker in _NEGATIVE_TASK_LINE_MARKERS
    )


def _filtered_task_lines(task_body: str) -> list[str]:
    return [line for line in (task_body or "").splitlines() if not _is_negative_task_line(line)]


def _field_name_from_line(line: str) -> str:
    return str(line or "").split(":", 1)[0].strip().lower() if ":" in str(line or "") else ""


def _gluon_l0_patch_target_text(task: AgentTask) -> str:
    parts: list[str] = []
    for key, field in (
        ("target_component", "Target component"),
        ("target_symbol", "Target symbol"),
        ("allowed_change", "Allowed change"),
        ("primary_atomic_component", "Primary atomic component"),
        ("primary_component", "Primary component"),
    ):
        raw_value = _task_contract_raw_value(task, key, field)
        if raw_value:
            parts.append(raw_value)
    for value in task.config.get("required_patch_target_symbols") or []:
        if value:
            parts.append(str(value))
    for line in _filtered_task_lines(task.task):
        field = _field_name_from_line(line)
        if any(field.startswith(prefix) for prefix in _PATCH_TARGET_TASK_FIELDS):
            parts.append(line)
    return "\n".join(part for part in parts if part).lower()


def _gluon_l0_non_negative_context_text(task: AgentTask) -> str:
    return "\n".join(_filtered_task_lines(task.task)).lower()


def _gluon_l0_positive_signal_text(task: AgentTask) -> str:
    parts = [
        task.label,
        _gluon_l0_patch_target_text(task),
        _task_contract_raw_value(task, "task_signals", "Task signals"),
        _task_contract_raw_value(task, "l0_coupling_reasons", "L0 coupling reasons"),
        _task_contract_raw_value(task, "kernel_family_signal", "Kernel family signal"),
    ]
    return "\n".join(str(part or "") for part in parts).lower()


def _l0_high_coupling_hits(task: AgentTask) -> list[str]:
    text = "\n".join(
        part
        for part in (
            task.label,
            _gluon_l0_patch_target_text(task),
            _task_contract_raw_value(task, "l0_coupling_reasons", "L0 coupling reasons"),
        )
        if part
    ).lower()
    hits: list[str] = []
    for marker in _HIGH_COUPLING_L0_MARKERS:
        if marker in text and marker not in hits:
            hits.append(marker)
    return hits


def _task_truthy_contract_value(task: AgentTask, key: str, field: str) -> bool:
    value = _task_contract_raw_value(task, key, field).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _task_forbidden_symbols(task: AgentTask) -> list[str]:
    symbols: list[str] = []
    raw = task.config.get("forbidden_patch_target_symbols")
    if isinstance(raw, str):
        symbols.extend(part.strip().strip("`") for part in raw.split(","))
    elif isinstance(raw, list):
        symbols.extend(str(part).strip().strip("`") for part in raw)
    for key in ("forbidden_change", "forbidden_changes"):
        symbols.extend(forbidden_symbols_from_scoped_text(task.config.get(key)))
    for field in ("Forbidden change", "Reject if"):
        value = _parse_prompt_field_value(task.task, field)
        if value:
            symbols.extend(forbidden_symbols_from_scoped_text(value))
    for value in do_not_clauses(task.task):
        symbols.extend(forbidden_symbols_from_scoped_text(value))
    unique: list[str] = []
    for symbol in symbols:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol) and symbol not in unique:
            unique.append(symbol)
    return unique


def _gluon_l0_soft_diagnostics(task: AgentTask) -> list[str]:
    """Return non-blocking diagnostics for L0 task plans.

    These hints are intentionally advisory. Hard gating remains limited to the
    existing output dialect, scope, and binding contracts.
    """
    if not _is_l0_gluon_overlay_task(task):
        return []
    patch_target_text = _gluon_l0_patch_target_text(task)
    context_text = _gluon_l0_non_negative_context_text(task)
    positive_signal_text = _gluon_l0_positive_signal_text(task)
    text = "\n".join(
        str(part or "")
        for part in (
            task.label,
            patch_target_text,
            task.config.get("task_signals"),
            task.config.get("target_component"),
            task.config.get("l0_coupling_reasons"),
            task.config.get("kernel_family_signal"),
        )
    ).lower()
    minimum_unit = _task_contract_value(task, "minimum_executable_unit", "Minimum executable unit")
    target_component = _task_contract_raw_value(task, "target_component", "Target component").lower()
    whole_reason = _task_contract_raw_value(task, "whole_kernel_required_reason", "Whole kernel required reason").lower()
    failure_text = " ".join(
        str(part or "")
        for part in (
            task.config.get("failure_layers"),
            _parse_prompt_field_value(task.task, "Failure layers"),
            task.config.get("expected_failure_layers"),
            _parse_prompt_field_value(task.task, "Expected failure layers"),
        )
    ).lower()
    matrix_required_raw = _task_contract_raw_value(task, "matrix_lowering_required", "Matrix lowering required")
    matrix_required = matrix_required_raw.strip().lower() in {"1", "true", "yes", "on"}

    diagnostics: list[str] = []
    local_markers = ("load", "store", "index", "mask", "path", "offset", "pointer")
    broad_markers = ("whole", "helper", "kernel", "stage")
    target_scope_text = f"{target_component}\n{patch_target_text}".lower()
    if minimum_unit == "whole_jit_kernel" and any(marker in target_scope_text for marker in local_markers):
        high_coupling_markers = {
            "matrix": ("matrix", "dot", "mfma", "wmma"),
            "reduction": ("reduction", "softmax", "accumulator", "sum", "max"),
            "wrapper": ("wrapper", "dispatch", "public api", "import path"),
            "layout": ("broadcast", "slice", "mask", "layout", "rope"),
        }
        missing_layers = [
            layer
            for layer, markers in high_coupling_markers.items()
            if any(marker in text or marker in whole_reason for marker in markers) and layer not in failure_text
        ]
        if not whole_reason or missing_layers:
            diagnostics.append("local_target_promoted_to_whole_kernel")
        elif (
            any(marker in target_scope_text for marker in broad_markers)
            and any(marker in text or marker in whole_reason for marker in ("matrix", "dot", "mfma", "wmma"))
            and not all(layer in failure_text for layer in ("layout", "matrix"))
        ):
            diagnostics.append("local_target_promoted_to_whole_kernel")

    target_has_matrix = any(marker in patch_target_text for marker in ("matrix", "dot", "mfma", "wmma", "matrix_operand"))
    whole_matrix_skeleton = minimum_unit == "whole_jit_kernel" and target_has_matrix
    if (target_has_matrix or whole_matrix_skeleton) and not matrix_required:
        diagnostics.append("matrix_metadata_inconsistent")
    if minimum_unit == "whole_jit_kernel" and any(
        marker in context_text for marker in ("broadcast", "slice", "[:,", "[none", "mask", "rope")
    ):
        if not any(layer in failure_text for layer in ("broadcast", "layout", "slice")):
            diagnostics.append("broadcast_heavy_missing_layout_map")
    if any(marker in positive_signal_text for marker in ("softmax", "reduction", "reduce", "rms", "sum", "max", "topk")):
        if not any(layer in failure_text for layer in ("reduction", "accumulator")):
            diagnostics.append("reduction_metadata_inconsistent")
    if any(marker in positive_signal_text for marker in ("scaled", "scale", "fp8", "fp4", "quant")):
        if not any(layer in failure_text for layer in ("scale", "dtype", "matrix")):
            diagnostics.append("scale_layout_missing")
    if "bundle_allowed=true" not in context_text:
        component_hits = sum(
            1
            for marker in _ATOMIC_COMPONENT_MARKERS
            if marker in patch_target_text
        )
        if component_hits > 1:
            diagnostics.append("component_bundle_too_broad")
    if minimum_unit == "whole_jit_kernel":
        family = str(task.config.get("kernel_family_signal") or _parse_prompt_field_value(task.task, "Kernel family signal") or "")
        if not family.strip() or family.strip().lower() in {"unknown", "generic", "none"}:
            diagnostics.append("unknown_family_whole_kernel")

    unique: list[str] = []
    for diagnostic in diagnostics:
        if diagnostic not in unique:
            unique.append(diagnostic)
    return unique


_REQUIRED_GLUON_REPAIR_DIAGNOSTICS = {
    "local_target_promoted_to_whole_kernel",
    "matrix_metadata_inconsistent",
    "component_bundle_too_broad",
    "unknown_family_whole_kernel",
}


def _required_gluon_soft_diagnostic_errors(task: AgentTask) -> list[str]:
    """Return repair-before-dispatch errors for required Gluon L0 plans."""
    if not _is_l0_gluon_overlay_task(task):
        return []
    required_output = str(task.config.get("required_output_dialect") or "").strip().lower()
    if required_output != "amd_gluon":
        return []
    diagnostics = [
        diagnostic
        for diagnostic in _gluon_l0_soft_diagnostics(task)
        if diagnostic in _REQUIRED_GLUON_REPAIR_DIAGNOSTICS
    ]
    if diagnostics:
        logger.warning(
            "High-risk Gluon L0 diagnostics for %s will be handled by worker/selector checks: %s",
            task.label,
            ", ".join(diagnostics),
        )
    return []


def _gluon_l0_execution_boundary_errors(task: AgentTask) -> list[str]:
    if not _is_l0_gluon_overlay_task(task):
        return []

    minimum_unit = _task_contract_value(task, "minimum_executable_unit", "Minimum executable unit")
    allowed_path = _task_contract_value(task, "allowed_execution_path", "Allowed execution path")
    infeasible_policy = _task_contract_value(task, "scope_infeasible_policy", "Scope infeasible policy")
    scope_classification = _task_contract_value(task, "l0_scope_classification", "L0 scope classification")
    coupling_reasons = _task_contract_raw_value(task, "l0_coupling_reasons", "L0 coupling reasons")
    required_output = str(task.config.get("required_output_dialect") or "").strip().lower()
    errors: list[str] = []

    if scope_classification and scope_classification not in _VALID_L0_SCOPE_CLASSIFICATIONS:
        errors.append(f"{task.label} has invalid L0 scope classification `{scope_classification}`")
    if scope_classification in {"high_coupling", "infeasible"} and not coupling_reasons:
        logger.warning(
            "%s declares %s L0 scope without coupling reasons; worker docs will route from task text.",
            task.label,
            scope_classification,
        )

    if minimum_unit and minimum_unit not in _VALID_MINIMUM_EXECUTABLE_UNITS:
        errors.append(f"{task.label} has invalid Minimum executable unit `{minimum_unit}`")

    if allowed_path and allowed_path not in _VALID_L0_ALLOWED_EXECUTION_PATHS:
        errors.append(f"{task.label} has invalid Allowed execution path `{allowed_path}`")

    if infeasible_policy and infeasible_policy not in _VALID_SCOPE_INFEASIBLE_POLICIES:
        errors.append(f"{task.label} has invalid Scope infeasible policy `{infeasible_policy}`")

    if minimum_unit == "infeasible" and required_output == "amd_gluon":
        errors.append(f"{task.label} cannot require AMD Gluon output when Minimum executable unit is infeasible")
    if scope_classification == "infeasible" and required_output == "amd_gluon":
        errors.append(f"{task.label} cannot require AMD Gluon output when L0 scope classification is infeasible")
    if minimum_unit in _VALID_L0_ALLOWED_EXECUTION_PATHS and allowed_path and minimum_unit != allowed_path:
        errors.append(
            f"{task.label} Minimum executable unit `{minimum_unit}` must match Allowed execution path `{allowed_path}`"
        )

    high_coupling_hits = _l0_high_coupling_hits(task)
    if allowed_path == "inline_scoped_helper" and (
        scope_classification == "high_coupling" or high_coupling_hits
    ):
        reasons = ", ".join(high_coupling_hits) if high_coupling_hits else coupling_reasons or scope_classification
        errors.append(
            f"{task.label} cannot use inline_scoped_helper for high-coupling L0 target ({reasons}); shrink the target, report infeasible, or justify a different executable unit"
        )

    forbidden_symbols = _task_forbidden_symbols(task)
    if allowed_path == "whole_jit_kernel" and "whole_kernel_rewrite" in forbidden_symbols:
        errors.append(f"{task.label} allows whole_jit_kernel but forbids whole-kernel rewrite")

    target_component = str(task.config.get("target_component") or _parse_prompt_field_value(task.task, "Target component") or "").strip()
    whole_reason = str(
        task.config.get("whole_kernel_required_reason")
        or _parse_prompt_field_value(task.task, "Whole kernel required reason")
        or ""
    ).strip()
    if minimum_unit == "whole_jit_kernel" and target_component and not whole_reason:
        logger.warning(
            "%s uses whole_jit_kernel for a scoped target without Whole kernel required reason; "
            "worker/selector checks will enforce scope and execution.",
            task.label,
        )
    if minimum_unit == "whole_jit_kernel" or allowed_path == "whole_jit_kernel":
        missing_compile_risk = [
            key
            for key in _WHOLE_JIT_COMPILE_RISK_FIELDS
            if not _task_contract_raw_value(task, key, key.replace("_", " ").title())
        ]
        if missing_compile_risk:
            logger.warning(
                "%s whole_jit_kernel compile-risk anchor missing advisory fields: %s",
                task.label,
                ", ".join(missing_compile_risk),
            )
        if not _task_truthy_contract_value(task, "do_not_optimize_before_compile", "Do not optimize before compile"):
            logger.warning("%s whole_jit_kernel did not set do_not_optimize_before_compile: true", task.label)

    return errors


def _mode_allows_low_priority_l0_anchor(gluon_feature_mode: str | None) -> bool:
    mode = str(gluon_feature_mode or "").strip().lower()
    return mode in {GLUON_FEATURE_MODE_FORCE, GLUON_FEATURE_MODE_FORCE_L0_ANCHOR}


def _mode_requires_viable_gluon(gluon_feature_mode: str | None) -> bool:
    return str(gluon_feature_mode or "").strip().lower() == GLUON_FEATURE_MODE_REQUIRE_VIABLE


def _dropped_gluon_overlay_summary(task: AgentTask, errors: list[str]) -> dict[str, Any]:
    summary = _task_audit_summary(task)
    text = f"{task.label}\n{task.task}"
    return {
        "label": task.label,
        "priority": task.priority,
        "drop_reason": "; ".join(errors),
        "drop_reasons": list(errors),
        "overlay_priority": _parse_prompt_field_value(text, "Overlay priority") or "",
        "source_base_family": summary.get("source_base_family") or "",
        "plain_competitor": summary.get("plain_competitor") or "",
        "l0_scope_classification": summary.get("l0_scope_classification") or "",
        "minimum_executable_unit": summary.get("minimum_executable_unit") or "",
        "allowed_execution_path": summary.get("allowed_execution_path") or "",
        "target_component": summary.get("target_component") or "",
    }


def _filter_low_priority_l0_overlays(
    tasks: list[AgentTask],
    *,
    gluon_feature_mode: str | None = None,
) -> tuple[list[AgentTask], list[dict[str, Any]]]:
    filtered: list[AgentTask] = []
    dropped: list[dict[str, Any]] = []
    allow_low_priority_anchor = _mode_allows_low_priority_l0_anchor(gluon_feature_mode)
    for task in tasks:
        priority = _parse_prompt_field_value(f"{task.label}\n{task.task}", "Overlay priority")
        if (
            _is_l0_gluon_overlay_task(task)
            and priority
            and not _has_valid_l0_overlay_priority(task)
            and not allow_low_priority_anchor
        ):
            errors = _gluon_overlay_priority_errors(task)
            logger.warning(
                "Dropping low-priority AMD Gluon L0 overlay %s before task write: %s",
                task.label,
                "; ".join(errors),
            )
            dropped.append(_dropped_gluon_overlay_summary(task, errors))
            continue
        filtered.append(task)
    return filtered, dropped


def _write_task_generation_gluon_diagnostics(
    *,
    diagnostics_dir: Path | None,
    dropped_gluon_overlays: list[dict[str, Any]],
    dropped_existing_amd_plain_overflow: list[dict[str, Any]] | None = None,
    no_viable_gluon_task: dict[str, Any] | None = None,
) -> None:
    dropped_existing_amd_plain_overflow = list(dropped_existing_amd_plain_overflow or [])
    if not dropped_gluon_overlays and not dropped_existing_amd_plain_overflow and not no_viable_gluon_task:
        _set_last_gluon_task_generation_diagnostics({})
        return
    payload: dict[str, Any] = {
        "dropped_gluon_overlays": list(dropped_gluon_overlays),
    }
    if dropped_existing_amd_plain_overflow:
        payload["dropped_existing_amd_plain_overflow"] = dropped_existing_amd_plain_overflow
    if no_viable_gluon_task:
        payload["no_viable_gluon_task"] = dict(no_viable_gluon_task)
    _set_last_gluon_task_generation_diagnostics(payload)

    out_dir = diagnostics_dir
    if out_dir is None:
        env_dir = os.getenv("GEAK_TASKGEN_AUDIT_DUMP_DIR")
        out_dir = Path(env_dir) if env_dir else None
    if out_dir is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "task_generation_gluon_diagnostics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
    except OSError as exc:
        logger.warning("Could not write task-generation Gluon diagnostics: %s", exc)


def _gluon_overlay_binding_errors(task: AgentTask, tasks: list[AgentTask]) -> list[str]:
    """Require Round-1 L0 overlays to bind to a concrete plain competitor task."""
    if not _is_l0_gluon_overlay_task(task):
        return []

    text = f"{task.label}\n{task.task}"
    source_family = _parse_tagged_value(text, "Source Base family")
    plain_ref = _parse_prompt_field_value(text, "Plain competitor")
    if not plain_ref:
        return [f"{task.label} missing per-task overlay binding field: Plain competitor"]

    normalized_ref = plain_ref.strip().strip("`").lower()
    plain_task = next((candidate for candidate in tasks if candidate.label.lower() == normalized_ref), None)
    if plain_task is None:
        return [f"{task.label} Plain competitor `{plain_ref}` does not match a task label in this batch"]
    if not _is_plain_competitor_task(plain_task):
        return [f"{task.label} Plain competitor `{plain_ref}` is not a plain Triton competitor task"]
    if source_family and not _task_matches_base_family(plain_task, source_family):
        return [
            f"{task.label} Source Base family `{source_family}` does not match Plain competitor `{plain_ref}`"
        ]
    plain_direction = _parse_prompt_field_value(plain_task.task, "Optimization direction")
    gluon_direction = _parse_prompt_field_value(text, "Optimization direction")
    if not plain_direction:
        return [f"{task.label} Plain competitor `{plain_ref}` lacks auditable Optimization direction"]
    if _normalize_audit_text(plain_direction) != _normalize_audit_text(gluon_direction):
        return [
            f"{task.label} Optimization direction `{gluon_direction}` does not match Plain competitor `{plain_ref}` direction `{plain_direction}`"
        ]

    plain_component = (
        str(plain_task.config.get("target_component") or "")
        or _parse_prompt_field_value(plain_task.task, "Target component")
        or _parse_prompt_field_value(plain_task.task, "Allowed change")
        or ""
    )
    gluon_component = (
        str(task.config.get("target_component") or "")
        or _parse_prompt_field_value(task.task, "Target component")
        or _parse_prompt_field_value(task.task, "Allowed change")
        or ""
    )
    def component_tokens(value: str) -> set[str]:
        tokens = set(target_symbols_from_scoped_text(value))
        stripped = str(value or "").strip().strip("`")
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stripped):
            tokens.add(stripped)
        return tokens

    def scope_tokens(task_obj: AgentTask) -> set[str]:
        values: list[str] = []
        for key in ("target_symbol", "target_component"):
            value = task_obj.config.get(key)
            if value:
                values.append(str(value))
        for value in task_obj.config.get("required_patch_target_symbols") or []:
            values.append(str(value))
        for field in ("Target symbol", "Target component", "Allowed change"):
            parsed = _parse_prompt_field_value(task_obj.task, field)
            if parsed:
                values.append(parsed)
        tokens: set[str] = set()
        for value in values:
            tokens.update(component_tokens(value))
        return tokens

    plain_tokens = component_tokens(plain_component)
    gluon_tokens = component_tokens(gluon_component)
    plain_scope_tokens = scope_tokens(plain_task)
    gluon_scope_tokens = scope_tokens(task)
    plain_target_symbol = (
        str(plain_task.config.get("target_symbol") or "")
        or _parse_prompt_field_value(plain_task.task, "Target symbol")
        or ""
    ).strip().strip("`")
    gluon_target_symbol = (
        str(task.config.get("target_symbol") or "")
        or _parse_prompt_field_value(task.task, "Target symbol")
        or ""
    ).strip().strip("`")
    if not plain_tokens:
        return [
            f"{task.label} Plain competitor `{plain_ref}` lacks auditable Target component or Allowed change for L0 overlay binding"
        ]
    if not gluon_tokens:
        return [f"{task.label} lacks auditable Target component or Allowed change for L0 overlay binding"]
    if plain_tokens.isdisjoint(gluon_tokens):
        return [
            f"{task.label} Target component `{gluon_component}` does not match Plain competitor `{plain_ref}` component `{plain_component}`"
        ]
    if plain_target_symbol and gluon_target_symbol and plain_target_symbol != gluon_target_symbol:
        return [
            f"{task.label} Target scope `{gluon_target_symbol}` does not match Plain competitor `{plain_ref}` target scope `{plain_target_symbol}`"
        ]
    if gluon_scope_tokens and plain_scope_tokens and plain_scope_tokens.isdisjoint(gluon_scope_tokens):
        return [
            f"{task.label} Target scope `{sorted(gluon_scope_tokens)}` does not match Plain competitor `{plain_ref}` target scope `{sorted(plain_scope_tokens)}`"
        ]
    return []


def _existing_amd_gluon_refinement_contract_errors(task: AgentTask) -> list[str]:
    if not _is_existing_amd_gluon_refinement_task(task):
        return []
    text = f"{task.label}\n{task.task}"
    errors: list[str] = []
    implementation_layer = _task_contract_value(task, "implementation_layer", "Implementation layer")
    if "amd_gluon" not in implementation_layer or "refinement" not in implementation_layer:
        errors.append(f"{task.label} existing AMD Gluon refinement missing `Implementation layer: amd_gluon in-dialect refinement`")
    if not _task_type(task):
        errors.append(f"{task.label} existing AMD Gluon refinement missing `Task type`")
    if not _task_contract_raw_value(task, "kernel_family_signal", "Kernel family signal"):
        errors.append(f"{task.label} existing AMD Gluon refinement missing `Kernel family signal`")
    target_component = _task_contract_raw_value(task, "target_component", "Target component")
    allowed_change = _parse_prompt_field_value(text, "Allowed change") or ""
    if not target_component:
        errors.append(f"{task.label} existing AMD Gluon refinement missing `Target component`")
    if not allowed_change:
        errors.append(f"{task.label} existing AMD Gluon refinement missing `Allowed change`")
    if not task.config.get("required_patch_target_symbols"):
        errors.append(
            f"{task.label} existing AMD Gluon refinement missing concrete patch target; "
            "`Target component` is a routing label, so provide `Target symbol`, backticked local names, "
            "or `components(...)` / `expressions(...)` scope"
        )
    comparison_target = _task_contract_value(task, "comparison_target", "Comparison target")
    if comparison_target not in {"true_baseline", "safe_anchor"}:
        errors.append(f"{task.label} existing AMD Gluon refinement requires `Comparison target: true_baseline|safe_anchor`")
    if not _parse_prompt_field_value(text, "Reject if"):
        errors.append(f"{task.label} existing AMD Gluon refinement missing `Reject if`")
    if not _task_contract_raw_value(task, "failure_layers", "Failure layers"):
        errors.append(f"{task.label} existing AMD Gluon refinement missing `Failure layers`")
    if not task.config.get("gluon_doc_profile"):
        errors.append(f"{task.label} existing AMD Gluon refinement missing `gluon_doc_profile`")
    if _task_contract_value(task, "gluon_tl_policy", "Gluon TL policy") != "production_source_allowed":
        errors.append(f"{task.label} existing AMD Gluon refinement requires `gluon_tl_policy: production_source_allowed`")
    if _task_contract_value(task, "layout_construction_policy", "Layout construction policy") != "source_preserve":
        errors.append(f"{task.label} existing AMD Gluon refinement requires `layout_construction_policy: source_preserve`")
    patch_target_text = _gluon_l0_patch_target_text(task)
    if "bundle_allowed=true" not in text.lower():
        component_hits = [
            marker
            for marker in (
                "wrapper_shape_dispatch",
                "layout_parent_slice",
                "load_store_buffer",
                "matrix_operand_mfma",
                "reduction_accumulator",
                "state_update_softmax",
                "epilogue_output_store",
                "scheduler_launch_runtime",
                "source_contract_integration",
            )
            if marker in patch_target_text
        ]
        if len(component_hits) > 1:
            errors.append(
                f"{task.label} existing AMD Gluon refinement targets multiple primary components {component_hits}; use one primary component or defer composition"
            )
    return errors


def _existing_amd_gluon_refinement_warnings(task: AgentTask, tasks: list[AgentTask]) -> list[str]:
    if not _is_existing_amd_gluon_refinement_task(task):
        return []
    text = f"{task.label}\n{task.task}"
    warnings: list[str] = []
    plain_ref = _parse_prompt_field_value(text, "Plain competitor")
    if plain_ref:
        normalized_ref = plain_ref.strip().strip("`").lower()
        plain_task = next((candidate for candidate in tasks if candidate.label.lower() == normalized_ref), None)
        if plain_task is None:
            warnings.append(f"{task.label} existing AMD Gluon refinement has unused Plain competitor `{plain_ref}` that does not match this batch; use `Comparison target` instead")
        elif not _is_plain_competitor_task(plain_task):
            warnings.append(f"{task.label} existing AMD Gluon refinement Plain competitor `{plain_ref}` is not a plain Triton task; use `Comparison target` instead")
    lower = text.lower()
    bundled_markers = sum(
        marker in lower
        for marker in (
            "wrapper dispatch",
            "shape dispatch",
            "matrix layout",
            "mfma layout",
            "softmax state",
            "reduction state",
            "partition policy",
            "persistent scheduling",
        )
    )
    if bundled_markers > 2 and "defer_composition" not in lower:
        warnings.append(f"{task.label} appears to bundle high-coupling AMD Gluon components; prefer `Task type: defer_composition` for round 1")
    return warnings


def _is_existing_amd_gluon_feature_meta(feature_meta: dict[str, Any] | None) -> bool:
    feature_meta = feature_meta or {}
    return (
        str(feature_meta.get("input_dialect") or "").strip().lower() == "amd_gluon"
        and str(feature_meta.get("source_origin") or "").strip().lower()
        == SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR
    )


def _is_existing_amd_gluon_plain_or_shared_task(task: AgentTask) -> bool:
    if _is_existing_amd_gluon_refinement_task(task) or _is_gluon_extension_task(task):
        return False
    task_type = _task_type(task)
    if task_type == "defer_composition":
        return False
    required_output = str(task.config.get("required_output_dialect") or "").strip().lower()
    search_set = str(task.config.get("search_set") or "").strip().lower()
    implementation_layer = str(task.config.get("implementation_layer") or "").strip().lower()
    text = f"{task.label}\n{task.task}".lower()
    return (
        required_output == "plain_triton"
        or search_set in {"base", "shared"}
        or "plain_triton" in implementation_layer
        or "base family:" in text
        or "shared source family:" in text
    )


def _is_typed_existing_amd_gluon_plain_or_shared_task(task: AgentTask) -> bool:
    return (
        _is_existing_amd_gluon_plain_or_shared_task(task)
        and _task_type(task) in _TYPED_EXISTING_AMD_GLUON_PLAIN_TASK_TYPES
    )


def _existing_amd_gluon_plain_overflow_summary(task: AgentTask, reason: str) -> dict[str, Any]:
    return {
        "label": task.label,
        "priority": task.priority,
        "drop_reason": reason,
        "task_type": _task_type(task),
        "required_output_dialect": task.config.get("required_output_dialect"),
        "search_set": task.config.get("search_set"),
        "optimization_direction": _parse_prompt_field_value(task.task, "Optimization direction") or "",
        "target_component": task.config.get("target_component") or _parse_prompt_field_value(task.task, "Target component") or "",
    }


def _filter_existing_amd_gluon_plain_overflow(
    tasks: list[AgentTask],
    *,
    feature_meta: dict[str, Any] | None,
    gluon_feature_mode: str | None,
) -> tuple[list[AgentTask], list[dict[str, Any]]]:
    if not _is_existing_amd_gluon_feature_meta(feature_meta):
        return tasks, []
    if not _strict_existing_amd_gluon_portfolio_enabled(gluon_feature_mode):
        return tasks, []

    refinement_tasks = [task for task in tasks if _is_existing_amd_gluon_refinement_task(task)]
    if not refinement_tasks:
        return tasks, []

    plain_or_shared_tasks = [task for task in tasks if _is_existing_amd_gluon_plain_or_shared_task(task)]
    generic_plain_tasks = [
        task
        for task in plain_or_shared_tasks
        if not _is_typed_existing_amd_gluon_plain_or_shared_task(task)
    ]
    if not generic_plain_tasks:
        return tasks, []
    if len(refinement_tasks) > 1 and len(plain_or_shared_tasks) <= 2:
        return tasks, []

    reason = (
        "existing AMD Gluon Gluon-first portfolio dropped generic plain/shared overflow; "
        "plain/shared tasks must be typed as `plain_subkernel_refine` or `shared_or_plain_comparison`, "
        "and the planner should emit fewer tasks instead of padding with generic plain/Base work"
    )
    dropped_labels = {task.label for task in generic_plain_tasks}
    filtered = [task for task in tasks if task.label not in dropped_labels]
    dropped = [_existing_amd_gluon_plain_overflow_summary(task, reason) for task in generic_plain_tasks]
    for task in generic_plain_tasks:
        logger.warning("Dropping existing AMD Gluon generic plain/shared overflow task %s: %s", task.label, reason)
    return filtered, dropped


def _strict_existing_amd_gluon_portfolio_enabled(gluon_feature_mode: str | None) -> bool:
    mode = str(gluon_feature_mode or "").strip().lower()
    return mode in {
        GLUON_FEATURE_MODE_FORCE,
        GLUON_FEATURE_MODE_FORCE_L0_ANCHOR,
        GLUON_FEATURE_MODE_REQUIRE_VIABLE,
    }


def _existing_amd_gluon_portfolio_issues(
    tasks: list[AgentTask],
    *,
    feature_meta: dict[str, Any] | None,
    gluon_feature_mode: str | None,
) -> tuple[list[str], list[str]]:
    if not _is_existing_amd_gluon_feature_meta(feature_meta):
        return [], []

    refinement_tasks = [task for task in tasks if _is_existing_amd_gluon_refinement_task(task)]
    plain_or_shared_tasks = [task for task in tasks if _is_existing_amd_gluon_plain_or_shared_task(task)]
    typed_plain_tasks = [
        task
        for task in plain_or_shared_tasks
        if _is_typed_existing_amd_gluon_plain_or_shared_task(task)
    ]
    generic_plain_tasks = [
        task
        for task in plain_or_shared_tasks
        if not _is_typed_existing_amd_gluon_plain_or_shared_task(task)
    ]
    warnings: list[str] = []

    if not refinement_tasks:
        warnings.append(
            "existing AMD Gluon portfolio has no in-dialect refinement task; "
            "generate at least one `amd_gluon_*_refine` task or report no viable Gluon task"
        )
    if generic_plain_tasks and len(plain_or_shared_tasks) > 2 and len(refinement_tasks) <= 1:
        warnings.append(
            "existing AMD Gluon portfolio contains generic plain/shared overflow: "
            f"{len(generic_plain_tasks)} generic plain/shared task(s), "
            f"{len(typed_plain_tasks)} typed optional plain/shared task(s), and "
            f"{len(refinement_tasks)} refinement task(s); "
            "drop untyped plain/Base padding or convert real fallback/plain subkernel work to "
            "`Task type: plain_subkernel_refine|shared_or_plain_comparison`"
        )
    elif len(plain_or_shared_tasks) > 2 and len(refinement_tasks) <= 1:
        warnings.append(
            "existing AMD Gluon portfolio has more optional plain/shared tasks than recommended: "
            f"{len(plain_or_shared_tasks)} typed optional plain/shared task(s) and {len(refinement_tasks)} refinement task(s); "
            "prefer fewer tasks or more in-dialect refinements for Gluon-priority runs"
        )

    untyped_plain = [
        task.label
        for task in generic_plain_tasks
    ]
    if untyped_plain:
        warnings.append(
            "existing AMD Gluon plain/shared tasks should declare `Task type: plain_subkernel_refine` "
            f"or `shared_or_plain_comparison`: {', '.join(untyped_plain)}"
        )

    if not _strict_existing_amd_gluon_portfolio_enabled(gluon_feature_mode):
        return [], warnings

    hard_errors = [
        warning
        for warning in warnings
        if "no in-dialect refinement" in warning or "generic plain/shared overflow" in warning
    ]
    return hard_errors, warnings


def _is_hard_task_generation_extension_error(error: str) -> bool:
    """Return whether an Extension audit issue should reject the whole batch.

    Planner-time audit should catch structurally invalid work that would waste a
    dispatch slot. Missing advisory routing fields, over-specific L0 metadata,
    or weak same-component evidence are left for normalization, warnings, worker
    doc routing, save_and_test, and selector checks.
    """
    text = str(error or "")
    hard_markers = (
        "expected at most",
        "single AMD Gluon Extension task must be an L0",
        "missing per-task overlay binding field: Plain competitor",
        "does not match a task label in this batch",
        "is not a plain Triton competitor task",
        "cannot require AMD Gluon output when",
        "has invalid L0 scope classification",
        "has invalid Minimum executable unit",
        "has invalid Allowed execution path",
        "has invalid Scope infeasible policy",
        "Minimum executable unit `",
        "allows whole_jit_kernel but forbids whole-kernel rewrite",
    )
    return any(marker in text for marker in hard_markers)


def _audit_base_family_coverage(
    tasks: list[AgentTask],
    required_families: list[str],
    *,
    expected_extension_slots: int | None = None,
    feature_meta: dict[str, Any] | None = None,
    gluon_feature_mode: str | None = None,
) -> None:
    """Reject task plans that drop mandatory Base families or over-expand Gluon."""
    existing_amd_gluon_feature = _is_existing_amd_gluon_feature_meta(feature_meta)
    if not required_families and expected_extension_slots is None and not existing_amd_gluon_feature:
        return

    missing = [
        family
        for family in required_families
        if not any(_is_plain_competitor_task(task) and _task_matches_base_family(task, family) for task in tasks)
    ]
    extension_tasks = [task for task in tasks if _is_gluon_extension_task(task)]
    existing_refinement_tasks = [task for task in tasks if _is_existing_amd_gluon_refinement_task(task)]
    extension_errors: list[str] = []
    if expected_extension_slots is not None and len(extension_tasks) > expected_extension_slots:
        extension_errors.append(
            f"expected at most {expected_extension_slots} AMD Gluon Extension task(s), found {len(extension_tasks)}"
        )
    if expected_extension_slots == 1 and len(extension_tasks) == 1 and not _is_l0_gluon_overlay_task(extension_tasks[0]):
        extension_errors.append("single AMD Gluon Extension task must be an L0 AMD Gluon overlay")
    for task in extension_tasks:
        extension_errors.extend(_gluon_overlay_contract_errors(task))
        extension_errors.extend(_gluon_overlay_priority_errors(task))
        extension_errors.extend(_gluon_l0_execution_boundary_errors(task))
        extension_errors.extend(_gluon_overlay_binding_errors(task, tasks))
        extension_errors.extend(_l1_anchor_contract_errors(task))
        extension_errors.extend(_required_gluon_soft_diagnostic_errors(task))
    existing_refinement_errors: list[str] = []
    for task in existing_refinement_tasks:
        existing_refinement_errors.extend(_existing_amd_gluon_refinement_contract_errors(task))
        for warning in _existing_amd_gluon_refinement_warnings(task, tasks):
            logger.warning("Task-generation existing AMD Gluon refinement warning: %s", warning)
    existing_portfolio_errors, existing_portfolio_warnings = _existing_amd_gluon_portfolio_issues(
        tasks,
        feature_meta=feature_meta,
        gluon_feature_mode=gluon_feature_mode,
    )
    for warning in existing_portfolio_warnings:
        logger.warning("Task-generation existing AMD Gluon portfolio warning: %s", warning)

    hard_extension_errors = [
        error for error in extension_errors if _is_hard_task_generation_extension_error(error)
    ]
    soft_extension_errors = [
        error for error in extension_errors if not _is_hard_task_generation_extension_error(error)
    ]
    for error in soft_extension_errors:
        logger.warning("Task-generation Extension audit warning: %s", error)

    errors = []
    if missing:
        errors.append("missing Base Set mandatory families: " + ", ".join(missing))
    errors.extend(hard_extension_errors)
    errors.extend(existing_refinement_errors)
    errors.extend(existing_portfolio_errors)
    if not errors:
        return

    message = "Task-generation coverage audit failed: " + "; ".join(errors)
    if hard_extension_errors:
        raise ValueError(message)
    if _strict_base_family_audit_enabled():
        raise ValueError(message)
    logger.warning(message)


def _task_audit_summary(task: AgentTask) -> dict[str, Any]:
    text = f"{task.label}\n{task.task}"
    return {
        "label": task.label,
        "priority": task.priority,
        "search_set": task.config.get("search_set"),
        "required_output_dialect": task.config.get("required_output_dialect"),
        "source_origin": task.config.get("source_origin"),
        "gluon_tl_policy": task.config.get("gluon_tl_policy"),
        "layout_construction_policy": task.config.get("layout_construction_policy"),
        "execution_mode": task.config.get("execution_mode"),
        "base_family": _parse_prompt_field_value(task.task, "Base family"),
        "source_base_family": task.config.get("source_base_family") or _parse_prompt_field_value(text, "Source Base family"),
        "plain_competitor": task.config.get("plain_competitor") or _parse_prompt_field_value(text, "Plain competitor"),
        "implementation_layer": task.config.get("implementation_layer") or _parse_prompt_field_value(text, "Implementation layer"),
        "extension_layer": task.config.get("extension_layer") or _parse_prompt_field_value(text, "Extension layer"),
        "task_type": task.config.get("task_type") or _parse_prompt_field_value(text, "Task type"),
        "primary_component": task.config.get("primary_component")
        or task.config.get("primary_atomic_component")
        or _parse_prompt_field_value(text, "Primary component")
        or _parse_prompt_field_value(text, "Primary atomic component"),
        "optimization_direction": _parse_prompt_field_value(text, "Optimization direction"),
        "target_symbol": task.config.get("target_symbol") or _parse_prompt_field_value(text, "Target symbol"),
        "target_component": task.config.get("target_component") or _parse_prompt_field_value(text, "Target component"),
        "l0_scope_classification": task.config.get("l0_scope_classification")
        or _parse_prompt_field_value(text, "L0 scope classification")
        or "",
        "l0_coupling_reasons": task.config.get("l0_coupling_reasons")
        or _parse_prompt_field_value(text, "L0 coupling reasons")
        or "",
        "expected_failure_layers": task.config.get("expected_failure_layers")
        or _parse_prompt_field_value(text, "Expected failure layers")
        or "",
        "first_patch_compile_goal": task.config.get("first_patch_compile_goal")
        or _parse_prompt_field_value(text, "First patch compile goal")
        or "",
        "task_signals": task.config.get("task_signals") or _parse_prompt_field_value(text, "Task signals") or "",
        "routed_doc_reasons": task.config.get("routed_doc_reasons")
        or _parse_prompt_field_value(text, "Routed doc reasons")
        or "",
        "kernel_family_signal": task.config.get("kernel_family_signal")
        or _parse_prompt_field_value(text, "Kernel family signal")
        or "",
        "failure_layers": task.config.get("failure_layers") or _parse_prompt_field_value(text, "Failure layers") or "",
        "minimum_executable_unit": task.config.get("minimum_executable_unit")
        or _parse_prompt_field_value(text, "Minimum executable unit")
        or "",
        "allowed_execution_path": task.config.get("allowed_execution_path")
        or _parse_prompt_field_value(text, "Allowed execution path")
        or "",
        "scope_infeasible_policy": task.config.get("scope_infeasible_policy")
        or _parse_prompt_field_value(text, "Scope infeasible policy")
        or "",
        "required_patch_target_symbols": task.config.get("required_patch_target_symbols") or [],
        "forbidden_patch_target_symbols": _task_forbidden_symbols(task),
        "soft_audit_diagnostics": _gluon_l0_soft_diagnostics(task),
        "first_lines": "\n".join(task.task.splitlines()[:18]),
    }


def _task_label_map(tasks: list[AgentTask]) -> dict[str, dict[str, Any]]:
    return {_task_audit_summary(task)["label"]: _task_audit_summary(task) for task in tasks}


def _audit_repair_hints(errors: list[str], tasks: list[AgentTask]) -> list[str]:
    """Return planner-facing repair hints for common task-generation audit errors."""
    summaries = _task_label_map(tasks)
    hints: list[str] = []
    inline_high_coupling_labels = {
        match.group("label")
        for error in errors
        if (
            match := re.search(
                r"(?:Task-generation coverage audit failed:\s*)?(?P<label>\S+) cannot use inline_scoped_helper for high-coupling L0 target",
                error,
            )
        )
    }
    for summary in summaries.values():
        label = summary.get("label") or "<task>"
        diagnostics = summary.get("soft_audit_diagnostics") or []
        if label in inline_high_coupling_labels:
            hints.append(
                f"{label}: `inline_scoped_helper` was rejected because the declared L0 target is high-coupling. "
                "Do not fix this by broadening a local smoke task in place. Either shrink to a genuinely "
                "low-coupling primary component with a same-component Base anchor, or skip the Gluon L0 overlay "
                "until a real whole-helper/stage Base direction exists."
            )
        if "local_target_promoted_to_whole_kernel" in diagnostics:
            hints.append(
                f"{label}: local target appears promoted to `whole_jit_kernel`. Choose exactly one L0 branch: "
                "Branch A local smoke/probe with `inline_scoped_helper`, or Branch B retargeted to a whole-helper "
                "layout skeleton with `whole_kernel_required_reason` and compile-risk fields. Do not keep local "
                "load/store/index/mask/path wording while declaring `whole_jit_kernel`."
            )
        if "matrix_metadata_inconsistent" in diagnostics:
            hints.append(
                f"{label}: matrix/dot signals are present but matrix metadata is inconsistent; set `matrix_lowering_required: true` for whole-kernel/matrix scope or shrink to a non-matrix smoke path."
            )
        if "broadcast_heavy_missing_layout_map" in diagnostics:
            hints.append(
                f"{label}: broadcast-heavy whole-kernel L0 should include parent-layout map / broadcast failure layers, or shrink to an index/mask/load smoke path."
            )
        if "reduction_metadata_inconsistent" in diagnostics:
            hints.append(
                f"{label}: reduction or accumulator signals are present; include reduction/accumulator failure layers or shrink away from the reduction path."
            )
        if "scale_layout_missing" in diagnostics:
            hints.append(
                f"{label}: scale/dtype signals are present; include scale layout, dtype, operand/result layout failure layers or keep the first patch as a scale-layout smoke path."
            )
        if "component_bundle_too_broad" in diagnostics:
            hints.append(
                f"{label}: multiple patch-target atomic components appear bundled; choose one primary component for "
                "`patch_0` and list the rest as secondary components/blockers or failure layers unless "
                "`bundle_allowed=true`."
            )
        if "unknown_family_whole_kernel" in diagnostics:
            hints.append(
                f"{label}: unknown or weak kernel-family classification should not default to whole-kernel Gluon; "
                "prefer Base/plain Triton or Branch A local index/load/layout smoke/probe."
            )
    for error in errors:
        if "existing AMD Gluon refinement" in error:
            match = re.search(r"(?:Task-generation coverage audit failed:\s*)?(?P<label>\S+) existing AMD Gluon refinement", error)
            label = match.group("label") if match else "<task>"
            hints.append(
                f"{label}: for measured production AMD Gluon input, emit an in-dialect refinement contract instead of a generated overlay contract: "
                "`source_origin: existing_amd_gluon_operator`, `Task type: amd_gluon_in_dialect_refine|amd_gluon_layout_or_matrix_refine|amd_gluon_shape_dispatch_refine`, "
                "`Implementation layer: amd_gluon in-dialect refinement`, one `Target component`, one `Allowed change`, "
                "`Comparison target: true_baseline|safe_anchor`, `Failure layers`, and `Reject if: correctness fails or any benchmark shape regresses`."
            )
            continue

        direction_match = re.search(
            r"(?:Task-generation coverage audit failed:\s*)?(?P<label>\S+) Optimization direction `(?P<gluon>.*?)` does not match Plain competitor `(?P<plain>.*?)` direction `(?P<plain_dir>.*?)`$",
            error,
        )
        if direction_match:
            label = direction_match.group("label")
            plain = direction_match.group("plain")
            gluon_direction = direction_match.group("gluon")
            summary = summaries.get(label, {})
            target_component = summary.get("target_component") or "<same target component as the Gluon L0>"
            source_family = summary.get("source_base_family") or "matching Base family"
            diagnostics = summary.get("soft_audit_diagnostics") or []
            if "local_target_promoted_to_whole_kernel" in diagnostics or label in inline_high_coupling_labels:
                hints.append(
                    f"{label}: current L0 overlay is neither exact local Branch A nor retargeted Branch B. "
                    f"Either create an exact same-component Base competitor for `{target_component}` with "
                    f"`Optimization direction: {gluon_direction}` and keep the Gluon task as Branch A "
                    "(`minimum_executable_unit: inline_scoped_helper`) only if the target is low-coupling, or "
                    "retarget the Gluon task to a whole helper/stage skeleton with `whole_kernel_required_reason` "
                    "and compile-risk fields. Do not bind a narrow or high-coupling local target to broad "
                    f"competitor `{plain}`."
                )
            hints.append(
                f"{label}: create a same-batch plain Triton Base competitor for optimization direction "
                f"`{gluon_direction}` and target component `{target_component}`, then set `Plain competitor` "
                f"to that new Base task, or reuse `{plain}` only if the Gluon task adopts that exact shared "
                "`Optimization direction:`. Keep Gluon mechanisms such as explicit layouts, DotOperandLayout, "
                f"or buffer ops in `Gluon overlay reason` / `Performance hypothesis`. The plain task should use "
                f"`Base family: {source_family}`, `required_output_dialect: plain_triton`, and the exact same "
                "`Optimization direction:` text, with a scoped plain Triton no-regression performance hypothesis."
            )
            continue

        component_match = re.search(
            r"(?:Task-generation coverage audit failed:\s*)?(?P<label>\S+) Target component `(?P<gluon>.*?)` does not match Plain competitor `(?P<plain>.*?)` component `(?P<plain_component>.*?)`$",
            error,
        )
        if component_match:
            label = component_match.group("label")
            plain = component_match.group("plain")
            gluon_component = component_match.group("gluon")
            summary = summaries.get(label, {})
            direction = summary.get("optimization_direction") or "<same optimization direction as the Gluon L0>"
            hints.append(
                f"{label}: bind to a plain Base task whose `Target component` / `Allowed change` names "
                f"`{gluon_component}` and whose `Optimization direction:` is `{direction}`. "
                f"If the existing competitor `{plain}` is intentionally local, shrink the Gluon L0 to that "
                "local component instead of using a broader helper target."
            )
            continue

        scope_match = re.search(
            r"(?:Task-generation coverage audit failed:\s*)?(?P<label>\S+) Target scope `(?P<gluon>.*?)` does not match Plain competitor `(?P<plain>.*?)` target scope `(?P<plain_scope>.*?)`$",
            error,
        )
        if scope_match:
            label = scope_match.group("label")
            plain = scope_match.group("plain")
            summary = summaries.get(label, {})
            direction = summary.get("optimization_direction") or "<same optimization direction as the Gluon L0>"
            target_component = summary.get("target_component") or summary.get("target_symbol") or "<Gluon target symbol/component>"
            source_family = summary.get("source_base_family") or "matching Base family"
            hints.append(
                f"{label}: create a same-batch plain Triton Base competitor for target `{target_component}` "
                f"with `Base family: {source_family}`, `required_output_dialect: plain_triton`, "
                f"and `Optimization direction: {direction}`. If `{plain}` is the desired anchor, shrink the "
                "Gluon task to the same stage/helper/component rather than binding a whole-helper overlay to a local task."
            )
            continue

        missing_component_match = re.search(
            r"(?:Task-generation coverage audit failed:\s*)?(?P<label>\S+) Plain competitor `(?P<plain>.*?)` lacks auditable Target component or Allowed change",
            error,
        )
        if missing_component_match:
            label = missing_component_match.group("label")
            plain = missing_component_match.group("plain")
            summary = summaries.get(label, {})
            direction = summary.get("optimization_direction") or "<same optimization direction as the Gluon L0>"
            source_family = summary.get("source_base_family") or "matching Base family"
            target_component = summary.get("target_component") or "<Gluon target component>"
            diagnostics = summary.get("soft_audit_diagnostics") or []
            if "local_target_promoted_to_whole_kernel" in diagnostics or label in inline_high_coupling_labels:
                hints.append(
                    f"{label}: current L0 overlay has neither an exact local Branch A Base anchor nor a "
                    "retargeted Branch B whole-helper skeleton. Either create/fix a same-batch plain Base "
                    f"anchor for target `{target_component}` with `Optimization direction: {direction}` and "
                    "keep the Gluon task local with `minimum_executable_unit: inline_scoped_helper` only when "
                    "the target is low-coupling, or retarget both the Base anchor and Gluon overlay to the same "
                    "whole helper/stage skeleton before using a whole-helper execution path. If that Base anchor "
                    "would be broad, speculative, or performance-neutral, do not emit the Gluon L0 overlay."
                )
            hints.append(
                f"{label}: `{plain}` is too broad to audit as an L0 plain competitor. "
                f"Either update it to include `Target component: {target_component}` and a scoped `Allowed change:` "
                f"with the exact same `Optimization direction: {direction}`, or create a new same-batch plain Base task "
                f"with `Base family: {source_family}`, `required_output_dialect: plain_triton`, "
                "that exact optimization direction, the same target component, and a credible plain Triton "
                "no-regression performance hypothesis. Do not create a Base anchor solely for Gluon."
            )
            continue

        if "Plain competitor `" in error and "does not match a task label" in error:
            hints.append(
                "Add the referenced plain Triton Base competitor to the same submitted batch, or remove the L0 overlay for this round."
            )
        elif "is not a plain Triton competitor task" in error:
            hints.append(
                "Point `Plain competitor:` at a Base task with `required_output_dialect=plain_triton`, not at Shared, Gluon, mixed, or hybrid tasks."
            )
        elif "missing L0 execution-boundary field" in error:
            hints.append(
                "Add parseable L0 execution-boundary fields to the Round-1 L0 overlay, preferably as top-level JSON metadata or exact task lines: `Minimum executable unit: ...`, `Allowed execution path: ...`, and `Scope infeasible policy: ...`. Snake_case prompt lines are accepted for compatibility, but title-case fields are preferred for audit readability. If the scoped path cannot execute, do not emit the overlay."
            )
        elif "missing L0 planner routing field" in error:
            hints.append(
                "Add planner routing metadata to the Gluon task: `task_signals`, `routed_doc_reasons`, `kernel_family_signal`, and `failure_layers`. Use these to map task signals to required split docs before emitting the task."
            )
        elif "whole_jit_kernel compile-risk anchor missing fields" in error:
            hints.append(
                "For `whole_jit_kernel`, add compile-risk metadata: `expected_failure_layers`, `first_patch_compile_goal`, `do_not_optimize_before_compile: true`, and `matrix_lowering_required: true|false`; otherwise shrink to a micro-anchor."
            )

    unique: list[str] = []
    for hint in hints:
        if hint not in unique:
            unique.append(hint)
    return unique


def _write_task_generation_audit_failure(
    *,
    diagnostics_dir: Path | None,
    raw_content: str,
    raw_tasks: list[Any],
    parsed_tasks: list[AgentTask],
    error: Exception,
    required_base_families: list[str],
    expected_extension_slots: int | None,
) -> None:
    if diagnostics_dir is None:
        env_dir = os.getenv("GEAK_TASKGEN_AUDIT_DUMP_DIR")
        diagnostics_dir = Path(env_dir) if env_dir else None
    if diagnostics_dir is None:
        return
    try:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / f"task_generation_audit_failed_{int(time.time() * 1000)}.json"
        payload = {
            "error": str(error),
            "required_base_families": list(required_base_families),
            "expected_extension_slots": expected_extension_slots,
            "raw_submitted_json": raw_content,
            "raw_tasks": raw_tasks,
            "parsed_task_summaries": [_task_audit_summary(task) for task in parsed_tasks],
            "repair_hints": _audit_repair_hints([part.strip() for part in str(error).split(";")], parsed_tasks),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.error("Task-generation audit failure diagnostics written to %s", path)
    except Exception as exc:
        logger.warning("Could not write task-generation audit failure diagnostics: %s", exc)


def _normalize_existing_amd_gluon_task_config(
    cfg: dict[str, Any],
    *,
    feature_meta: dict[str, Any] | None,
    task_prompt: str,
) -> None:
    """Fill conservative defaults for measured production AMD Gluon refinements."""
    feature_meta = feature_meta or {}
    if str(feature_meta.get("source_origin") or "").strip().lower() != SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR:
        return
    if str(feature_meta.get("input_dialect") or "").strip().lower() != "amd_gluon":
        return
    required_output = str(cfg.get("required_output_dialect") or "").strip().lower()
    if required_output != "amd_gluon":
        return
    if str(cfg.get("source_origin") or "").strip().lower() in {
        SOURCE_ORIGIN_GENERATED_OVERLAY,
        SOURCE_ORIGIN_NV_GLUON_TRANSLATION,
    }:
        return
    cfg.setdefault("source_origin", SOURCE_ORIGIN_EXISTING_AMD_GLUON_OPERATOR)
    cfg.setdefault("gluon_tl_policy", "production_source_allowed")
    cfg.setdefault("layout_construction_policy", "source_preserve")
    implementation_layer = str(cfg.get("implementation_layer") or "").strip()
    if not implementation_layer or implementation_layer.lower() in {"amd_gluon", "amd gluon"}:
        cfg["implementation_layer"] = "amd_gluon in-dialect refinement"
    cfg.setdefault("task_type", _infer_existing_amd_gluon_task_type(task_prompt))


def _infer_existing_amd_gluon_task_type(task_prompt: str) -> str:
    text = str(task_prompt or "").lower()
    if any(marker in text for marker in ("shape dispatch", "shape_bucket", "partition", "ps path", "persistent scheduling", "wrapper")):
        return "amd_gluon_shape_dispatch_refine"
    if any(marker in text for marker in ("matrix", "mfma", "wmma", "dotoperandlayout", "warp layout", "layout")):
        return "amd_gluon_layout_or_matrix_refine"
    return "amd_gluon_in_dialect_refine"


def _parse_llm_response(
    content: str,
    agent_class: type,
    *,
    kernel_path: str | None = None,
    commandment_path: str | None = None,
    baseline_metrics_path: str | None = None,
    required_base_families: list[str] | None = None,
    expected_extension_slots: int | None = None,
    audit_diagnostics_dir: Path | None = None,
    gluon_feature_mode: str | None = None,
    feature_meta: dict[str, Any] | None = None,
) -> list[AgentTask]:
    """Parse JSON response into AgentTask objects."""
    _set_last_gluon_task_generation_diagnostics({})
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

        search_set, required_output_dialect = _infer_search_set_and_required_output(label, task_prompt, item)
        cfg: dict[str, Any] = {
            "search_set": search_set,
            "required_output_dialect": required_output_dialect,
        }
        source_base_family = _parse_prompt_field_value(task_prompt, "Source Base family")
        plain_competitor = _parse_prompt_field_value(task_prompt, "Plain competitor")
        implementation_layer = _parse_prompt_field_value(task_prompt, "Implementation layer")
        extension_layer = _parse_prompt_field_value(task_prompt, "Extension layer")
        if source_base_family:
            cfg["source_base_family"] = source_base_family.strip().strip("`")
        if plain_competitor:
            cfg["plain_competitor"] = plain_competitor.strip().strip("`")
        if implementation_layer:
            cfg["implementation_layer"] = implementation_layer.strip().strip("`")
        if extension_layer:
            cfg["extension_layer"] = extension_layer.strip().strip("`")
        gluon_doc_profile = str(item.get("gluon_doc_profile") or "").strip().lower()
        if not gluon_doc_profile:
            gluon_doc_profile = _infer_gluon_doc_profile(label, task_prompt, search_set, required_output_dialect)
        cfg_with_profile = {**cfg, "gluon_doc_profile": gluon_doc_profile}
        if _task_config_requires_gluon_worker_docs(
            kernel_type="triton",
            cfg=cfg_with_profile,
            task_body=task_prompt,
            label=label,
        ):
            required_gluon_docs = _infer_required_gluon_doc_keys(
                label,
                task_prompt,
                search_set,
                required_output_dialect,
                gluon_doc_profile,
            )
            raw_required_docs = item.get("required_gluon_docs")
            if isinstance(raw_required_docs, str):
                explicit_docs = [part.strip() for part in raw_required_docs.split(",") if part.strip()]
            elif isinstance(raw_required_docs, list):
                explicit_docs = [str(part).strip() for part in raw_required_docs if str(part).strip()]
            else:
                explicit_docs = []
            for doc_key in explicit_docs:
                add_unique_doc_key(required_gluon_docs, doc_key)
            cfg["gluon_doc_profile"] = gluon_doc_profile
            cfg["required_gluon_docs"] = required_gluon_docs
        required_patch_target_symbols = _infer_required_patch_target_symbols(task_prompt, item)
        if required_patch_target_symbols:
            cfg["required_patch_target_symbols"] = required_patch_target_symbols
        executed_route_symbols = _infer_executed_route_symbols(
            task_prompt,
            item,
            required_patch_target_symbols=required_patch_target_symbols,
        )
        if executed_route_symbols:
            cfg["executed_route_symbols"] = executed_route_symbols
        for key, tag in _OPTIONAL_GLUON_TASK_METADATA_FIELDS:
            value = _optional_task_metadata_from_item_or_prompt(item, task_prompt, key, tag)
            if value not in (None, ""):
                cfg[key] = value
        _normalize_existing_amd_gluon_task_config(
            cfg,
            feature_meta=feature_meta,
            task_prompt=task_prompt,
        )

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

    tasks, dropped_gluon_overlays = _filter_low_priority_l0_overlays(
        tasks,
        gluon_feature_mode=gluon_feature_mode,
    )
    tasks, dropped_existing_amd_plain_overflow = _filter_existing_amd_gluon_plain_overflow(
        tasks,
        feature_meta=feature_meta,
        gluon_feature_mode=gluon_feature_mode,
    )
    extension_tasks_after_filter = [
        task
        for task in tasks
        if _is_gluon_extension_task(task) or _is_existing_amd_gluon_refinement_task(task)
    ]
    if _mode_requires_viable_gluon(gluon_feature_mode) and not extension_tasks_after_filter:
        _write_task_generation_gluon_diagnostics(
            diagnostics_dir=audit_diagnostics_dir,
            dropped_gluon_overlays=dropped_gluon_overlays,
            dropped_existing_amd_plain_overflow=dropped_existing_amd_plain_overflow,
            no_viable_gluon_task={
                "reason": "require_viable_gluon mode found no dispatchable AMD Gluon task",
                "dropped_gluon_overlay_count": len(dropped_gluon_overlays),
            },
        )
        return []
    _write_task_generation_gluon_diagnostics(
        diagnostics_dir=audit_diagnostics_dir,
        dropped_gluon_overlays=dropped_gluon_overlays,
        dropped_existing_amd_plain_overflow=dropped_existing_amd_plain_overflow,
    )
    tasks = _ensure_mandatory_plain_families(
        tasks,
        required_base_families or [],
        agent_class=agent_class,
    )
    sorted_tasks = sorted(tasks, key=lambda t: t.priority)
    try:
        _audit_base_family_coverage(
            sorted_tasks,
            required_base_families or [],
            expected_extension_slots=expected_extension_slots,
            feature_meta=feature_meta,
            gluon_feature_mode=gluon_feature_mode,
        )
    except ValueError as exc:
        _write_task_generation_audit_failure(
            diagnostics_dir=audit_diagnostics_dir,
            raw_content=content,
            raw_tasks=raw_tasks,
            parsed_tasks=sorted_tasks,
            error=exc,
            required_base_families=required_base_families or [],
            expected_extension_slots=expected_extension_slots,
        )
        raise
    return sorted_tasks


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
        audit_diagnostics_dir=Path(args.output) if args.output else None,
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
