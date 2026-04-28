"""Orchestrator dispatcher: heterogeneous-only thin shim.

This module exposes :func:`run_orchestrator` for callers that have already
run preprocess (notably ``minisweagent.run.mini``).  All heavy lifting
lives in ``agents.heterogeneous.orchestrator`` and the shared postprocess
package (``run.postprocess.evaluation``, ``run.postprocess.results``).

Note: the historical ``geak-orchestrate`` console script that wrapped
this module has been removed.  Its semantics for ``--mode`` diverged
from ``geak`` (preprocess elapsed was always assumed to be 0, so
``--mode quick`` from the standalone CLI gave a *fresh* 1h optimization
budget instead of the 1h preprocess+optimization budget the same flag
gives via ``geak``).  Rather than carry a second flavour of the budget
contract, the entry point was deleted; use ``geak`` end-to-end and rely
on its in-flight resume support if you need to re-enter mid-run.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from minisweagent.run.pipeline_helpers import (
    DEFAULT_HETEROGENEOUS,
    DEFAULT_PIPELINE_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def _max_rounds_source(max_rounds: int, env_value: str | None) -> str:
    """Return 'arg', 'env', or 'default' indicating where max_rounds came from."""
    default = int(env_value) if env_value else 5
    if max_rounds != default:
        return "arg"
    return "env" if env_value else "default"


def run_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    model,
    model_factory,
    *,
    output_dir: Path | None = None,
    max_rounds: int | None = None,
    start_round: int = 1,
    heterogeneous: bool = DEFAULT_HETEROGENEOUS,
    deadline=None,
    soft_stop=None,
    registry=None,
) -> dict[str, Any]:
    """Run the orchestrator agent loop.

    Parameters
    ----------
    preprocess_ctx:
        Context dict returned by ``run_preprocessor()``.
    gpu_ids:
        List of GPU device IDs available for task execution.
    model:
        LLM model instance for the orchestrator.
    model_factory:
        Callable returning a new model instance (for sub-agents).
    output_dir:
        Override output directory (defaults to preprocess_ctx source).
    max_rounds:
        Maximum optimisation rounds (default: from GEAK_MAX_ROUNDS env or 5).
    start_round:
        Round number to start from (1-based, default 1).
    heterogeneous:
        If True, use LLM-generated diverse tasks per round.
        If False (default), homogeneous mode is handled directly by
        ``mini`` and is not supported here.
    deadline:
        Optional ``run.budget.Deadline`` snapshot. Threaded into the round loop
        and ``run_llm_steps`` so per-step polls can short-circuit cleanly.
    soft_stop:
        Optional ``threading.Event``. Set ~``finalize_grace_s`` before the
        deadline by the optimization watchdog; orchestrator polls it to stop
        dispatching new work and finalize with best-so-far.
    registry:
        Optional ``run.state.ProcessRegistry``. Sub-agent dispatchers register
        their ``Popen`` / ``Future`` here so the watchdog and SIGINT handler
        can ``terminate_all()``.
    """
    _out = output_dir or Path(preprocess_ctx.get("output_dir", DEFAULT_PIPELINE_OUTPUT_DIR))
    _out = Path(_out)
    _out.mkdir(parents=True, exist_ok=True)

    _env_rounds = os.getenv("GEAK_MAX_ROUNDS")
    max_rounds = max_rounds or int(_env_rounds or "5")
    logger.info(
        "run_orchestrator: output_dir=%s, max_rounds=%d (source=%s), start_round=%d, heterogeneous=%s",
        _out,
        max_rounds,
        _max_rounds_source(max_rounds, _env_rounds),
        start_round,
        heterogeneous,
    )

    if not heterogeneous:
        raise NotImplementedError(
            "Homogeneous mode is not supported via run_orchestrator(); "
            "the homogeneous code path is invoked directly from the geak CLI."
        )

    from minisweagent.agents.heterogeneous.orchestrator import run_heterogeneous_orchestrator

    return run_heterogeneous_orchestrator(
        preprocess_ctx,
        gpu_ids,
        model,
        model_factory,
        _out,
        max_rounds,
        start_round,
        deadline=deadline,
        soft_stop=soft_stop,
        registry=registry,
    )


def _probe_preprocess_dir(pp_dir: Path):
    """Backward-compatible fallback: reconstruct PreprocessContext by probing files.

    Retained for tests and for any tool that consumes a preprocess artefact
    directory without a ``preprocess_context.json`` manifest.  Not part of
    a CLI flow anymore.
    """
    from minisweagent.run.pipeline_types import PreprocessContext
    from minisweagent.run.preprocess.discovery_types import build_gluon_feature_metadata

    logger.debug("_probe_preprocess_dir: probing %s for preprocessor artefacts.", pp_dir)
    kernel_path = ""
    repo_root = str(pp_dir)
    harness_path = ""

    resolved_path = pp_dir / "resolved.json"
    if resolved_path.exists():
        resolved = json.loads(resolved_path.read_text())
        kernel_path = resolved.get("local_file_path", "")
        repo_path = resolved.get("local_repo_path")
        if kernel_path:
            kp = Path(kernel_path).resolve()
            git_root = None
            cur = kp if kp.is_dir() else kp.parent
            while cur != cur.parent:
                if (cur / ".git").exists():
                    git_root = cur
                    break
                cur = cur.parent
            if git_root:
                repo_root = str(git_root)
                logger.debug("_probe_preprocess_dir: repo_root from git walk: %s", repo_root)
            elif repo_path:
                repo_root = repo_path
                logger.debug("_probe_preprocess_dir: repo_root from resolved.json: %s", repo_root)
            else:
                repo_root = str(Path(kernel_path).parent)
                logger.debug("_probe_preprocess_dir: repo_root defaulted to kernel parent: %s", repo_root)

    testcase_sel_path = pp_dir / "testcase_selection.json"
    if testcase_sel_path.exists():
        try:
            ts = json.loads(testcase_sel_path.read_text())
            if isinstance(ts, dict):
                harness_path = ts.get("harness_path", "")
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("_probe_preprocess_dir: failed to read testcase_selection.json: %s", exc)

    discovery = None
    discovery_path = pp_dir / "discovery.json"
    if discovery_path.exists():
        try:
            discovery = json.loads(discovery_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("_probe_preprocess_dir: failed to read discovery.json: %s", exc)

    _kernel_dict = (discovery or {}).get("kernel") or {}
    # When resolved.json is missing (e.g. an older preprocess output or a
    # manually-curated preprocess_dir) fall back to ``discovery["kernel"]["file"]``
    # so the kernel-type re-inference path below still has source to read.
    # Otherwise Gluon would silently default to ``off`` on disk-resume runs
    # where only discovery.json exists.
    if not kernel_path:
        _disc_file = _kernel_dict.get("file") or ""
        if _disc_file:
            kernel_path = str(_disc_file)
            logger.debug("_probe_preprocess_dir: kernel_path defaulted from discovery['kernel']['file']: %s", kernel_path)
            if not repo_root or repo_root == str(pp_dir):
                _kp = Path(kernel_path).resolve()
                _cur = _kp if _kp.is_dir() else _kp.parent
                while _cur != _cur.parent:
                    if (_cur / ".git").exists():
                        repo_root = str(_cur)
                        logger.debug(
                            "_probe_preprocess_dir: repo_root re-derived from discovery file: %s",
                            repo_root,
                        )
                        break
                    _cur = _cur.parent
    _baseline_for_probe: dict = {}
    _bm_path = pp_dir / "baseline_metrics.json"
    if _bm_path.exists():
        try:
            _baseline_for_probe = json.loads(_bm_path.read_text())
        except (OSError, ValueError) as exc:
            logger.debug("_probe_preprocess_dir: could not parse baseline_metrics.json: %s", exc)
    baseline_cases = (
        _baseline_for_probe.get("benchmark_test_cases")
        if isinstance(_baseline_for_probe, dict)
        else None
    )
    baseline_has_authoritative_cases = bool(baseline_cases)
    shape_count = (
        _baseline_for_probe.get("benchmark_shape_count")
        if baseline_has_authoritative_cases
        else _kernel_dict.get("benchmark_shape_count") or _baseline_for_probe.get("benchmark_shape_count")
    )
    shape_cases = (
        baseline_cases
        if baseline_has_authoritative_cases
        else _kernel_dict.get("benchmark_test_cases") or _baseline_for_probe.get("benchmark_test_cases")
    )
    shape_profile = (
        _baseline_for_probe.get("shape_coverage_profile")
        if baseline_has_authoritative_cases
        else _kernel_dict.get("shape_coverage_profile") or _baseline_for_probe.get("shape_coverage_profile")
    )
    # Re-infer kernel_type from source content when discovery says
    # "unknown" / "" -- otherwise build_gluon_feature_metadata defaults
    # gluon_feature_mode to "off" and the resume-from-disk path silently
    # disables Gluon. Mirrors task_generator._extract_kernel_meta which
    # already does this on the orchestrator-driven path.
    _raw_type = str(_kernel_dict.get("type") or "").strip().lower()
    if _raw_type in {"", "unknown"} and kernel_path:
        try:
            from minisweagent.agents.heterogeneous.task_generator import (
                _infer_kernel_type as _infer_task_kernel_type,
            )

            _resolved_type = _infer_task_kernel_type(Path(kernel_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("_probe_preprocess_dir: _infer_kernel_type failed: %s", exc)
            _resolved_type = _raw_type or "unknown"
    else:
        _resolved_type = _raw_type or "unknown"
    feature_meta = build_gluon_feature_metadata(
        Path(kernel_path) if kernel_path else Path("unknown.py"),
        _resolved_type,
        input_dialect=_kernel_dict.get("input_dialect"),
        gluon_feature_mode=_kernel_dict.get("gluon_feature_mode"),
        gluon_baseline_profile=_kernel_dict.get("gluon_baseline_profile"),
        allowed_output_dialects=_kernel_dict.get("allowed_output_dialects"),
        preferred_output_dialects=_kernel_dict.get("preferred_output_dialects"),
        output_dialect_search_policy=_kernel_dict.get("output_dialect_search_policy"),
        target_backend=_kernel_dict.get("target_backend"),
        benchmark_shape_count=shape_count,
        benchmark_test_cases=shape_cases,
        shape_coverage_profile=shape_profile,
    )

    return PreprocessContext(
        kernel_path=kernel_path,
        repo_root=repo_root,
        harness_path=harness_path,
        preprocess_dir=str(pp_dir),
        commandment_path=str(pp_dir / "COMMANDMENT.md") if (pp_dir / "COMMANDMENT.md").exists() else "",
        codebase_context_path=str(pp_dir / "CODEBASE_CONTEXT.md") if (pp_dir / "CODEBASE_CONTEXT.md").exists() else "",
        baseline_metrics_path=str(pp_dir / "baseline_metrics.json")
        if (pp_dir / "baseline_metrics.json").exists()
        else "",
        profiling_result_path=str(pp_dir / "profile.json") if (pp_dir / "profile.json").exists() else "",
        input_dialect=feature_meta["input_dialect"],
        gluon_feature_mode=feature_meta["gluon_feature_mode"],
        gluon_baseline_profile=feature_meta["gluon_baseline_profile"],
        allowed_output_dialects=feature_meta["allowed_output_dialects"],
        preferred_output_dialects=feature_meta["preferred_output_dialects"],
        output_dialect_search_policy=feature_meta["output_dialect_search_policy"],
        target_backend=feature_meta["target_backend"],
        benchmark_shape_count=feature_meta.get("benchmark_shape_count"),
        benchmark_test_cases=feature_meta.get("benchmark_test_cases"),
        benchmark_test_cases_path=str(pp_dir / "benchmark_test_cases.json")
        if (pp_dir / "benchmark_test_cases.json").exists()
        else None,
        shape_coverage_profile=feature_meta.get("shape_coverage_profile"),
        discovery=discovery,
    )
