"""Shared helpers for the GEAK preprocessing and orchestration pipelines.

All CLI entry points (``geak``, ``geak-preprocess``, ``run-tasks``,
``task-generator``) import from this module so that harness
extraction, validation, profiling, model loading, agent filtering, and
pipeline-context injection are always identical regardless of entry point.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import logging
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from minisweagent import get_repo_root
from minisweagent.run.preprocess.discovery_types import (
    AMD_GLUON_DIALECT,
    DEFAULT_SHAPE_COVERAGE_PROFILE,
    NV_GLUON_DIALECT,
    SHAPE_COVERAGE_BUCKETED,
    SHAPE_COVERAGE_MULTI,
    build_gluon_feature_prompt_block,
)
from minisweagent.run.gluon_doc_profiles import task_requires_gluon_worker_docs
from minisweagent.run.utils.gpu_arch import (
    detect_gpu_arch,
    is_wmma_capable,
    rdna_arch_context,
    rdna_compute_bound_guidance,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = get_repo_root()

REQUIRED_HARNESS_FLAGS = ("--profile", "--correctness", "--benchmark", "--full-benchmark")
# ``--iterations`` is RECOMMENDED, not required. See
# ``minisweagent.run.preprocess.harness_utils`` for the canonical contract.
RECOMMENDED_HARNESS_FLAGS = ("--iterations",)

MAX_HARNESS_RETRIES = 2

# Use one canonical benchmark definition everywhere. The legacy
# GEAK_AGENT_BENCHMARK_ITERATIONS split is intentionally ignored so
# agent-time patch testing and final verification stay apples-to-apples.
DEFAULT_EVAL_BENCHMARK_ITERATIONS = int(os.getenv("GEAK_EVAL_BENCHMARK_ITERATIONS", "30"))
DEFAULT_AGENT_BENCHMARK_ITERATIONS = DEFAULT_EVAL_BENCHMARK_ITERATIONS
DEFAULT_PIPELINE_OUTPUT_DIR = "geak_output"
DEFAULT_HETEROGENEOUS = False

# Modes accepted by ``--mode``. See ``run/budget.py`` and ``run.budgets`` /
# ``run.presets`` in ``geak.yaml``.
RUN_MODES: tuple[str, ...] = ("quick", "full")
DEFAULT_RUN_MODE: str = "full"


def apply_mode_presets(config: dict, mode: str) -> dict:
    """Deep-merge ``config["run"]["presets"][mode]`` into *config*.

    Mode controls only ``orchestrator.max_rounds`` (and any other future
    non-env knobs); step / cost / iteration limits intentionally remain
    user-controlled to avoid silent overrides of ``GEAK_*_STEP_LIMIT`` etc.
    that users may have set in their shell.

    Precedence:
      - For ``max_rounds``: CLI ``--max-rounds`` > mode preset > ``GEAK_MAX_ROUNDS``
        env > built-in default. Mode wins over env because that is its job.
      - For step/cost limits: CLI > YAML ``agent.*`` > ``GEAK_*`` env > default.
        Mode is *not* in this chain.
      - For ``total_s`` / ``finalize_grace_s`` / preprocess caps: CLI override
        flag > mode preset > built-in default. No env vars participate.

    Returns the same dict (mutated) for caller convenience.
    """
    if mode not in RUN_MODES:
        raise ValueError(f"Unknown run mode {mode!r}; expected one of {RUN_MODES}")

    presets = (((config.get("run") or {}).get("presets")) or {}).get(mode) or {}
    if not presets:
        logger.debug("apply_mode_presets: no presets defined for mode=%s", mode)
        return config

    # Snapshot the leaf values that *will* change under deep-merge so the
    # log line reflects the actual config delta instead of the preset tree.
    # Walking the preset tree alone would, e.g., claim we "injected" a dict
    # at a key where ``_deep_merge`` actually replaced a non-dict scalar
    # with a fresh dict subtree.
    def _collect_preset_changes(base: dict, override: dict, prefix: str = "") -> list[tuple[str, Any, Any]]:
        changes: list[tuple[str, Any, Any]] = []
        for k, v in override.items():
            key = f"{prefix}.{k}" if prefix else k
            base_v = base.get(k) if isinstance(base, dict) else None
            if isinstance(v, dict) and isinstance(base_v, dict):
                changes.extend(_collect_preset_changes(base_v, v, key))
            else:
                changes.append((key, base_v, v))
        return changes

    deltas = _collect_preset_changes(config, presets)

    def _deep_merge(base: dict, override: dict) -> dict:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _deep_merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)
        return base

    _deep_merge(config, presets)

    parts: list[str] = []
    for key, before, after in deltas:
        if before is None:
            parts.append(f"+{key}={after}")
        elif before == after:
            parts.append(f"={key}={after}")
        else:
            parts.append(f"{key}: {before}->{after}")
    logger.info("apply_mode_presets: mode=%s applied %s", mode, ", ".join(parts) if parts else "(no changes)")
    return config


def resolve_max_rounds(
    *,
    cli_max_rounds: int | None,
    config: dict | None = None,
    default: int = 5,
) -> tuple[int, str]:
    """Resolve ``max_rounds`` per the documented precedence chain.

    Returns ``(value, source)`` where ``source`` is one of
    ``"cli"``, ``"mode"``, ``"env"``, ``"default"`` and is suitable for
    surfacing in the budget banner so users can tell that ``--mode`` overrode
    ``GEAK_MAX_ROUNDS`` (or did not).
    """
    if cli_max_rounds is not None:
        return int(cli_max_rounds), "cli"

    if config is not None:
        mode_val = (config.get("orchestrator") or {}).get("max_rounds")
        if mode_val is not None:
            return int(mode_val), "mode"

    env_val = os.environ.get("GEAK_MAX_ROUNDS")
    if env_val:
        try:
            return int(env_val), "env"
        except ValueError:
            logger.warning("GEAK_MAX_ROUNDS=%r is not an integer; falling back to default", env_val)

    return int(default), "default"


# ── agent filtering ──────────────────────────────────────────────────


def add_agent_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--allowed-agents`` and ``--excluded-agents`` to *parser*."""
    parser.add_argument(
        "--allowed-agents",
        default=None,
        help=("Comma-separated list of allowed agent types (e.g. strategy_agent). Sets GEAK_ALLOWED_AGENTS."),
    )
    parser.add_argument(
        "--excluded-agents",
        default=None,
        help=("Comma-separated list of excluded agent types (e.g. openevolve). Sets GEAK_EXCLUDED_AGENTS."),
    )


def apply_agent_filter_env(args: argparse.Namespace) -> None:
    """Propagate ``--allowed-agents`` / ``--excluded-agents`` to env vars."""
    configure_agent_filter_env(
        getattr(args, "allowed_agents", None),
        getattr(args, "excluded_agents", None),
    )


def configure_agent_filter_env(
    allowed_agents: str | None,
    excluded_agents: str | None,
) -> None:
    """Apply generic default agent filters.

    Default behavior excludes ``openevolve`` unless the user explicitly
    supplies an allowlist/excludelist or pre-sets ``GEAK_EXCLUDED_AGENTS``.
    This keeps the default pipeline focused on the lighter-weight agents while
    still allowing users to opt in deliberately.
    """

    if allowed_agents:
        os.environ["GEAK_ALLOWED_AGENTS"] = allowed_agents
        if excluded_agents is not None:
            os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents
        return

    if excluded_agents:
        os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents
        return

    os.environ.setdefault("GEAK_EXCLUDED_AGENTS", "openevolve")


# ── model loading ────────────────────────────────────────────────────


def load_geak_model(
    model_name: str | None,
    *,
    config_spec: str = "geak",
) -> Any:
    """Load an LLM model using the standard GEAK config-resolution pattern.

    Reads the YAML config for *config_spec*, extracts the ``model`` section,
    and delegates to ``get_model``.  Falls back to the ``GEAK_MODEL``
    environment variable when *model_name* is ``None``.
    """
    import yaml

    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    resolved_name = model_name or os.environ.get("GEAK_MODEL") or "claude-opus-4.6"
    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    return get_model(resolved_name, config=model_config)


def geak_model_factory(
    model_name: str | None,
    *,
    config_spec: str = "geak",
):
    """Return a zero-arg callable that creates a fresh model each time."""
    import yaml

    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    resolved_name = model_name or os.environ.get("GEAK_MODEL") or "claude-opus-4.6"
    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    def _factory():
        return get_model(resolved_name, config=copy.deepcopy(model_config))

    return _factory


def _ensure_mcp_importable() -> None:
    """Add MCP tool source directories to sys.path if not already present."""
    for sub in (
        "mcp_tools/profiler-mcp/src",
        "mcp_tools/automated-test-discovery/src",
    ):
        p = str(_REPO_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


# ── harness path extraction ──────────────────────────────────────────


def extract_harness_path(test_command: str) -> str:
    """Extract the harness script path from a test command string.

    Handles patterns like::

        'pytest /path/to/test.py -v'                -> '/path/to/test.py'
        'python /path/to/harness.py --correctness'  -> '/path/to/harness.py'
        '/path/to/harness.py'                       -> '/path/to/harness.py'
    """
    try:
        tokens = shlex.split(test_command)
    except ValueError:
        tokens = test_command.split()

    for token in tokens:
        if token.endswith(".py") and "/" in token:
            return token

    for token in tokens:
        if token.endswith(".py"):
            return token

    return tokens[-1] if tokens else test_command


def _preferred_harness_path(log_dir: Path, kernel_path: Path | None) -> Path:
    if kernel_path is not None:
        stem = kernel_path.stem or "kernel"
        return log_dir / f"test_{stem}_harness.py"
    return log_dir / "geak_test_harness.py"


def _materialized_harness_bootstrap(
    *,
    repo_root: Path,
    kernel_path: Path | None,
) -> str:
    kernel_dir = kernel_path.resolve().parent if kernel_path is not None else None
    rel_kernel_dir: Path | None = None
    if kernel_dir is not None:
        try:
            rel_kernel_dir = kernel_dir.relative_to(repo_root.resolve())
        except ValueError:
            rel_kernel_dir = None

    rel_kernel_dir_text = str(rel_kernel_dir).replace("\\", "/") if rel_kernel_dir is not None else ""
    original_kernel_dir = str(kernel_dir) if kernel_dir is not None else ""
    return (
        "# GEAK materialized harness bootstrap\n"
        "def _resolve_geak_kernel_dir():\n"
        "    candidates = []\n"
        '    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()\n'
        "    if work_dir:\n"
        "        candidates.append(work_dir)\n"
        '    repo_root = os.environ.get("GEAK_REPO_ROOT", "").strip()\n'
        f"    rel_kernel_dir = {rel_kernel_dir_text!r}\n"
        "    if repo_root and rel_kernel_dir:\n"
        "        candidates.append(os.path.join(repo_root, rel_kernel_dir))\n"
        f"    original_kernel_dir = {original_kernel_dir!r}\n"
        "    if original_kernel_dir:\n"
        "        candidates.append(original_kernel_dir)\n"
        "    for candidate in candidates:\n"
        '        if candidate and os.path.isfile(os.path.join(candidate, "kernel.py")):\n'
        "            return candidate\n"
        "    return original_kernel_dir or os.getcwd()\n"
        "\n"
        "_KERNEL_DIR = _resolve_geak_kernel_dir()\n"
        "if _KERNEL_DIR not in sys.path:\n"
        "    sys.path.insert(0, _KERNEL_DIR)\n"
    )


def _rewrite_materialized_harness_source(
    source_text: str,
    *,
    repo_root: Path,
    kernel_path: Path | None,
) -> str:
    bootstrap = _materialized_harness_bootstrap(
        repo_root=repo_root,
        kernel_path=kernel_path,
    )
    legacy_patterns = [
        re.compile(
            r"(?ms)^# Ensure the kernel directory is importable\n"
            r"_KERNEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n"
            r"if _KERNEL_DIR not in sys\.path:\n"
            r"\s+sys\.path\.insert\(0, _KERNEL_DIR\)\n"
        ),
        re.compile(
            r"(?ms)^_KERNEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n"
            r"if _KERNEL_DIR not in sys\.path:\n"
            r"\s+sys\.path\.insert\(0, _KERNEL_DIR\)\n"
        ),
    ]
    for pattern in legacy_patterns:
        if pattern.search(source_text):
            return pattern.sub(bootstrap, source_text, count=1)

    import_block = re.compile(r"(?ms)\A((?:from __future__ import annotations\n)?(?:import .+\n|from .+ import .+\n)+)")
    match = import_block.match(source_text)
    if match:
        return source_text[: match.end()] + "\n" + bootstrap + source_text[match.end() :]
    return bootstrap + "\n" + source_text


def _materialize_validated_harness(
    *,
    test_command: str,
    harness_path: str,
    repo_root: Path,
    log_dir: Path | None,
    kernel_path: Path | None,
    gpu_id: int,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    if log_dir is None:
        return None

    source_harness = Path(harness_path).resolve()
    target_dir = log_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_harness = _preferred_harness_path(target_dir, kernel_path)
    if source_harness == target_harness:
        return None

    rewritten_text = _rewrite_materialized_harness_source(
        source_harness.read_text(),
        repo_root=repo_root,
        kernel_path=kernel_path,
    )
    target_harness.write_text(rewritten_text)
    shutil.copymode(source_harness, target_harness)

    materialized_command = test_command.replace(str(source_harness), str(target_harness))
    valid, static_errors = validate_harness(str(target_harness))
    if not valid:
        raise RuntimeError("Materialized harness static validation failed: " + "; ".join(static_errors))

    exec_ok, exec_errors, harness_results = execute_harness_validation(
        str(target_harness),
        repo_root=str(repo_root),
        gpu_id=gpu_id,
    )
    if not exec_ok:
        raise RuntimeError(
            "Materialized harness runtime validation failed: " + "; ".join(e.splitlines()[0] for e in exec_errors)
        )
    return materialized_command, str(target_harness), harness_results


# ── harness validation ───────────────────────────────────────────────


_GPU_ALLOC_IN_PROFILE_RE = re.compile(
    r"""torch\.(?:randn?|empty|zeros|ones|full)\s*\("""
    r"""[^)]*device\s*=\s*["']cuda["']""",
)


def validate_harness(harness_path: str) -> tuple[bool, list[str]]:
    """Static-analyse a harness script to verify it supports required CLI flags.

    Mirror of :func:`minisweagent.run.preprocess.harness_utils.validate_harness`
    -- see that docstring for the contract. The two copies exist because of
    the deliberate decoupling between ``run/`` and ``run/preprocess/``;
    helpers like ``_strip_python_comments`` are imported from
    ``harness_utils`` so the comment-stripping behaviour stays in lockstep.
    """
    from minisweagent.run.preprocess.harness_utils import _strip_python_comments

    harness = Path(harness_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not harness.is_file():
        return False, [f"Harness file not found: {harness}"]

    source = harness.read_text()

    has_parser = "argparse" in source or "ArgumentParser" in source or "click" in source or "typer" in source
    if not has_parser:
        errors.append(
            "Harness does not use argparse/click/typer -- "
            "CLI flags like --profile and --correctness will be silently ignored"
        )

    # Strip comments before substring-checking required flags so that a
    # ``# --iterations N not yet supported`` comment does not falsely
    # satisfy the validator. Must mirror the equivalent helper in
    # ``preprocess/harness_utils.py``.
    code_only_source = _strip_python_comments(source)
    for flag in REQUIRED_HARNESS_FLAGS:
        if flag not in code_only_source:
            errors.append(f"Harness source does not define '{flag}' flag")

    for flag in RECOMMENDED_HARNESS_FLAGS:
        if flag not in code_only_source:
            msg = (
                f"Harness {harness} does not define recommended flag '{flag}'; "
                f"GEAK will skip passing it on the harness CLI and rely on "
                f"GEAK_BENCHMARK_ITERATIONS only. Have your harness honour "
                f"that env var if you want to control iteration counts."
            )
            logger.warning(msg)
            warnings.append(msg)

    # Check for GPU-side tensor allocation inside the profile function.
    # rocprofv3 captures ALL GPU kernels, so torch.randn(..., device='cuda')
    # inside run_profile pollutes the trace with RNG kernels.
    _in_profile_fn = False
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("def ") and "profile" in stripped:
            _in_profile_fn = True
            continue
        if _in_profile_fn and stripped.startswith("def "):
            _in_profile_fn = False
        if _in_profile_fn and _GPU_ALLOC_IN_PROFILE_RE.search(line):
            errors.append(
                f"Line {lineno}: GPU tensor allocation inside profile function "
                f"(device='cuda'). Use device='cpu' then .to('cuda') to avoid "
                f"polluting the profiler trace with RNG/memset kernels. "
                f"See src/minisweagent/run/preprocess/INSTRUCTIONS.md point 8."
            )
            break  # one warning is enough

    valid = len(errors) == 0
    return valid, (warnings if valid else errors)


# ── harness runtime execution ─────────────────────────────────────────


def execute_harness_validation(
    harness_path: str,
    repo_root: str | None = None,
    gpu_id: int = 0,
    benchmark_extra_args: str | None = None,
) -> tuple[bool, list[str], list[dict]]:
    """Run the harness across all modes and return ``(ok, errors, results)``.

    Delegates to :func:`minisweagent.tools.run_harness.run_harness` with
    ``mode="all"`` which executes correctness -> profile -> benchmark ->
    full-benchmark in sequence, short-circuiting on first failure.

    Parameters
    ----------
    benchmark_extra_args:
        Extra benchmark tuning args. ``--iterations N`` is normalized into
        ``GEAK_BENCHMARK_ITERATIONS`` so harnesses do not need to expose it
        as a CLI flag. Any remaining args are passed via
        ``GEAK_BENCHMARK_EXTRA_ARGS``.

    Returns
    -------
    ok : bool
        True if every mode passed.
    errors : list[str]
        Human-readable error descriptions for failed modes (empty on success).
    results : list[dict]
        Per-mode result dicts from :func:`run_harness`.
    """
    from minisweagent.run.preprocess.run_harness import results_errors, run_harness

    env_overrides: dict[str, str] = {}
    # Keep validation fast: override iterations to a small number unless
    # the caller explicitly provides benchmark_extra_args.
    if not benchmark_extra_args:
        env_overrides["GEAK_BENCHMARK_ITERATIONS"] = "5"
    else:
        import re as _re

        remaining_extra_args = benchmark_extra_args.strip()
        _iter_match = _re.search(r"(?:^|\s)--iterations\s+(\d+)(?=\s|$)", remaining_extra_args)
        if _iter_match:
            env_overrides["GEAK_BENCHMARK_ITERATIONS"] = _iter_match.group(1)
            remaining_extra_args = _re.sub(
                r"(?:^|\s)--iterations\s+\d+(?=\s|$)",
                " ",
                remaining_extra_args,
            ).strip()
        if remaining_extra_args:
            env_overrides["GEAK_BENCHMARK_EXTRA_ARGS"] = remaining_extra_args

    results = run_harness(
        harness_path,
        mode="all",
        repo_root=repo_root,
        gpu_id=gpu_id,
        env_overrides=env_overrides,
    )
    if not isinstance(results, list):
        results = [results]

    ok = all(r["success"] for r in results)
    errors = results_errors(results) if not ok else []
    return ok, errors, results


# ── validated harness creation (UnitTestAgent + retry) ───────────────


def create_validated_harness(
    *,
    model: Any,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None,
    kernel_path: Path | None,
    discovery_context: str,
    max_retries: int = MAX_HARNESS_RETRIES,
    gpu_id: int = 0,
) -> tuple[str, list[dict]]:
    """Run UnitTestAgent with static + runtime validation and retry loop.

    After the agent produces a harness:
      1. :func:`validate_harness` performs static analysis (argparse,
         ``--profile``, ``--correctness`` flags, GPU allocation patterns).
      2. :func:`execute_harness_validation` actually runs the harness in
         all four modes (correctness, profile, benchmark, full-benchmark)
         to catch import errors, shape mismatches, OOM, etc.

    If either step fails the errors are fed back into the discovery context
    and the agent is re-invoked, up to *max_retries* additional attempts.

    Returns ``(test_command, harness_results)`` on success where
    *harness_results* is the list of per-mode result dicts.

    Raises
    ------
    RuntimeError
        If validation still fails after all retries.
    """
    from minisweagent.run.preprocess.unit_test_agent import run_unit_test_agent

    max_attempts = max_retries + 1
    harness_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        ctx = discovery_context
        if harness_errors:
            ctx += (
                f"\n\nHARNESS VALIDATION FAILED (attempt {attempt}/{max_attempts}):\n"
                + "\n".join(f"- {e}" for e in harness_errors)
                + "\n\nYou MUST fix the harness so that ALL modes work: "
                "--correctness, --profile, --benchmark, --full-benchmark. "
                "See src/minisweagent/run/preprocess/INSTRUCTIONS.md sections 1a and 1b."
            )

        test_command = run_unit_test_agent(
            model=model,
            repo=repo,
            kernel_name=kernel_name,
            log_dir=log_dir,
            preferred_harness_path=_preferred_harness_path(log_dir, kernel_path) if log_dir else None,
            kernel_path=kernel_path,
            discovery_context=ctx,
        )
        logger.info("UnitTestAgent test_command (attempt %d): %s", attempt, test_command)

        harness = extract_harness_path(test_command)

        # Phase 1: static analysis
        valid, harness_errors = validate_harness(harness)
        if not valid:
            logger.warning(
                "Harness static validation failed (attempt %d/%d): %s",
                attempt,
                max_attempts,
                harness_errors,
            )
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Harness validation failed after {max_attempts} attempts: " + "; ".join(harness_errors)
                )
            continue

        logger.info("Harness static validation: OK")

        # Phase 2: runtime execution of all modes
        repo_root = str(repo) if repo else None
        exec_ok, exec_errors, harness_results = execute_harness_validation(
            harness,
            repo_root=repo_root,
            gpu_id=gpu_id,
        )
        if exec_ok:
            try:
                materialized = _materialize_validated_harness(
                    test_command=test_command,
                    harness_path=harness,
                    repo_root=repo,
                    log_dir=log_dir,
                    kernel_path=kernel_path,
                    gpu_id=gpu_id,
                )
            except Exception as exc:
                harness_errors = [str(exc)]
                logger.warning(
                    "Harness materialization failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    harness_errors,
                )
                if attempt == max_attempts:
                    raise RuntimeError(f"Harness materialization failed after {max_attempts} attempts: {exc}")
                continue
            if materialized is not None:
                test_command, harness, harness_results = materialized
                logger.info("Materialized harness to %s", harness)
            logger.info("Harness runtime validation: ALL MODES PASSED")
            return test_command, harness_results

        harness_errors = exec_errors
        logger.warning(
            "Harness runtime validation failed (attempt %d/%d): %s",
            attempt,
            max_attempts,
            [e.splitlines()[0] for e in exec_errors],
        )

        if attempt == max_attempts:
            raise RuntimeError(
                f"Harness runtime validation failed after {max_attempts} attempts: "
                + "; ".join(e.splitlines()[0] for e in exec_errors)
            )

    raise AssertionError("unreachable")  # pragma: no cover


# ── bottleneck-specific optimization guidance ────────────────────────

_BOTTLENECK_GUIDANCE: dict[str, str] = {
    "balanced": (
        "## Optimization Guidance (bottleneck: balanced)\n"
        '"Balanced" means no single resource is saturated. Actionable kernel-body approaches:\n'
        "1. INCREASE ARITHMETIC INTENSITY: Fuse adjacent operations into the kernel loop "
        "so more compute happens per memory access.\n"
        "2. REDUCE MEMORY TRAFFIC: Cache intermediate results in registers or LDS "
        "instead of reading/writing global memory.\n"
        "3. IMPROVE PARALLELISM: Restructure loops to expose more independent work per "
        "wavefront; consider split-K or multi-pass approaches.\n"
        "4. ALTERNATIVE ALGORITHMS: Try a fundamentally different algorithm for the same "
        "computation (different reduction tree, different scan, tiled vs non-tiled, etc.).\n"
        "5. COMPILER GUIDANCE: Restructure Triton/HIP code to help the compiler generate "
        "better ISA -- avoid tl.where in hot loops, use tl.constexpr aggressively, "
        "minimize live variables across tl.dot calls.\n"
    ),
    "memory-bound": (
        "## Optimization Guidance (bottleneck: memory-bound)\n"
        "The kernel is limited by memory bandwidth. Focus on kernel-body changes:\n"
        "1. VECTORIZED LOADS: Use float4/float2 vector loads to maximize HBM throughput.\n"
        "2. COALESCED ACCESS: Ensure adjacent threads access adjacent memory addresses.\n"
        "3. LDS STAGING: Stage global memory reads through LDS to improve access patterns.\n"
        "4. REDUCE DATA MOVEMENT: Recompute values instead of storing and reloading them.\n"
        "5. OPERATION FUSION: Fuse the memory-bound kernel with adjacent elementwise ops "
        "to amortize memory access cost over more computation.\n"
        "6. TILING / BLOCKING: Increase tile sizes to improve data reuse from L2 cache.\n"
    ),
    "compute-bound": (
        "## Optimization Guidance (bottleneck: compute-bound)\n"
        "The kernel is limited by arithmetic throughput. Focus on kernel-body changes:\n"
        "1. REDUCE INSTRUCTION COUNT: Simplify expressions, use hardware intrinsics "
        "(tl.math.rsqrt, fma), eliminate redundant computations.\n"
        "2. USE MFMA INSTRUCTIONS: On AMD GPUs, restructure computation to use Matrix "
        "Fused Multiply-Add for dense linear algebra.\n"
        "3. STRENGTH REDUCTION: Replace expensive ops (div, mod, pow) with cheaper "
        "equivalents (shifts, masks, lookup tables).\n"
        "4. LOOP UNROLLING: Manually unroll inner loops to help the compiler schedule "
        "instructions more aggressively.\n"
        "5. ALGORITHM CHANGE: Switch to an algorithm with lower computational complexity "
        "(e.g., O(n log n) vs O(n^2), approximate methods).\n"
    ),
    "latency-bound": (
        "## Optimization Guidance (bottleneck: latency-bound)\n"
        "The kernel is too short to saturate any resource. Focus on kernel-body changes:\n"
        "1. INCREASE WORK PER KERNEL: Process more elements per thread or per block "
        "to amortize kernel launch overhead.\n"
        "2. FUSE KERNELS: Merge this kernel with adjacent ones to eliminate launch gaps.\n"
        "3. PERSISTENT KERNEL: Convert to a persistent kernel pattern that stays resident "
        "and processes multiple tiles without relaunching.\n"
        "4. INCREASE BLOCK SIZE: Use larger thread blocks to improve GPU occupancy for "
        "this short-running kernel.\n"
    ),
    "lds-bound": (
        "## Optimization Guidance (bottleneck: lds-bound)\n"
        "The kernel is limited by LDS (Local Data Share) bandwidth or capacity.\n"
        "1. REDUCE LDS BANK CONFLICTS: Pad shared memory arrays to avoid stride-32 "
        "access patterns (on AMD: 32 banks, 4 bytes each).\n"
        "2. REDUCE LDS USAGE: Move data from LDS to registers where possible to free "
        "LDS capacity and improve occupancy.\n"
        "3. OPTIMIZE LDS ACCESS PATTERN: Restructure loops so that LDS reads/writes "
        "are coalesced within each wavefront.\n"
        "4. SPLIT COMPUTATION: Break the kernel into phases that use LDS at different "
        "times to reduce peak LDS pressure.\n"
    ),
}


_SEARCH_WORKLOAD_HINTS = (
    "binary_search",
    "lower_bound",
    "upper_bound",
    "search_n",
    "device_search",
    "haystack",
    "needle",
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _search_workload_guidance(metrics: dict) -> list[str]:
    """Add narrower guidance for latency-bound HIP search workloads."""
    evidence_chunks = [str(metrics.get("kernel_name", ""))]
    for top in metrics.get("top_kernels", []) or []:
        evidence_chunks.append(str(top.get("name", "")))
    haystack = " ".join(evidence_chunks).lower()
    if not any(hint in haystack for hint in _SEARCH_WORKLOAD_HINTS):
        return []

    bottleneck = str(metrics.get("bottleneck", "")).lower()
    if "latency" not in bottleneck:
        return []

    derived = metrics.get("metrics", {}) or {}
    hbm_util = _safe_float(derived.get("memory.hbm_bandwidth_utilization"))
    l2_hit = _safe_float(derived.get("memory.l2_hit_rate"))
    if hbm_util is not None and hbm_util >= 10.0:
        return []

    hbm_text = f"{hbm_util:.1f}%" if hbm_util is not None else "unknown"
    l2_text = f"{l2_hit:.1f}%" if l2_hit is not None else "unknown"
    return [
        "## Workload Guidance (HIP search / pointer-chasing)",
        (
            "Profiler evidence suggests a latency-bound search workload: "
            f"HBM utilization={hbm_text}, L2 hit rate={l2_text}."
        ),
        "Prioritize branchless search logic, operation-specific specialization, and size-specialized variants.",
        "Also consider wavefront-cooperative upper-level search and amortized pivot-table narrowing when correctness rules allow it.",
        "Deprioritize generic vectorization or bandwidth-maximization ideas unless later profiling shows memory throughput is actually the limiter.",
        "",
    ]


def _bottleneck_guidance(bottleneck: str, metrics: dict, arch: str = "") -> list[str]:
    """Return actionable optimization guidance lines based on bottleneck type."""
    bn_lower = bottleneck.lower().strip()
    bn_aliases = {
        "latency": "latency-bound",
        "memory": "memory-bound",
        "compute": "compute-bound",
        "lds": "lds-bound",
    }
    bn_lower = bn_aliases.get(bn_lower, bn_lower)
    for key, text in _BOTTLENECK_GUIDANCE.items():
        if key in bn_lower:
            if key == "compute-bound" and is_wmma_capable(arch):
                text = rdna_compute_bound_guidance()
            lines = text.strip().splitlines()
            lines.extend(_search_workload_guidance(metrics))
            lines.append("")
            return lines

    lines = _BOTTLENECK_GUIDANCE["balanced"].strip().splitlines()
    lines.extend(_search_workload_guidance(metrics))
    lines.append("")
    return lines


# ── GPU architecture context from profiling data ─────────────────────


def _gpu_arch_context(profiling_path: str) -> list[str]:
    """Extract GPU architecture info from profile.json and format it."""
    import json as _json

    try:
        data = _json.loads(Path(profiling_path).read_text())
    except Exception:
        logger.debug("Could not read or parse profiling JSON at %s", profiling_path, exc_info=True)
        return []

    results = data.get("results", [])
    if not results:
        return []

    gpu_info = results[0].get("gpu_info", {}) if isinstance(results[0], dict) else {}
    if not gpu_info:
        for r in results:
            if isinstance(r, dict) and r.get("gpu_info"):
                gpu_info = r["gpu_info"]
                break

    if not gpu_info:
        return []

    arch = gpu_info.get("architecture", gpu_info.get("gfx_version", "unknown"))
    name = gpu_info.get("name", gpu_info.get("model", "AMD GPU"))
    cus = gpu_info.get("compute_units", "?")
    hbm_bw = gpu_info.get("peak_hbm_bandwidth_gbps", gpu_info.get("hbm_bandwidth", "?"))
    lds_per_cu = gpu_info.get("lds_per_cu_kb", 64)
    vgprs = gpu_info.get("vgprs_per_cu", 512)
    rdna_ctx = rdna_arch_context(gpu_info, arch)
    if rdna_ctx is not None:
        return rdna_ctx

    return [
        f"## GPU Architecture: {name} ({arch})",
        f"- Architecture: {arch}",
        f"- Compute Units: {cus}",
        f"- Peak HBM bandwidth: {hbm_bw} GB/s",
        f"- LDS per CU: {lds_per_cu} KB (32 banks on gfx9xx)",
        f"- VGPRs per CU: {vgprs}",
        "- Wavefront size: 64 (AMD default), some kernels can use 32",
        "- MFMA (Matrix Fused Multiply-Add) instructions available for dense math",
        "- Use these specs to guide your kernel optimizations (tile sizes, occupancy, LDS usage).",
        "",
    ]


# ── gluon guidance injection ──────────────────────────────────────────


def _task_needs_gluon_worker_context(
    feature_metadata: dict[str, Any] | None,
    task_body: str = "",
) -> bool:
    """Return whether this dispatched task should receive Gluon worker guidance."""
    feature_metadata = feature_metadata or {}
    required_output = str(feature_metadata.get("required_output_dialect") or "").strip().lower()
    task_type = str(feature_metadata.get("task_type") or _task_body_field(task_body, "Task type")).strip().lower()
    if required_output == "plain_triton" or task_type == "plain_subkernel_refine":
        return False
    return task_requires_gluon_worker_docs(
        kernel_type=str(feature_metadata.get("kernel_type") or ""),
        required_output=required_output,
        implementation_layer=str(feature_metadata.get("implementation_layer") or ""),
        extension_layer=str(feature_metadata.get("extension_layer") or ""),
        doc_profile=str(feature_metadata.get("gluon_doc_profile") or ""),
        task_body=task_body,
        label=str(feature_metadata.get("label") or ""),
    )


def _build_gluon_reference_block(
    *,
    knowledge_base_path: str | None = None,
    gluon_skill_path: str | None = None,
    gluon_kb_path: str | None = None,
    gluon_examples_path: str | None = None,
    gluon_always_read_path: str | None = None,
    gluon_search_policies_path: str | None = None,
    gluon_component_traits_path: str | None = None,
    gluon_architecture_notes_path: str | None = None,
    gluon_examples_doc_path: str | None = None,
    gluon_api_reference_path: str | None = None,
    gluon_real_patterns_path: str | None = None,
    gluon_backup_details_path: str | None = None,
) -> list[str]:
    refs: list[tuple[str, str]] = []
    split_refs: list[tuple[str, str | None]] = [
        ("Triton-Gluon skill", gluon_skill_path),
        ("Split-doc entrypoint", gluon_always_read_path),
        ("Planner/task allocation", gluon_search_policies_path),
        ("Implementation traits", gluon_component_traits_path),
        ("Architecture, target, JIT/AOT, version sensitivity", gluon_architecture_notes_path),
        ("API syntax, launch skeletons, compatibility checks, failure-fix order", gluon_api_reference_path),
        ("Real aiter patterns, family differences, source-first triggers, benchmark rules", gluon_real_patterns_path),
        ("Schematic examples", gluon_examples_doc_path),
        ("Residual backup routing", gluon_backup_details_path),
    ]
    refs.extend((label, path) for label, path in split_refs if path)
    if gluon_kb_path:
        refs.append(("Structured Gluon knowledge base", gluon_kb_path))
    if gluon_examples_path:
        refs.append(("Gluon examples and harness notes", gluon_examples_path))
    if knowledge_base_path and all(path != knowledge_base_path for _, path in refs):
        refs.append(("Knowledge base path", knowledge_base_path))

    if not refs:
        return []

    lines = ["## Canonical Gluon References"]
    for label, path in refs:
        lines.append(f"- {label}: {path}")
    lines.append(
        "- Read the short working-set rules below first. Use these paths with `view` only when you need deeper detail."
    )
    if gluon_always_read_path:
        lines.append(
            f"- Mandatory split-doc routing: first read the absolute entrypoint `{gluon_always_read_path}`, "
            "then use its `Task routing` table to choose exact split-doc files and headings."
        )
    else:
        lines.append(
            "- Mandatory split-doc routing: first read the absolute `00_always_read.md` path listed above, "
            "then use its `Task routing` table to choose exact split-doc files and headings."
        )
    lines.append(
        "- Do not implement from memory or guess Gluon API names. If the routed primary split docs do not contain the required detail, read `70_backup_details.md`; if it still lacks detail, report the missing split-doc route so the docs can be updated."
    )
    lines.append(
        "- These reference files are read-only guidance. They may live outside REPO ROOT; you may `view` them, but do not modify them."
    )
    lines.append("")
    return lines


def _build_gluon_working_set(
    feature_metadata: dict[str, Any] | None,
    *,
    gluon_always_read_path: str | None = None,
) -> list[str]:
    feature_metadata = feature_metadata or {}
    input_dialect = str(feature_metadata.get("input_dialect") or "").strip().lower()
    entrypoint = str(gluon_always_read_path or "skills/triton-gluon/docs/00_always_read.md").strip()
    source_origin = str(feature_metadata.get("source_origin") or "unknown").strip()
    gluon_tl_policy = str(feature_metadata.get("gluon_tl_policy") or "strict_generated").strip()
    layout_policy = str(feature_metadata.get("layout_construction_policy") or "host_preferred").strip()
    execution_mode = str(feature_metadata.get("execution_mode") or "").strip()
    extension_intent = str(feature_metadata.get("extension_intent") or "").strip().lower()
    expected_outcome = str(feature_metadata.get("expected_outcome") or "").strip()
    l0_scope_classification = str(feature_metadata.get("l0_scope_classification") or "").strip()
    l0_coupling_reasons = str(feature_metadata.get("l0_coupling_reasons") or "").strip()
    expected_failure_layers = str(feature_metadata.get("expected_failure_layers") or "").strip()
    first_patch_compile_goal = str(feature_metadata.get("first_patch_compile_goal") or "").strip()
    do_not_optimize_before_compile = str(feature_metadata.get("do_not_optimize_before_compile") or "").strip()
    matrix_lowering_required = str(feature_metadata.get("matrix_lowering_required") or "").strip()
    task_signals = str(feature_metadata.get("task_signals") or "").strip()
    routed_doc_reasons = str(feature_metadata.get("routed_doc_reasons") or "").strip()
    kernel_family_signal = str(feature_metadata.get("kernel_family_signal") or "").strip()
    failure_layers = str(feature_metadata.get("failure_layers") or "").strip()
    minimum_executable_unit = str(feature_metadata.get("minimum_executable_unit") or "").strip()
    target_symbol = str(feature_metadata.get("target_symbol") or "").strip()
    target_component = str(feature_metadata.get("target_component") or "").strip()
    forbidden_change = str(feature_metadata.get("forbidden_change") or "").strip()
    allowed_execution_path = str(feature_metadata.get("allowed_execution_path") or "").strip()
    scope_infeasible_policy = str(feature_metadata.get("scope_infeasible_policy") or "").strip()
    whole_kernel_required_reason = str(feature_metadata.get("whole_kernel_required_reason") or "").strip()
    forbidden_symbols = feature_metadata.get("forbidden_patch_target_symbols") or []
    if isinstance(forbidden_symbols, str):
        forbidden_symbols_text = forbidden_symbols
    else:
        forbidden_symbols_text = ", ".join(str(item) for item in forbidden_symbols)

    lines = [
        "## Gluon Working Set",
        f"- Use targeted reading: start with `{entrypoint}`, use its `stable_split_doc_index` and `Task routing` table, then jump to the exact split-doc headings relevant to this task.",
        "- When a REQUIRED BEFORE EDITING OR SAVE_AND_TEST block is present, those absolute paths are the doc-gate paths that `save_and_test` verifies.",
        "- Do not implement from memory or guess Gluon API names. Read the routed split-doc entry before writing a Gluon patch; use `70_backup_details.md` when primary split docs lack detail, then report the missing route instead of relying on long-form backup.",
        "- `save_and_test` enforces this for Gluon tasks: use `str_replace_editor` with `command=\"view\"` on the required absolute split-doc paths before saving or benchmarking a patch.",
        "- Before editing, write a `Gluon knowledge lookup plan` in your strategy notes: task signals, exact split-doc files/headings, viewed=yes/no for each, source sections viewed, and missing details.",
        "- Do not start implementation while a required lookup row is still `viewed=no`; either view that section or record the exact missing-doc route.",
        "- Preserve launcher shape, indexing, masks, correctness behavior, and benchmark intent before changing algorithms.",
        "- Before editing a Gluon path, write a `Gluon implementation plan`, `Performance hypothesis`, and `Patch evolution` plan. It must name `Optimization direction`, `Implementation layer`, `Gluon overlay reason`, `Measurement boundary`, `Same ABI comparison`, `Comparison target`, and `Allowed change`; detailed fields live in `00_always_read.md`, `10_search_policies.md`, and `60_real_patterns.md`.",
        "- Required AMD Gluon patches must be real executed Gluon paths, not import-only, helper-only, empty, or plain Triton fallbacks.",
        "- Keep task scope narrow: one subpath/component unless `bundle_allowed=true`; use the docs for detailed rejection conditions and fix order.",
        "",
        "## Patch Evolution Working Set",
        "- This task received Gluon worker docs/context. Before editing, write a `Patch evolution plan` and use `10_search_policies.md::atomic_component_lattice` / `l0_scope_decision_before_emit` plus `60_real_patterns.md::patch_evolution_by_task_type` / `failure_to_next_patch_map` for pass/fail next-patch rules.",
        "- For `patch_0`, name one `primary_component` from the atomic lattice. Treat other components as blockers or expected failure layers unless the task explicitly allows a bundle.",
        "- Branch A local smoke/probe stays local: if it cannot execute independently, record `Task correction` and shrink/report instead of upgrading to whole-kernel inside the worker patch.",
        "- Branch B whole-helper skeleton starts as compile/wiring/layout evidence; do not add MFMA, buffer ops, scheduler, epilogue, or tuning before the skeleton executes and a later patch names one removable overhead.",
        "- Use `strategy_manager.mark` or `strategy_manager.note` when helpful to record failure-layer status, compiler/runtime errors, `Task correction`, and the next single-variable improvement. The strategy manager is only a readable notebook; it is not a postprocess contract.",
        "",
        "## Task consistency check",
        "- Before editing, compare the task body, metadata, source, and routed docs. Use `10_search_policies.md::overlay_direction_vs_mechanism` / `atomic_component_lattice` / `l0_scope_decision_before_emit` for same-direction component and branch checks, and `60_real_patterns.md::Task consistency check` / `Whole-kernel L0 comparison anchor` / `l0_scope_by_kernel_family` for whole-kernel anchors.",
        "- For broadcast-heavy or whole-kernel L0, write a parent-layout map before editing. If that map cannot be made self-consistent, use `scope_infeasible_policy` to shrink/report instead of widening scope.",
        "- If metadata is incomplete or inconsistent with source evidence, record `Task correction` in notes/summary and continue within the existing task boundary. Do not add new metadata fields or widen the allowed scope.",
        "- If the scoped Gluon change cannot be implemented without touching forbidden paths, do not widen the scope. Use the task's allowed execution path, shrink/report infeasible, or ask for a new task.",
        "- For low-latency kernels or tiny stages, L0 is a smallest executed anchor. If a correctness-passing L0 is slower than Base, record the overhead evidence and avoid repeated `num_warps`, block-size, or launch-constant sweeps.",
        f"- Source origin: `{source_origin}`. Missing/unknown origin uses strict generated-overlay rules; only verified production Gluon source may preserve existing mixed `tl.*` patterns.",
        f"- Gluon TL policy: `{gluon_tl_policy}`. `tl.arange`, `tl.load`, `tl.store`, `tl.zeros`, `tl.full`, and `tl.dot` are forbidden in newly generated Gluon device paths.",
        f"- Layout construction policy: `{layout_policy}`. Generated L0 prefers host-created layouts; existing production `gl.constexpr` layout declarations may be preserved when the source contract is clear.",
    ]
    if execution_mode:
        lines.append(f"- Execution mode: `{execution_mode}`. Preserve AOT/JIT/prebuilt fallback and signature contracts unless the task explicitly changes that boundary.")
    if extension_intent == "execution_anchor":
        lines.append(
            "- Extension intent: `execution_anchor`. Treat correctness-passing but slower Gluon as overhead evidence, not as permission to expand the same scope into L1/MFMA without a named removable overhead."
        )
    if expected_outcome:
        lines.append(f"- Expected outcome: `{expected_outcome}`.")
    if l0_scope_classification:
        lines.append(f"- L0 scope classification: `{l0_scope_classification}`.")
    if l0_coupling_reasons:
        lines.append(f"- L0 coupling reasons: {l0_coupling_reasons}")
    if expected_failure_layers:
        lines.append(f"- Expected failure layers: {expected_failure_layers}")
    if first_patch_compile_goal:
        lines.append(f"- First patch compile goal: {first_patch_compile_goal}")
    if do_not_optimize_before_compile:
        lines.append(f"- Do not optimize before compile: `{do_not_optimize_before_compile}`.")
    if matrix_lowering_required:
        lines.append(f"- Matrix lowering required: `{matrix_lowering_required}`.")
    if task_signals:
        lines.append(f"- Task signals: {task_signals}")
    if routed_doc_reasons:
        lines.append(f"- Routed doc reasons: {routed_doc_reasons}")
    if kernel_family_signal:
        lines.append(f"- Kernel family signal: `{kernel_family_signal}`.")
    if failure_layers:
        lines.append(f"- Failure layers: {failure_layers}")
    if minimum_executable_unit:
        lines.append(f"- Minimum executable unit: `{minimum_executable_unit}`.")
    if target_symbol:
        lines.append(f"- Target symbol: `{target_symbol}`; do not report success through a different symbol.")
    if target_component:
        lines.append(f"- Target component: {target_component}; keep the patch scoped to this component.")
    if forbidden_change:
        lines.append(f"- Forbidden change: {forbidden_change}")
    if forbidden_symbols_text:
        lines.append(f"- Forbidden target scopes checked by tools: {forbidden_symbols_text}.")
    if allowed_execution_path:
        lines.append(f"- Allowed execution path: `{allowed_execution_path}`.")
        if allowed_execution_path == "inline_scoped_helper":
            lines.append(
                "- Inline scoped helper boundary: do not add a replacement whole-kernel `@gluon.jit`, do not reroute the wrapper's main path to a new full Gluon kernel, and do not report success by widening beyond the named target component."
            )
    if scope_infeasible_policy:
        lines.append(f"- Scope infeasible policy: `{scope_infeasible_policy}`.")
    if whole_kernel_required_reason:
        lines.append(f"- Whole kernel required reason: {whole_kernel_required_reason}")
    lines.append(
        "- L0 execution-path choices: `inline_scoped_helper` only when the language boundary permits the scoped change; `separate_gluon_kernel` only when the task explicitly allows a second launch/temp buffer; `whole_jit_kernel` only when the whole helper/kernel is declared as the minimum executable unit; `infeasible` means report or shrink scope instead of converting the whole kernel."
    )

    if input_dialect == NV_GLUON_DIALECT:
        lines.extend(
            [
                "- Treat `nv_gluon` as translation, not rename. Re-evaluate wave32-centric layouts, descriptor paths, async-copy paths, and tensor-memory features before carrying them to AMD.",
                "- Preserve semantic parity first, then optimize inside AMD-facing Gluon.",
            ]
        )
    elif input_dialect == AMD_GLUON_DIALECT:
        lines.extend(
            [
                "- Preserve the existing AMD-facing structure first. Stay inside `amd_gluon` space unless benchmark evidence clearly justifies a fallback comparison.",
            ]
        )
    else:
        lines.extend(
            [
                "- For `plain_triton -> amd_gluon`, only create a Gluon patch as a same-direction overlay with a concrete Gluon reason; keep the plain Triton competitor for that optimization direction.",
            ]
        )

    lines.extend(
        [
            "- High-frequency failure modes:",
            "  - layout/tensor creation and broadcast mistakes: read `20_component_traits.md` and `50_api_reference.md`.",
            "  - buffer dtype, MFMA layout, and target-specific API mistakes: read `20_component_traits.md`, `50_api_reference.md`, and `60_real_patterns.md`.",
            "  - helper wiring, execution-path, and bundled-patch attribution mistakes: read `00_always_read.md` and `60_real_patterns.md`.",
            "- When you see `DistributedLinearLayout`, `PartitionedSharedLayout`, host `TensorDescriptor`, `reshape` / `permute` / `trans` tile unshuffle, nested 3D/5D `SliceLayout`, or JIT/AOT packaging gates, stop generic rewriting and read the operator-local source.",
        ]
    )
    shape_profile = str(
        feature_metadata.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE
    ).lower()
    if shape_profile in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        lines.append(
            "- Multi-shape Gluon constraint: `BLOCK_*`, `num_warps`, layout `instr_shape`, "
            "and `DotOperandLayout` are `constexpr`. Every Gluon candidate must stay valid "
            "across all observed shapes (parametric layout, host-built layout passed as "
            "`constexpr`, or shape-bucketed dispatch)."
        )
    lines.append("")
    return lines


def _build_required_gluon_doc_gate_block(required_paths: list[str] | None) -> list[str]:
    paths = [str(path).strip() for path in (required_paths or []) if str(path).strip()]
    if not paths:
        return []
    lines = [
        "## REQUIRED BEFORE EDITING OR SAVE_AND_TEST",
        "Use `str_replace_editor` with `command=\"view\"` on every required Triton-Gluon doc path below before calling `save_and_test`.",
        "Read lazily: start with the headings needed for the current task. If a `view_range` overshoots EOF, the editor clamps to the available range.",
        "`save_and_test` will fail with `GLUON_DOC_GATE_FAILED` until these exact paths have been viewed.",
    ]
    lines.extend(f"- {path}" for path in paths)
    lines.append("")
    return lines


def _task_requires_amd_gluon_output(feature_metadata: dict[str, Any] | None) -> bool:
    """Return whether the task contract requires a real AMD Gluon patch."""
    if not feature_metadata:
        return False
    task_type = str(feature_metadata.get("task_type") or "").strip().lower()
    if task_type == "plain_subkernel_refine":
        return False
    return (
        str(feature_metadata.get("kernel_type") or "").strip().lower() == "triton"
        and str(feature_metadata.get("required_output_dialect") or "").strip().lower() == "amd_gluon"
    )


def _task_requires_route_proof(feature_metadata: dict[str, Any] | None, task_body: str = "") -> bool:
    """Return whether the worker must establish a pre-edit execution route proof."""
    if not feature_metadata:
        return False
    required_output = str(feature_metadata.get("required_output_dialect") or "").strip().lower()
    if required_output in {"amd_gluon", "mixed"}:
        return True
    return _task_needs_gluon_worker_context(feature_metadata, task_body)


def _gluon_route_priority_task(feature_metadata: dict[str, Any] | None, task_body: str = "") -> bool:
    """Return whether wrapper/route work should outrank generic kernel-body advice."""
    if not feature_metadata:
        return False
    text = "\n".join(
        str(part or "")
        for part in (
            task_body,
            feature_metadata.get("label"),
            feature_metadata.get("failure_layers"),
            feature_metadata.get("target_component"),
            feature_metadata.get("task_signals"),
            feature_metadata.get("measured_output_dependency"),
            feature_metadata.get("integration_boundary"),
            feature_metadata.get("allowed_execution_path"),
            feature_metadata.get("minimum_executable_unit"),
        )
    ).lower()
    measurement_boundary = str(feature_metadata.get("measurement_boundary") or "").lower()
    required_output = str(feature_metadata.get("required_output_dialect") or "").strip().lower()
    if required_output not in {"amd_gluon", "mixed"}:
        return False
    return (
        "full_operator" in measurement_boundary
        or "wrapper" in text
        or "reduction" in text
        or "one-shot" in text
        or "one_shot" in text
        or "reduce launch" in text
        or "output feeding" in text
        or "output-feeding" in text
        or "shape_dispatch" in text
    )


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _task_body_field(task_body: str, field_name: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(task_body or "")
    return match.group(1).strip() if match else ""


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _extract_commandment_harness(commandment_text: str | None, test_command: str | None) -> str:
    text = "\n".join(part for part in (commandment_text or "", test_command or "") if part)
    match = re.search(r"(?:python3?|python)\s+([^\s;&]+\.py)\b", text)
    return match.group(1) if match else ""


def _partition_hints_from_cases(feature_metadata: dict[str, Any] | None) -> list[str]:
    feature_metadata = feature_metadata or {}
    cases = feature_metadata.get("benchmark_test_cases") or []
    hints: list[str] = []
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, dict):
            continue
        params = case.get("params") if isinstance(case.get("params"), dict) else {}
        context_len = params.get("context_len") or params.get("context_length") or params.get("seq_len")
        partition = (
            params.get("context_partition_size")
            or feature_metadata.get("context_partition_size")
            or feature_metadata.get("partition_size")
        )
        if context_len and partition:
            try:
                parts = (int(context_len) + int(partition) - 1) // int(partition)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            case_id = str(case.get("case_id") or "?")
            hints.append(f"{case_id}: max_context_partition_num=ceil({context_len}/{partition})={parts}")
    return hints


def _path_status(path: str | None) -> str:
    if not path:
        return "missing"
    try:
        return "verified" if Path(path).exists() else "missing"
    except OSError:
        return "missing"


def _doc_gate_status(required_paths: list[str] | None) -> tuple[str, list[str]]:
    paths = [str(path).strip() for path in (required_paths or []) if str(path).strip()]
    missing = [path for path in paths if _path_status(path) != "verified"]
    if not paths:
        return "not_required", []
    return ("verified_by_runner" if not missing else "missing"), missing


def _candidate_config_lines(task_body: str) -> list[str]:
    lines: list[str] = []
    for line in str(task_body or "").splitlines():
        stripped = line.strip()
        if re.match(r"^(?:[-*]\s*)?[a-z]\.\s+", stripped, re.IGNORECASE) and any(
            marker in stripped
            for marker in ("KV_COMPUTE_BLOCK_SIZE", "waves_per_eu", "num_stages", "BLOCK_", "num_warps")
        ):
            lines.append(stripped)
    return lines


def _compact_benchmark_baseline(benchmark_baseline: str, *, limit: int = 8) -> list[str]:
    """Return a short benchmark excerpt for worker orientation, not ranking."""
    selected: list[str] = []
    for raw_line in benchmark_baseline.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if (
            "geak_result_latency_ms" in lowered
            or "geomean latency" in lowered
            or "total" in lowered
            or " ms" in lowered
        ):
            selected.append(line)
        if len(selected) >= limit:
            break
    if selected:
        return selected
    return [line.strip() for line in benchmark_baseline.splitlines() if line.strip()][:limit]


def _gluon_doc_heading_hints(feature_metadata: dict[str, Any]) -> list[str]:
    """Return a tiny, profile-based heading hint list for worker fallback reads."""
    profile = str(feature_metadata.get("gluon_doc_profile") or "").strip().lower()
    component = str(feature_metadata.get("target_component") or "").strip().lower()
    failure_layers = str(feature_metadata.get("failure_layers") or "").strip().lower()
    text = " ".join(part for part in (profile, component, failure_layers) if part)
    hints: list[str] = []
    if any(marker in text for marker in ("matrix", "dot", "mfma", "wmma")):
        hints.extend(
            [
                "20_component_traits.md::matrix_dot",
                "50_api_reference.md::dot_lowering_minimal_recipe",
            ]
        )
    if any(marker in text for marker in ("layout", "broadcast", "convert_layout")):
        hints.append("20_component_traits.md::layout_broadcast")
    if any(marker in text for marker in ("reduction", "softmax", "accumulator", "state_update")):
        hints.append("20_component_traits.md::reduction_accumulator")
    if any(marker in text for marker in ("memory", "load", "store", "buffer")):
        hints.extend(
            [
                "20_component_traits.md::load_store",
                "50_api_reference.md::buffer_ops",
            ]
        )
    if any(marker in text for marker in ("attention", "decode", "aiter", "existing")):
        hints.append("60_real_patterns.md::existing_amd_gluon_in_dialect_refinement")
    unique: list[str] = []
    for hint in hints:
        if hint not in unique:
            unique.append(hint)
    return unique[:4]


def _build_gluon_clean_task_packet(
    task_body: str,
    feature_metadata: dict[str, Any] | None,
    *,
    task_file_path: str | None,
    commandment_path: str | None,
    baseline_metrics_path: str | None,
    benchmark_baseline_path: str | None,
    kernel_path: str | None,
    repo_root: str | None,
    test_command: str | None,
    commandment_text: str | None,
    benchmark_baseline: str | None,
    gluon_doc_gate_required_paths: list[str] | None,
) -> list[str]:
    """Build the task-first packet shown before any auto-injected policy."""
    feature_metadata = feature_metadata or {}
    required_symbols = _as_list(feature_metadata.get("required_patch_target_symbols"))
    executed_route_symbols = _as_list(feature_metadata.get("executed_route_symbols"))
    forbidden_symbols = _as_list(feature_metadata.get("forbidden_patch_target_symbols"))
    harness = _extract_commandment_harness(commandment_text, test_command)
    doc_status, missing_docs = _doc_gate_status(gluon_doc_gate_required_paths)
    cases = feature_metadata.get("benchmark_test_cases") or []
    required_output = str(feature_metadata.get("required_output_dialect") or "<derive from task>").strip()
    source_origin = str(feature_metadata.get("source_origin") or "<derive from task>").strip()
    task_type = str(feature_metadata.get("task_type") or "<derive from task>").strip()
    doc_profile = str(feature_metadata.get("gluon_doc_profile") or "<none>").strip()
    target_component = str(feature_metadata.get("target_component") or "<read original task body>").strip()
    failure_layers = str(feature_metadata.get("failure_layers") or "<read original task body>").strip()
    doc_heading_hints = _gluon_doc_heading_hints(feature_metadata)

    lines = [
        "## Minimal Gluon Worker Packet",
        "Original task body/frontmatter is the single source of task semantics. Runner supplies compact evidence only.",
        "",
        "### External Contract Evidence",
        f"- Task file: {task_file_path or '<current task file>'} ({_path_status(task_file_path)})",
        f"- COMMANDMENT.md: {commandment_path or '<derive from task metadata>'} ({_path_status(commandment_path)})",
        f"- Baseline metrics: {baseline_metrics_path or '<derive from task metadata>'} ({_path_status(baseline_metrics_path)})",
        f"- Benchmark baseline: {benchmark_baseline_path or '<derive from task metadata>'} ({_path_status(benchmark_baseline_path)})",
        f"- Target source: {kernel_path or '<unknown>'} ({_path_status(kernel_path)})",
        f"- Gluon docs: profile={doc_profile}; status={doc_status}; docs are runner/tool contract, not a worker full-read list.",
        "",
        "### Machine Index (Do Not Treat As A Second Task)",
        f"- Required output dialect: {required_output}",
        f"- Source origin: {source_origin}",
        f"- Task type: {task_type}",
        f"- Patch target symbols: {', '.join(required_symbols) if required_symbols else '<none declared>'}",
        f"- Executed route symbols: {', '.join(executed_route_symbols) if executed_route_symbols else '<derive from source/harness>'}",
        f"- Target component: {target_component}",
        f"- Failure layers: {failure_layers}",
    ]
    if forbidden_symbols:
        lines.append(f"- Forbidden target symbols: {', '.join(forbidden_symbols)}")
    lines.extend(
        [
            "",
            "### Route-Proof Inputs",
            f"- KERNEL FILE TO EDIT: {kernel_path or '<unknown>'}",
            f"- REPO ROOT: {repo_root or '<unknown>'}",
            f"- Harness path: {harness or '<derive from COMMANDMENT>'}",
            f"- TEST COMMAND: {test_command or '<derive from COMMANDMENT>'}",
        ]
    )
    if benchmark_baseline:
        lines.extend(["- Benchmark baseline excerpt:", "```", "\n".join(_compact_benchmark_baseline(benchmark_baseline)), "```"])
    if cases:
        lines.append("- Benchmark cases:")
        for case in list(cases)[:6]:
            if isinstance(case, dict):
                lines.append(f"  - {case.get('case_id') or '?'} params={case.get('params') or {}}")
        if len(cases) > 6:
            lines.append(f"  - ... ({len(cases) - 6} more cases)")
    partition_hints = _partition_hints_from_cases(feature_metadata)
    if partition_hints:
        lines.append("- Partition/shape hints:")
        lines.extend(f"  - {hint}" for hint in partition_hints)
    if missing_docs:
        lines.append("- Missing required doc paths:")
        lines.extend(f"  - {path}" for path in missing_docs[:5])
    if doc_heading_hints:
        lines.append("- Optional doc heading hints for implementation details:")
        lines.extend(f"  - {hint}" for hint in doc_heading_hints)
    candidate_lines = _candidate_config_lines(task_body)
    if candidate_lines:
        lines.extend(
            [
                "",
                "### Candidate Experiment Protocol",
                "- The task names explicit candidate configurations. Treat each candidate as a separate round or an equivalent per-candidate artifact.",
                "- For each candidate, record config, patch/test artifact, per-shape correctness/benchmark, and keep/reject decision.",
                "- Do not rank a candidate unless compile, correctness, and full-benchmark are captured by `save_and_test` or an approved round-sandbox artifact.",
            ]
        )
        lines.extend(f"- {line}" for line in candidate_lines)
    lines.extend(
        [
            "",
            "### Minimal Execution Contract",
            "- Write `strategy_notes.md` or `route_proof.md` before the first edit.",
            "- Route proof fields: current route, target route, guards, output feeding, reduce/temporary preserve-or-skip, same ABI, measurement boundary.",
            "- Use `save_and_test`; raw bash benchmark output is diagnostic only and is not ranked.",
            "- For existing AMD Gluon, in-place edits to the executed target body are valid if no new unexecuted helper/fallback is introduced.",
            "",
            "### Original Task Body",
            task_body.strip() or "<empty task body>",
            "",
        ]
    )
    return lines


def _build_gluon_protocol_working_set(
    feature_metadata: dict[str, Any] | None,
    *,
    gluon_always_read_path: str | None = None,
) -> list[str]:
    feature_metadata = feature_metadata or {}
    entrypoint = str(gluon_always_read_path or "skills/triton-gluon/docs/00_always_read.md").strip()
    source_origin = str(feature_metadata.get("source_origin") or "unknown").strip()
    target_component = str(feature_metadata.get("target_component") or "").strip()
    failure_layers = str(feature_metadata.get("failure_layers") or "").strip()
    allowed_execution_path = str(feature_metadata.get("allowed_execution_path") or "").strip()
    lines = [
        "## Gluon Working Set",
        "- Reading order is hard: task packet -> COMMANDMENT -> required docs -> source -> harness -> route proof -> edit.",
        f"- Split-doc entrypoint: `{entrypoint}`. Use it to route headings; do not paste or reread unrelated doc sections.",
        "- Before the first edit, create `strategy_notes.md` (or update the configured strategy artifact) with:",
        "  - `Task objective extraction`",
        "  - `Gluon knowledge lookup plan`",
        "  - `Gluon implementation plan`",
        "  - `Performance hypothesis`",
        "  - `Same ABI comparison`",
        "  - `Required execution route proof`",
        "  - `Measurement boundary reconciliation`",
        "  - `Patch evolution ledger`",
        "- Lookup rows may start as `viewed=no`, but all required paths must be viewed before save_and_test. If the implementation plan cannot satisfy the scoped path fields from the docs, reduce scope before editing.",
        "- Required execution route proof must name current wrapper/kernel path, target wrapper/kernel path, guard conditions, output feeding, reduce/temporary skip-or-preserve behavior, same ABI proof, and measurement boundary reconciliation.",
        "",
        "## Patch Evolution Working Set",
        "- Use `10_search_policies.md::atomic_component_lattice` / `l0_scope_decision_before_emit` plus `60_real_patterns.md::patch_evolution_by_task_type` / `failure_to_next_patch_map` for pass/fail next-patch rules.",
        "- For `patch_0`, name one `primary_component`; Branch A local smoke/probe stays local, and Branch B whole-helper skeleton starts as compile/wiring/layout evidence.",
        "- Patch evolution ledger fields per round: Changed component, Expected effect, Observed effect, Keep/Revert, Next patch allowed scope.",
        "- Failure-layer lock: `helper_not_executed` or `target_not_touched` -> wiring/output-feeding only; `scope_violation` -> revert forbidden scope; `slow_correct` -> one named removable overhead; `compile/layout` -> fix only that layer.",
        "- For full_operator + wrapper_shape_dispatch/reduction tasks, patch_0 is wrapper dispatch/output feeding only unless the task explicitly allows kernel-body rewrite.",
        "",
        "## Task consistency check",
        "- Compare task body, metadata, source, and routed docs before editing. Use `10_search_policies.md::overlay_direction_vs_mechanism`, `60_real_patterns.md::Task consistency check`, `l0_scope_by_kernel_family`, `Whole-kernel L0 comparison anchor`, and a parent-layout map when relevant.",
        "- If metadata is inconsistent with source evidence, record `Task correction` and stay within the existing task boundary.",
        f"- Source origin: `{source_origin}`.",
        f"- Failure layers: {failure_layers or '<not declared>'}.",
        f"- Target component: {target_component or '<not declared>'}.",
        f"- Allowed execution path: `{allowed_execution_path or '<derive from task>'}`.",
        "- Required AMD Gluon patches must be real executed Gluon paths, not import-only, helper-only, empty, or plain Triton fallbacks.",
    ]
    if allowed_execution_path == "inline_scoped_helper":
        lines.append(
            "- Inline scoped helper boundary: do not add a replacement whole-kernel `@gluon.jit`, do not reroute the wrapper's main path to a new full Gluon kernel, and do not report success by widening beyond the named target component."
        )
    optional_fields = [
        ("Extension intent", "extension_intent", True),
        ("Expected outcome", "expected_outcome", False),
        ("L0 scope classification", "l0_scope_classification", True),
        ("L0 coupling reasons", "l0_coupling_reasons", False),
        ("Expected failure layers", "expected_failure_layers", False),
        ("First patch compile goal", "first_patch_compile_goal", False),
        ("Do not optimize before compile", "do_not_optimize_before_compile", True),
        ("Matrix lowering required", "matrix_lowering_required", True),
        ("Task signals", "task_signals", False),
        ("Routed doc reasons", "routed_doc_reasons", False),
        ("Kernel family signal", "kernel_family_signal", True),
        ("Minimum executable unit", "minimum_executable_unit", True),
        ("Scope infeasible policy", "scope_infeasible_policy", True),
        ("Whole kernel required reason", "whole_kernel_required_reason", False),
    ]
    for label, key, code_style in optional_fields:
        value = feature_metadata.get(key)
        if value in (None, ""):
            continue
        text = str(value)
        lines.append(f"- {label}: `{text}`." if code_style else f"- {label}: {text}")
    lines.append("")
    return lines


def _build_required_gluon_contract_context() -> list[str]:
    """Return compact triton-gluon contract guidance without invoking skill selection."""
    skill_path = get_repo_root() / "skills" / "triton-gluon" / "SKILL.md"
    always_read_path = get_repo_root() / "skills" / "triton-gluon" / "docs" / "00_always_read.md"
    lines = [
        "## Required AMD Gluon Contract",
        "The triton-gluon skill/docs are required as contract inputs, but the worker should follow the clean task packet first.",
        f"Skill path: `{skill_path}`",
        f"Split-doc entrypoint: `{always_read_path}`",
        "Use only the supported Gluon imports:",
        "```python",
        "from triton.experimental import gluon",
        "from triton.experimental.gluon import language as gl",
        "```",
        "Do NOT probe `from triton import gluon`; that is not the supported import path.",
        "Before save_and_test, view every absolute path listed in the REQUIRED DOC PATHS block.",
        "",
        "Before the first edit, persist `strategy_notes.md` with `Task objective extraction`, `Required execution route proof`, `Measurement boundary reconciliation`, and `Patch evolution ledger`.",
        "Required AMD Gluon tasks must not add broad `try/except Exception` fallback that silently succeeds through the plain Triton path.",
        "If the task names `Target symbol`, `required_patch_target_symbols`, or scoped items in `Allowed change`, only report success when that target path is modified and executes the intended Gluon code.",
        "Keep scope to one subpath/component unless the task explicitly says `bundle_allowed=true`.",
        "If task failure-layer metadata is missing or inconsistent with source evidence, record `Task correction` in notes/summary and continue within the existing task boundary rather than adding new metadata fields.",
    ]
    lines.append("")
    return lines


def _build_shape_coverage_working_set(feature_metadata: dict[str, Any] | None) -> list[str]:
    """Return generic multi-shape working-set rules for the worker prompt.

    These rules apply to any Triton-family kernel (and to HIP / CK kernels
    that opt into the Gluon feature meta) whose benchmark exposes more
    than one shape. They intentionally avoid Gluon-specific vocabulary so
    plain Triton or HIP workers also get the multi-shape contract.
    """
    feature_metadata = feature_metadata or {}
    profile = str(
        feature_metadata.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE
    ).lower()
    if profile not in (SHAPE_COVERAGE_MULTI, SHAPE_COVERAGE_BUCKETED):
        return []
    shape_count = feature_metadata.get("benchmark_shape_count") or "?"
    cases = feature_metadata.get("benchmark_test_cases") or []

    lines = [
        "## Shape Coverage Working Set",
        f"- Multi-shape contract: this benchmark runs `{shape_count}` cases; "
        "the same set is used for correctness and performance.",
        "- Avoid hardcoding any of `M`, `N`, `K`, `seq_len`, batch, or hidden size unless your patch is "
        "explicitly shape-bucketed with a documented host-side dispatch.",
        "- Per-shape correctness MUST hold for every case. A regression on any single shape disqualifies a "
        "'win' unless an alternative shape-bucketed path covers that case.",
        "- Compare per-shape baseline vs candidate, not only overall geomean. State per-case timings in "
        "your save_and_test summary when the harness exposes them.",
    ]
    if profile == SHAPE_COVERAGE_BUCKETED:
        lines.append(
            "- Bucketed coverage: prefer explicit host-side dispatch that chooses shape-specialized kernels "
            "or launch parameters (small / medium / large). Do not hide bucket selection inside "
            "`@triton.heuristics`, and do not let a single tile size dominate every bucket."
        )
    if cases:
        lines.append("- Observed cases:")
        for case in list(cases)[:6]:
            params = case.get("params") if isinstance(case, dict) else {}
            cid = case.get("case_id") if isinstance(case, dict) else None
            lines.append(f"  - `{cid or '?'}` params={params or {}}")
        if len(cases) > 6:
            lines.append(f"  - ... ({len(cases) - 6} more cases)")
    lines.append("")
    return lines


# ── pipeline context injection ───────────────────────────────────────


def inject_pipeline_context(
    task_body: str,
    config: dict,
    *,
    task_file_path: str | None = None,
    commandment_path: str | None = None,
    baseline_metrics_path: str | None = None,
    benchmark_baseline_path: str | None = None,
    commandment_text: str | None = None,
    baseline_metrics: dict | None = None,
    profiling_path: str | None = None,
    kernel_path: str | None = None,
    repo_root: str | None = None,
    test_command: str | None = None,
    codebase_context: str | None = None,
    benchmark_baseline: str | None = None,
    feature_metadata: dict[str, Any] | None = None,
    knowledge_base_path: str | None = None,
    gluon_skill_path: str | None = None,
    gluon_kb_path: str | None = None,
    gluon_examples_path: str | None = None,
    gluon_always_read_path: str | None = None,
    gluon_search_policies_path: str | None = None,
    gluon_component_traits_path: str | None = None,
    gluon_architecture_notes_path: str | None = None,
    gluon_examples_doc_path: str | None = None,
    gluon_api_reference_path: str | None = None,
    gluon_real_patterns_path: str | None = None,
    gluon_backup_details_path: str | None = None,
    gluon_doc_gate_required_paths: list[str] | None = None,
) -> tuple[str, dict]:
    """Prepend pipeline context to *task_body* and augment *config*.

    This is the single canonical context-injection path.  Both
    ``dispatch.task_file_to_agent_task`` and the ``geak`` parallel path
    call this so that every agent -- regardless of dispatch route --
    receives identical pipeline context.

    Returns ``(enriched_body, updated_config)``.
    """

    cfg = dict(config)
    gluon_worker_context = bool(
        feature_metadata
        and _task_needs_gluon_worker_context(
            feature_metadata,
            task_body,
        )
    )
    gluon_route_priority = _gluon_route_priority_task(feature_metadata, task_body)
    if gluon_worker_context:
        cfg["gluon_route_proof_required"] = True
        cfg["gluon_strategy_artifacts_required"] = True
        cfg["gluon_route_priority_override"] = gluon_route_priority
        cfg["gluon_failure_layers"] = str(feature_metadata.get("failure_layers") or "")
        # The clean packet already carries route/task evidence. Low-similarity
        # cross-session memory has been a source of off-route Gluon patches.
        cfg["suppress_cross_session_memory"] = True
    front_ctx: list[str] = []
    if gluon_worker_context:
        front_ctx.extend(
            _build_gluon_clean_task_packet(
                task_body,
                feature_metadata,
                task_file_path=task_file_path,
                commandment_path=commandment_path,
                baseline_metrics_path=baseline_metrics_path,
                benchmark_baseline_path=benchmark_baseline_path,
                kernel_path=kernel_path,
                repo_root=repo_root,
                test_command=test_command,
                commandment_text=commandment_text,
                benchmark_baseline=benchmark_baseline,
                gluon_doc_gate_required_paths=gluon_doc_gate_required_paths,
            )
        )
    ctx: list[str] = []
    if not gluon_worker_context:
        ctx = [
            "## Pipeline Context (auto-injected from task metadata)",
            "",
        ]
        if kernel_path:
            ctx.append(f"KERNEL FILE TO EDIT: {kernel_path}")
        if repo_root:
            ctx.append(f"REPO ROOT: {repo_root}")
        if test_command:
            ctx.append(f"TEST COMMAND: {test_command}")
        ctx.append("")

    if feature_metadata and not gluon_worker_context:
        ctx.append(
            build_gluon_feature_prompt_block(
                feature_metadata,
                heading="## Gluon Feature Context (auto-injected from task metadata)",
            )
        )
        ctx.append("")
        if _task_requires_amd_gluon_output(feature_metadata):
            ctx.extend(_build_required_gluon_contract_context())
        # Apply shape-coverage rules to ALL kernel types, not only Gluon paths.
        ctx.extend(_build_shape_coverage_working_set(feature_metadata))

    if not gluon_worker_context:
        ctx.append(
            "IMPORTANT: Only edit files within your REPO ROOT directory. "
            "Do NOT search or modify files outside of it. "
            "The KERNEL FILE TO EDIT path above is the exact file you should optimize."
        )
        ctx.append("")

    if commandment_text and not gluon_worker_context:
        ctx.append("## COMMANDMENT (evaluation contract -- you MUST follow these rules)")
        ctx.append(commandment_text.strip())
        ctx.append("")

    if baseline_metrics and not gluon_worker_context:
        dur = baseline_metrics.get("duration_us", "unknown")
        bn = baseline_metrics.get("bottleneck", "unknown")
        ctx.append("## Baseline Performance (your optimization must improve on these)")
        ctx.append(f"Total duration: {dur} us")
        ctx.append(f"Bottleneck: {bn}")
        top = baseline_metrics.get("top_kernels", [])
        if top:
            ctx.append("Top kernels by duration:")
            for k in top[:5]:
                bn_tag = f" [{k['bottleneck']}]" if k.get("bottleneck") else ""
                ctx.append(
                    f"  - {k.get('name', '?')}: {k.get('duration_us', '?')} us ({k.get('pct_of_total', '?')}%){bn_tag}"
                )
        ctx.append("")

        if gluon_route_priority:
            ctx.extend(
                [
                    "## Optimization Guidance (Gluon route-priority task)",
                    "This task is gated by full-operator route/wrapper/output-feeding evidence. Do not substitute generic kernel-body rewrites for the declared wrapper/reduction failure layer.",
                    "Prioritize: (1) route proof, (2) wrapper dispatch/shape guard, (3) measured output feeding and temporary/reduce path skip-or-preserve, (4) one named removable overhead only after a correct executed route exists.",
                    "",
                ]
            )
        else:
            ctx.extend(_bottleneck_guidance(str(bn), baseline_metrics, arch=detect_gpu_arch()))

    if profiling_path and Path(profiling_path).exists() and not gluon_worker_context:
        ctx.append(f"PROFILING DATA: {profiling_path}")
        ctx.append("(Read this file for detailed per-kernel profiling metrics)")
        ctx.append("")

        ctx.extend(_gpu_arch_context(profiling_path))

    if benchmark_baseline and not gluon_worker_context:
        ctx.append("## Benchmark Baseline (compare your save_and_test output against this)")
        ctx.append(
            "This is the original kernel's canonical benchmark output from the same full benchmark contract used for patch testing."
        )
        ctx.append("Your save_and_test output includes canonical benchmark results -- compare against these numbers.")
        ctx.append(f"```\n{benchmark_baseline.strip()}\n```")
        ctx.append("")

    if codebase_context and not gluon_worker_context:
        ctx.append("## Codebase Context (kernel dependency tree)")
        ctx.append(
            "The dependency tree below shows in-repo files the target kernel "
            "imports. Every listed dependency is a potential optimization "
            "target -- improving any of them can reduce overall latency."
        )
        ctx.append(codebase_context.strip())
        ctx.append("")
        cfg["codebase_context"] = codebase_context.strip()

    if not gluon_worker_context:
        ctx.append(
            "IMPORTANT: Baseline profiling and performance metrics are already "
            "established and provided above. Do NOT run save_and_test for a "
            "baseline run. Start optimizing immediately."
        )
        ctx.append("")

    try:
        integration = importlib.import_module("minisweagent.memory.integration")
        assemble_memory_context = getattr(integration, "assemble_memory_context", None)
        if assemble_memory_context is not None and not cfg.get("suppress_cross_session_memory"):
            _bm = baseline_metrics or {}
            _mem_ctx = assemble_memory_context(
                kernel_path=kernel_path,
                bottleneck_type=_bm.get("bottleneck"),
                profiling_metrics=_bm,
            )
            if _mem_ctx:
                ctx.append("## Optimization Memory (from past kernel optimization runs)")
                ctx.append(_mem_ctx.strip())
                ctx.append("")
    except Exception:
        logger.debug("Could not assemble optimization memory context", exc_info=True)

    if front_ctx:
        enriched = "\n".join(front_ctx + ctx)
    else:
        enriched = "\n".join(ctx) + "\n" + task_body
    return enriched, cfg


# ── baseline profiling (via profiler-mcp, with warmup) ───────────────


def run_baseline_profile(test_command: str, gpu_id: int = 0) -> dict:
    """Profile the test harness via profiler-mcp (includes warmup).

    Uses ``profiler_mcp.server.profile_kernel`` which performs backend-agnostic
    warmup runs before the actual instrumented profiling pass.
    """
    _ensure_mcp_importable()
    profile_server = importlib.import_module("profiler_mcp.server")
    profile_kernel = profile_server.profile_kernel

    harness = extract_harness_path(test_command)
    profile_cmd = f"python {harness} --profile"

    _profile_fn = getattr(profile_kernel, "fn", profile_kernel)
    return _profile_fn(
        command=profile_cmd,
        backend="metrix",
        num_replays=3,
        quick=False,
        gpu_devices=str(gpu_id),
    )
