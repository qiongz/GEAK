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
    feature_uses_gluon_guidance_from_meta,
)
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


def _build_gluon_working_set(feature_metadata: dict[str, Any] | None) -> list[str]:
    feature_metadata = feature_metadata or {}
    input_dialect = str(feature_metadata.get("input_dialect") or "").strip().lower()

    lines = [
        "## Gluon Working Set",
        "- Use targeted reading: start with `skills/triton-gluon/docs/00_always_read.md`, use its `stable_split_doc_index` and `Task routing` table, then jump to the exact split-doc headings relevant to this task.",
        "- Do not implement from memory or guess Gluon API names. Read the routed split-doc entry before writing a Gluon patch; use `70_backup_details.md` when primary split docs lack detail, then report the missing route instead of relying on long-form backup.",
        "- `save_and_test` enforces this for Gluon tasks: use `str_replace_editor` with `command=\"view\"` on the required absolute split-doc paths before saving or benchmarking a patch.",
        "- Before editing, write a `Gluon knowledge lookup plan` in your strategy notes: task signals, exact split-doc files/headings, viewed=yes/no for each, source sections viewed, and missing details.",
        "- Do not start implementation while a required lookup row is still `viewed=no`; either view that section or record the exact missing-doc route.",
        "- Preserve launcher shape, indexing, masks, correctness behavior, and benchmark intent before changing algorithms.",
        "- Before editing a Gluon path, write a `Gluon implementation plan`, `Performance hypothesis`, and `Patch evolution` plan. Use `00_always_read.md` for required fields, `20_component_traits.md` for layout/memory/matrix traits, `50_api_reference.md` for exact API patterns, `60_real_patterns.md` for L0/L1 evolution, and `10_search_policies.md` for round composition rules.",
        "- Required AMD Gluon patches must be real executed Gluon paths, not import-only, helper-only, empty, or plain Triton fallbacks.",
        "- Keep task scope narrow: one subpath/component unless `bundle_allowed=true`; use the docs for detailed rejection conditions and fix order.",
    ]

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
                "- For `plain_triton -> amd_gluon`, first obtain a minimal compileable and correctness-passing amd_gluon baseline before trying persistent scheduling, split-K, or atomics.",
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
        "Before editing the kernel or calling `save_and_test`, use `str_replace_editor` with `command=\"view\"` on every required Triton-Gluon doc path below.",
        "`save_and_test` will fail with `GLUON_DOC_GATE_FAILED` until these exact paths have been viewed.",
    ]
    lines.extend(f"- {path}" for path in paths)
    lines.append("")
    return lines


def _task_requires_amd_gluon_output(feature_metadata: dict[str, Any] | None) -> bool:
    """Return whether the task contract requires a real AMD Gluon patch."""
    if not feature_metadata:
        return False
    return (
        str(feature_metadata.get("kernel_type") or "").strip().lower() == "triton"
        and str(feature_metadata.get("required_output_dialect") or "").strip().lower() == "amd_gluon"
    )


def _build_forced_triton_gluon_skill_context() -> list[str]:
    """Inline triton-gluon skill guidance for required AMD Gluon tasks."""
    skill_path = get_repo_root() / "skills" / "triton-gluon" / "SKILL.md"
    always_read_path = get_repo_root() / "skills" / "triton-gluon" / "docs" / "00_always_read.md"
    lines = [
        "## Forced Triton-Gluon Skill Context",
        "This task requires a real AMD Gluon output. The triton-gluon skill is mandatory for this task.",
        f"Skill path: `{skill_path}`",
        f"Split-doc entrypoint: `{always_read_path}`",
        "Use only the supported Gluon imports:",
        "```python",
        "from triton.experimental import gluon",
        "from triton.experimental.gluon import language as gl",
        "```",
        "Do NOT probe `from triton import gluon`; that is not the supported import path.",
        "Plain Triton fallback is only valid after a real Gluon patch using `triton.experimental.gluon`, `@gluon.jit`, or `gl.*` has been saved/tested and failed with a recorded compile/runtime error.",
        "",
        "Before the first edit, write the `Gluon knowledge lookup plan`, `Gluon implementation plan`, `Performance hypothesis`, and `Patch evolution` blocks defined in `SKILL.md` and `00_always_read.md`.",
        "If lookup rows are still viewed=no, view those sections before editing. If the implementation plan cannot satisfy the scoped path fields from the docs, reduce scope before editing.",
    ]
    if skill_path.is_file():
        try:
            lines.extend(["", "### triton-gluon/SKILL.md", "```markdown", skill_path.read_text(), "```"])
        except OSError:
            logger.debug("Could not read forced triton-gluon skill at %s", skill_path)
    if always_read_path.is_file():
        try:
            lines.extend(
                [
                    "",
                    "### triton-gluon/docs/00_always_read.md",
                    "```markdown",
                    always_read_path.read_text(),
                    "```",
                ]
            )
        except OSError:
            logger.debug("Could not read forced triton-gluon entrypoint at %s", always_read_path)
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
    ctx: list[str] = [
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

    if feature_metadata:
        ctx.append(
            build_gluon_feature_prompt_block(
                feature_metadata,
                heading="## Gluon Feature Context (auto-injected from task metadata)",
            )
        )
        ctx.append("")
        if feature_uses_gluon_guidance_from_meta(feature_metadata):
            ctx.extend(
                _build_gluon_reference_block(
                    knowledge_base_path=knowledge_base_path,
                    gluon_skill_path=gluon_skill_path,
                    gluon_kb_path=gluon_kb_path,
                    gluon_examples_path=gluon_examples_path,
                    gluon_always_read_path=gluon_always_read_path,
                    gluon_search_policies_path=gluon_search_policies_path,
                    gluon_component_traits_path=gluon_component_traits_path,
                    gluon_architecture_notes_path=gluon_architecture_notes_path,
                    gluon_examples_doc_path=gluon_examples_doc_path,
                    gluon_api_reference_path=gluon_api_reference_path,
                    gluon_real_patterns_path=gluon_real_patterns_path,
                    gluon_backup_details_path=gluon_backup_details_path,
                )
            )
            ctx.extend(_build_gluon_working_set(feature_metadata))
            ctx.extend(_build_required_gluon_doc_gate_block(gluon_doc_gate_required_paths))
        if _task_requires_amd_gluon_output(feature_metadata):
            ctx.extend(_build_forced_triton_gluon_skill_context())
        # Apply shape-coverage rules to ALL kernel types, not only Gluon paths.
        ctx.extend(_build_shape_coverage_working_set(feature_metadata))

    ctx.append(
        "IMPORTANT: Only edit files within your REPO ROOT directory. "
        "Do NOT search or modify files outside of it. "
        "The KERNEL FILE TO EDIT path above is the exact file you should optimize."
    )
    ctx.append("")

    if commandment_text:
        ctx.append("## COMMANDMENT (evaluation contract -- you MUST follow these rules)")
        ctx.append(commandment_text.strip())
        ctx.append("")

    if baseline_metrics:
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

        ctx.extend(_bottleneck_guidance(str(bn), baseline_metrics, arch=detect_gpu_arch()))

    if profiling_path and Path(profiling_path).exists():
        ctx.append(f"PROFILING DATA: {profiling_path}")
        ctx.append("(Read this file for detailed per-kernel profiling metrics)")
        ctx.append("")

        ctx.extend(_gpu_arch_context(profiling_path))

    if benchmark_baseline:
        ctx.append("## Benchmark Baseline (compare your save_and_test output against this)")
        ctx.append(
            "This is the original kernel's canonical benchmark output from the same full benchmark contract used for patch testing."
        )
        ctx.append("Your save_and_test output includes canonical benchmark results -- compare against these numbers.")
        ctx.append(f"```\n{benchmark_baseline.strip()}\n```")
        ctx.append("")

    if codebase_context:
        ctx.append("## Codebase Context (kernel dependency tree)")
        ctx.append(
            "The dependency tree below shows in-repo files the target kernel "
            "imports. Every listed dependency is a potential optimization "
            "target -- improving any of them can reduce overall latency."
        )
        ctx.append(codebase_context.strip())
        ctx.append("")
        cfg["codebase_context"] = codebase_context.strip()

    ctx.append(
        "IMPORTANT: Baseline profiling and performance metrics are already "
        "established and provided above. Do NOT run save_and_test for a "
        "baseline run. Start optimizing immediately."
    )
    ctx.append("")

    try:
        integration = importlib.import_module("minisweagent.memory.integration")
        assemble_memory_context = getattr(integration, "assemble_memory_context", None)
        if assemble_memory_context is not None:
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
