#!/usr/bin/env python3
"""
Homogeneous Agent Runner - Run multiple identical agents in parallel.

This module provides a simplified interface to run ParallelAgent with
homogeneous configuration (all agents run the same task with identical settings).
"""

import copy
import json
import logging
import time
from pathlib import Path

from rich.console import Console

from minisweagent.agents.parallel_agent import BestPatchResult, ParallelAgent
from minisweagent.agents.optimization_agent import OptimizationAgent
from minisweagent.models import get_model
from minisweagent.run.pool_runner import build_fixed_tasks

logger = logging.getLogger(__name__)


def parse_gpu_ids(gpu_ids_str: str | None) -> list[int]:
    """Parse a gpu_ids spec into a list of ints.

    Accepts multiple human-friendly forms, all seen in the wild from
    both the LLM-extracted task config AND explicit ``--gpu-ids`` CLI:

      - ``"4,5,6,7"``      — comma-separated list
      - ``"4-7"``           — range (inclusive)
      - ``"0,1,4-7"``       — mixed
      - ``"4"``             — single GPU
      - ``None`` / ``""``   — defaults to ``[0]``

    Delegates to ``run/utils/config_editor._parse_gpu_ids_string`` for
    the parse; that same helper is the source of truth for
    ``num_parallel = len(gpu_ids)`` auto-derivation in apply_config_changes.
    """
    from minisweagent.run.utils.config_editor import _parse_gpu_ids_string

    result = _parse_gpu_ids_string(gpu_ids_str)
    return result if result else [0]


def run_fixed_mode(
    config: dict,
    task_content: str,
    model,
    env,
    env_class,
    env_kwargs: dict,
    agent_config: dict,
    repo: Path | None = None,
    num_parallel: int | None = None,
    gpu_ids: str | None = None,
    output_dir: Path | None = None,
    model_name: str | None = None,
    console: Console | None = None,
) -> BestPatchResult | None:
    """Run ``fixed`` mode: N identical copies of the same task body in parallel.

    Every copy runs the same task prompt through its own ``OptimizationAgent``
    instance on its own GPU slot.  Variance across copies comes from LLM
    sampling alone (temperature > 0 or different tool-trajectory seeds).

    Called from ``run/unified.py::_run_fixed`` inside the unified round
    loop; not typically invoked directly.

    Args:
        config: Merged configuration dict
        task_content: Task description
        model: Model instance
        env: Environment instance
        env_class: Environment class for factory
        env_kwargs: Environment kwargs for factory
        tools_settings: Tools settings from config
        agent_config: Base agent configuration
        repo: Repository path for git worktree management
        num_parallel: Number of parallel agents
        gpu_ids: Comma-separated GPU IDs
        output_dir: Output directory
        model_name: Model name for factory
        console: Rich console for output

    Returns:
        The ParallelAgent instance after execution
    """
    if console is None:
        console = Console(highlight=False)

    # Parse configuration values
    parallel_config = config.get("parallel", {})

    # Number of parallel agents
    final_num_parallel = (
        num_parallel or parallel_config.get("num_parallel") or config.get("agent", {}).get("num_parallel") or 1
    )
    _np_source = (
        "arg"
        if num_parallel
        else "parallel config"
        if parallel_config.get("num_parallel")
        else "agent config"
        if config.get("agent", {}).get("num_parallel")
        else "default"
    )
    logger.debug("num_parallel=%d (source=%s)", final_num_parallel, _np_source)

    # GPU IDs
    final_gpu_ids = parse_gpu_ids(gpu_ids or parallel_config.get("gpu_ids") or config.get("agent", {}).get("gpu_ids"))
    logger.debug("gpu_ids=%s", final_gpu_ids)

    # Repository path
    final_repo = repo
    if not final_repo:
        final_repo = parallel_config.get("repo") or config.get("agent", {}).get("repo")

    final_repo = Path(final_repo).resolve()
    if not final_repo.exists():
        raise ValueError(f"Repository path does not exist: {final_repo}")

    # GEAK homogeneous flow always uses strategy interactive agent.
    base_agent_class = OptimizationAgent

    # Configure agent for homogeneous mode
    agent_config["mode"] = "yolo"
    agent_config["confirm_exit"] = False
    agent_config.setdefault("use_strategy_manager", True)
    agent_config["num_parallel"] = final_num_parallel
    agent_config["gpu_ids"] = final_gpu_ids
    agent_config["repo"] = str(final_repo)
    agent_config["agent_class"] = base_agent_class

    # Create output directory (pop from agent_config as ParallelAgentConfig doesn't accept it)
    final_output_dir = Path(agent_config.pop("output_dir", None) or output_dir or "optimization_logs")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Set patch_output_dir to output_dir so patches are saved alongside logs
    agent_config["patch_output_dir"] = str(final_output_dir)

    # Get model config for factory
    model_config = config.get("model", {})

    logger.info(
        "\n[bold cyan]%s[/bold cyan]\n  [bold]Homogeneous Agent[/bold] (%d agents, GPUs %s)\n[bold cyan]%s[/bold cyan]",
        "=" * 60,
        final_num_parallel,
        final_gpu_ids,
        "=" * 60,
    )
    logger.info("  repo=%s, output_dir=%s", final_repo, final_output_dir)
    logger.info("[dim]Sub-agents are working — expect no output for several minutes.[/dim]")

    # Build an identical-copies AgentTask list so fixed mode flows through
    # the shared ``run_pool`` scheduler.  Same pool, same worktrees, same
    # logs as the planned-mode path — only the task body differs.
    task_body_with_wt = task_content + "\n\n" + "The current worktree is: " + str(final_repo)
    fixed_tasks = build_fixed_tasks(
        num_parallel=final_num_parallel,
        agent_class=base_agent_class,
        task_body=task_body_with_wt,
        base_label="parallel",
    )
    # ParallelAgentConfig carries ``tasks`` alongside ``agent_class`` — when
    # ``tasks`` is set, ParallelAgent.run_parallel skips its inline fixed
    # branch and calls run_pool directly.
    agent_config["tasks"] = fixed_tasks

    agent = ParallelAgent(model, env, **agent_config)

    try:
        _t0 = time.monotonic()
        best_result = agent.run(
            task_body_with_wt,
            console=console,
            model_factory=lambda: get_model(model_name, model_config.copy()),
            env_factory=lambda: env_class(**copy.deepcopy(env_kwargs)),
        )
        _elapsed = time.monotonic() - _t0

        if best_result:
            logger.info(
                "Homogeneous run completed in %.0fs. Best patch: %s (agent %d)",
                _elapsed,
                best_result.patch_id,
                best_result.agent_id,
            )
            console.print(
                f"\n[bold green]Best patch:[/bold green] {best_result.patch_id} (agent {best_result.agent_id})"
            )
        else:
            logger.info("Homogeneous run completed in %.0fs. No best patch selected.", _elapsed)
            console.print("\n[bold yellow]No best patch selected[/bold yellow]")

        # Write final_report.json (aligned with heterogeneous output structure)
        speedup = best_result.best_speedup if best_result else None
        best_patch_path = (
            str(best_result.patch_dir / best_result.patch_id) if best_result and best_result.patch_dir else None
        )
        report = {
            "status": "complete" if best_result else "complete_no_patch",
            "best_patch": (best_result.best_patch_file or best_patch_path) if best_result else None,
            "best_speedup": speedup,
            "summary": best_result.llm_conclusion if best_result else "No best patch selected",
        }
        report_path = final_output_dir / "final_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Wrote final_report.json to %s", report_path)

    except Exception as e:
        logger.error("Homogeneous agent failed: %s", e, exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {e}")
        error_report = {
            "status": "error",
            "best_patch": None,
            "best_speedup": None,
            "summary": str(e),
        }
        error_report_path = final_output_dir / "final_report.json"
        try:
            error_report_path.write_text(json.dumps(error_report, indent=2, default=str))
        except Exception:
            logger.debug("Failed to write error report to %s", error_report_path)
        raise

    return best_result


# ------------------------------------------------------------------
# Back-compat alias — deprecated naming.
#
# The "homogeneous" terminology is a legacy artifact of pre-refactor
# code that used separate agent CLASSES for homogeneous vs
# heterogeneous dispatch.  With the unified ``OptimizationAgent``,
# the only difference is the task BODY (identical copies vs planner-
# generated strategies), so "fixed" / "planned" is the accurate
# naming.  This alias keeps old imports working for one release; new
# code must use ``run_fixed_mode``.
# ------------------------------------------------------------------

run_homogeneous_agent = run_fixed_mode
