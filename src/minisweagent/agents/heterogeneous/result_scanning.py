"""Scan prior-round results and task directories.

Shared utility used by both the orchestrator's ``tool_collect_results``
and the task generator's ``_run_task_agent`` to summarize what happened
in previous rounds.
"""

from __future__ import annotations

import json
import logging
import re as _re
from pathlib import Path

logger = logging.getLogger(__name__)

_LATENCY_RE = _re.compile(r"GEAK_RESULT_LATENCY_MS=([\d.]+(?:e[+-]?\d+)?)")
_SPEEDUP_RE = _re.compile(r"GEAK_RESULT_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)")
_GEOMEAN_SPEEDUP_RE = _re.compile(r"GEAK_RESULT_GEOMEAN_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)")
_CORRECTNESS_RE = _re.compile(r"(ALL\s+PASS|CORRECTNESS\s+TEST\s+PASSED|Status:\s*ALL\s+PASS)", _re.IGNORECASE)
_CORRECTNESS_FAIL_RE = _re.compile(r"(FAIL|CORRECTNESS\s+TEST\s+FAILED|Status:\s*FAIL)", _re.IGNORECASE)


def _extract_metrics(content: str) -> dict[str, str]:
    """Extract structured metrics from test output content."""
    metrics: dict[str, str] = {}

    m = _GEOMEAN_SPEEDUP_RE.search(content) or _SPEEDUP_RE.search(content)
    if m:
        metrics["speedup"] = f"{float(m.group(1)):.4f}x"

    m = _LATENCY_RE.search(content)
    if m:
        metrics["latency_ms"] = m.group(1)

    if _CORRECTNESS_RE.search(content):
        metrics["correctness"] = "PASS"
    elif _CORRECTNESS_FAIL_RE.search(content):
        metrics["correctness"] = "FAIL"

    return metrics


def scan_single_round_results(results_dir: Path) -> list[str]:
    """Scan a single round's results directory and return Markdown sections.

    For each task directory:
    - Reads ``best_results.json`` for the authoritative best-patch selection
    - Scans ALL test outputs for structured GEAK_RESULT markers
    - Includes full content of the best patch's test output
    - Reports log status (errors/completed)
    """
    sections: list[str] = []
    task_dirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name not in ("worktrees",) and not d.name.startswith(".")
    )
    if not task_dirs:
        return sections

    for td in task_dirs:
        label = td.name
        patches = sorted(td.glob("patch_*.patch"))
        test_outputs = sorted(td.glob("patch_*_test.txt"))
        log_files = sorted(td.glob("*.log"))

        section = [f"### {label}"]
        section.append(f"- Patches produced: {len(patches)}")

        best_patch_id = None
        best_results_path = td / "best_results.json"
        if best_results_path.exists():
            try:
                br = json.loads(best_results_path.read_text())
                best_patch_id = br.get("best_patch_id")
                speedup = br.get("best_patch_speedup")
                baseline_ms = br.get("baseline_latency_ms")
                candidate_ms = br.get("candidate_latency_ms")
                section.append(
                    f"- **Best patch**: {best_patch_id}"
                    f" (speedup={speedup}x, baseline={baseline_ms}ms, candidate={candidate_ms}ms)"
                )
                required_dialect = br.get("required_output_dialect", "unknown")
                actual_dialect = br.get("actual_output_dialect", "unknown")
                dialect_ok = br.get("dialect_contract_satisfied")
                execution_ok = br.get("gluon_execution_contract_satisfied")
                if actual_dialect not in {"amd_gluon", "mixed"} and required_dialect not in {"amd_gluon", "mixed"}:
                    execution_ok = False
                target_symbols = br.get("required_patch_target_symbols") or []
                section.append(
                    "- Contracts: "
                    f"required_output_dialect={required_dialect}, "
                    f"actual_output_dialect={actual_dialect}, "
                    f"dialect_contract_satisfied={dialect_ok}, "
                    f"gluon_execution_contract_satisfied={execution_ok}, "
                    f"required_patch_target_symbols={target_symbols}"
                )
                if br.get("has_significant_shape_regression") is not None:
                    section.append(
                        "- Shape regression: "
                        f"has_significant_shape_regression={br.get('has_significant_shape_regression')}"
                    )
                else:
                    section.append("- Shape regression: has_significant_shape_regression=unknown")
                comparison_target = br.get("comparison_target")
                if comparison_target:
                    section.append(
                        "- Comparison target: "
                        f"{comparison_target}"
                        + (f", safe_anchor={br.get('safe_anchor')}" if br.get("safe_anchor") else "")
                    )
                if br.get("scope_compliant") is not None or br.get("forbidden_scope_violation"):
                    section.append(
                        "- Scope compliance: "
                        f"scope_compliant={br.get('scope_compliant')}"
                        + (
                            f", forbidden_scope_violation={br.get('forbidden_scope_violation')}"
                            if br.get("forbidden_scope_violation")
                            else ""
                        )
                    )
                if br.get("allowed_execution_path") or br.get("scope_infeasible_policy") or br.get("scope_infeasible_reported") is not None:
                    section.append(
                        "- Scope execution path: "
                        f"allowed_execution_path={br.get('allowed_execution_path') or 'unknown'}"
                        + (
                            f", scope_infeasible_policy={br.get('scope_infeasible_policy')}"
                            if br.get("scope_infeasible_policy")
                            else ""
                        )
                        + (
                            f", scope_infeasible_reported={br.get('scope_infeasible_reported')}"
                            if br.get("scope_infeasible_reported") is not None
                            else ""
                        )
                    )
                per_shape = br.get("per_shape_speedups") or {}
                if isinstance(per_shape, dict) and per_shape:
                    shape_parts = []
                    for shape, info in per_shape.items():
                        if isinstance(info, dict) and isinstance(info.get("speedup"), (int, float)):
                            shape_parts.append(f"{shape}={float(info['speedup']):.4f}x")
                    if shape_parts:
                        section.append("- Per-shape speedups: " + ", ".join(shape_parts))
                anchor_viability = str(br.get("gluon_l1_anchor_viability") or "unknown")
                try:
                    numeric_speedup = float(speedup)
                except (TypeError, ValueError):
                    numeric_speedup = 0.0
                shape_regression = br.get("has_significant_shape_regression")
                if anchor_viability == "unknown" and br.get("not_viable_for_l1") is True:
                    anchor_viability = "not_viable_for_l1"
                if (
                    anchor_viability == "unknown"
                    and str(br.get("extension_intent") or "").strip().lower() == "execution_anchor"
                    and br.get("not_viable_for_l1_if_slower_than_base") is True
                    and numeric_speedup < 1.0
                ):
                    anchor_viability = "not_viable_for_l1"
                if anchor_viability == "unknown" and actual_dialect in {"amd_gluon", "mixed"} and dialect_ok and execution_ok:
                    if numeric_speedup >= 1.0 and shape_regression is False:
                        anchor_viability = "viable_for_l1"
                    elif numeric_speedup < 0.5 or shape_regression is True:
                        anchor_viability = "not_viable_for_l1"
                    elif shape_regression is False:
                        anchor_viability = "neutral_or_slow_anchor"
                section.append(f"- Gluon L1 anchor viability: {anchor_viability}")
                if br.get("extension_intent"):
                    section.append(
                        "- Gluon extension intent: "
                        f"{br.get('extension_intent')}"
                        + (f", expected_outcome={br.get('expected_outcome')}" if br.get("expected_outcome") else "")
                        + (f", overhead_source={br.get('overhead_source') or br.get('overhead_source_to_record')}" if (br.get("overhead_source") or br.get("overhead_source_to_record")) else "")
                    )
                attribution = "unknown"
                if actual_dialect in {"amd_gluon", "mixed"} and dialect_ok and execution_ok:
                    if numeric_speedup >= 1.0 and shape_regression is False:
                        attribution = "Gluon-positive"
                    elif numeric_speedup > 0.0 and (shape_regression is not None or numeric_speedup < 1.0):
                        attribution = "Gluon-slower"
                elif br.get("gluon_informed") is True and actual_dialect == "plain_triton":
                    attribution = "Gluon-informed"
                elif required_dialect == "amd_gluon" and actual_dialect == "plain_triton":
                    attribution = "invalid-gluon-fallback"
                elif "gluon" in label.lower():
                    attribution = "Gluon-neutral"
                section.append(f"- Gluon result attribution: {attribution}")
                if br.get("llm_selection_analysis"):
                    section.append(f"- Selection: {br['llm_selection_analysis']}")
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("scan_single_round: bad best_results.json in %s: %s", label, exc)

        for tf in test_outputs:
            try:
                content = tf.read_text(errors="replace")
                metrics = _extract_metrics(content)

                patch_id = tf.stem.replace("_test", "")
                is_best = patch_id == best_patch_id

                if metrics:
                    parts = [f"{k}={v}" for k, v in metrics.items()]
                    marker = " **[BEST]**" if is_best else ""
                    section.append(f"- {tf.name}: {', '.join(parts)}{marker}")
                else:
                    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    tail = lines[-3:] if len(lines) >= 3 else lines
                    section.append(f"- {tf.name} (tail): {' | '.join(tail)}")

                if is_best:
                    tail_content = content[-3000:] if len(content) > 3000 else content
                    section.append(f"\n<details><summary>{tf.name} (best patch full output)</summary>\n")
                    section.append(f"```\n{tail_content.strip()}\n```\n</details>")
            except Exception as exc:
                logger.debug("scan_single_round: could not read test output %s: %s", tf.name, exc)
                section.append(f"- {tf.name}: (unreadable)")

        for lf in log_files[:2]:
            try:
                content = lf.read_text(errors="replace")[-1500:]
                if "ERROR" in content or "Traceback" in content:
                    error_lines = [
                        ln.strip() for ln in content.splitlines() if "error" in ln.lower() or "traceback" in ln.lower()
                    ]
                    section.append(
                        f"- Log ({lf.name}): ERRORS -- {error_lines[-1][:200] if error_lines else 'see log'}"
                    )
                else:
                    section.append(f"- Log ({lf.name}): completed")
            except Exception as exc:
                logger.debug("scan_single_round: could not read log %s: %s", lf.name, exc)

        sections.append("\n".join(section))

    return sections


def scan_previous_results(results_dir: Path) -> str:
    """Scan previous round results and build a combined summary.

    Accepts either a single round directory (e.g. results/round_1) or
    the parent results/ directory containing multiple round_N subdirs.
    Returns a Markdown summary covering ALL prior rounds.
    """
    sections: list[str] = []

    round_subdirs = (
        sorted(d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("round_"))
        if results_dir.is_dir()
        else []
    )

    if round_subdirs:
        for rd in round_subdirs:
            round_sections = scan_single_round_results(rd)
            if round_sections:
                sections.append(f"## {rd.name.replace('_', ' ').title()} Results\n")
                sections.extend(round_sections)
    else:
        single_sections = scan_single_round_results(results_dir)
        if single_sections:
            sections.append("## Previous Round Results\n")
            sections.extend(single_sections)

    if not sections:
        return ""

    return "\n\n".join(sections) + "\n"


def scan_previous_tasks(tasks_dir: Path, current_round: int) -> str:
    """Scan prior rounds' task directories and summarize what was planned.

    Reads YAML frontmatter (label, agent_type, priority) and the first
    ~200 chars of the task body from each .md task file under
    tasks/round_1 through tasks/round_{current_round - 1}.
    """
    sections: list[str] = []
    for r in range(1, current_round):
        round_dir = tasks_dir / f"round_{r}"
        if not round_dir.is_dir():
            continue
        task_files = sorted(round_dir.glob("*.md"))
        if not task_files:
            continue
        round_items: list[str] = []
        for tf in task_files:
            try:
                text = tf.read_text(errors="replace")
                parts = _re.split(r"^---\s*$", text, maxsplit=2, flags=_re.MULTILINE)
                if len(parts) >= 3:
                    import yaml as _yaml

                    fm = _yaml.safe_load(parts[1]) or {}
                    body_preview = parts[2].strip()[:200]
                else:
                    fm = {}
                    body_preview = text.strip()[:200]
                label = fm.get("label", tf.stem)
                agent_type = fm.get("agent_type", "unknown")
                priority = fm.get("priority", "?")
                # Surface dialect / policy on each bullet so downstream
                # signal helpers can tell apart Base Set plain-Triton tasks
                # from Gluon Extension Set tasks without re-parsing the
                # frontmatter. Without this the previous-round signal
                # would conflate a Base Set Triton win (e.g. ``speedup=1.5x``)
                # with a Gluon Extension Set win and wrongly expand the
                # next round's Gluon quota.
                input_dialect = fm.get("input_dialect")
                policy = fm.get("output_dialect_search_policy")
                meta_bits = [f"agent={agent_type}", f"priority={priority}"]
                if input_dialect:
                    meta_bits.append(f"input_dialect={input_dialect}")
                if policy:
                    meta_bits.append(f"policy={policy}")
                round_items.append(
                    f"- **{label}** ({', '.join(meta_bits)}): "
                    f"{body_preview}{'...' if len(body_preview) >= 200 else ''}"
                )
            except Exception as exc:
                logger.debug("scan_previous_tasks: could not read %s: %s", tf.name, exc)
                round_items.append(f"- {tf.name}: (unreadable)")

        if round_items:
            sections.append(f"## Round {r} Planned Tasks\n\n" + "\n".join(round_items))

    if not sections:
        return ""

    return "\n\n".join(sections) + "\n"
