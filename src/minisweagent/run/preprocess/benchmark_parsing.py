"""Deterministic benchmark output parsing and patch selection.

Provides regex-based extraction of latency metrics from harness output
and a ``compute_best_patch()`` function that selects the best non-empty
patch by comparing benchmark numbers -- no LLM involved.

Measurement methodology:
- Uses ``benchmark_baseline.txt`` (the canonical unmodified baseline benchmark)
- Prioritizes ``GEAK_RESULT_LATENCY_MS=<number>`` marker (standardized)
- Falls back to legacy parsers and universal latency keyword scanner
- Only reports speedups > 1.0 (genuine improvements over true baseline)
- Clamps LLM-inflated results to 1.0 when no real improvement exists
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from minisweagent.run.target_contracts import (
    do_not_clauses,
    forbidden_symbols_from_scoped_text,
    patch_touches_forbidden_target_symbol,
    target_symbols_from_scoped_text,
)
from minisweagent.run.postprocess.benchmark_parsing import (
    _legacy_classify_patch_output_dialect,
    classify_gluon_api_contract,
    classify_layout_contract,
    classify_patch_output_dialect as _postprocess_classify_patch_output_dialect,
)

logger = logging.getLogger(__name__)
_PER_SHAPE_REGRESSION_SPEEDUP_FLOOR = 0.95
_AMD_GLUON_PATCH_MARKERS = (
    "triton.experimental.gluon",
    "from triton.experimental import gluon",
    "@gluon.jit",
    "AMDMFMALayout",
    "AMDWMMALayout",
    "DotOperandLayout",
    "buffer_load",
    "buffer_store",
)
_PLAIN_TRITON_PATCH_MARKERS = (
    "@triton.jit",
    "triton.language",
    "tl.load",
    "tl.store",
    "tl.dot",
)


def parse_median_latency_ms(output: str) -> float | None:
    """Extract median latency (ms) from harness benchmark output."""
    m = re.search(
        r"(?:[Mm]edian\s+(?:latency|time)[\w\s]*|total\s+median\s+time)\s*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
        output,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def parse_total_kernel_time_ms(output: str) -> float | None:
    """Extract TOTAL_KERNEL_TIME_MS or BENCHMARK_LATENCY_MS from harness benchmark output."""
    m = re.search(
        r"(?:TOTAL_KERNEL_TIME_MS|BENCHMARK_LATENCY_MS):\s*([\d.]+(?:e[+-]?\d+)?)",
        output,
    )
    return float(m.group(1)) if m else None


def _parse_benchmark_metric(output: str) -> float | None:
    """Extract from BENCHMARK_METRIC:, median_latency_ms:, or Geomean (ms): lines."""
    for pat in (
        r"BENCHMARK_METRIC:\s*median_latency_ms=([\d.]+(?:e[+-]?\d+)?)",
        r"median_latency_ms:\s*([\d.]+(?:e[+-]?\d+)?)",
        r"Geomean\s*\(ms\)\s*:\s*([\d.]+(?:e[+-]?\d+)?)",
    ):
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_google_benchmark_ms(output: str) -> float | None:
    """Parse Google Benchmark format: <name> <iters> <latency> ms."""
    m = re.search(r"^\S+\s+\d+\s+([\d.]+(?:e[+-]?\d+)?)\s+ms", output, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_shape_count(output: str) -> int | None:
    """Extract shape count from harness benchmark output."""
    m = re.search(r"(\d+)\s+shapes", output, re.IGNORECASE)
    return int(m.group(1)) if m else None


_TEST_CASE_COUNT_PATTERNS = (
    r"measured\s+(\d+)\s+test\s+cases?",
    r"(\d+)\s+test\s+cases?",
    r"(\d+)\s+cases?\b",
    r"(\d+)\s+shapes?",
)


def parse_test_case_count(output: str) -> int | None:
    """Recognize multi-case wording across GEAK harness and AgentKernelArena.

    Handles patterns such as ``"5 shapes"``, ``"measured 5 test case(s)"``,
    or ``"5 cases"`` emitted by the various performance runners.
    """
    if not output:
        return None
    for pat in _TEST_CASE_COUNT_PATTERNS:
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            try:
                value = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def parse_performance_report_json(path: Any) -> list[dict] | None:
    """Read AgentKernelArena-style ``performance_report.json`` files.

    The format ``[{test_case_id, execution_time_ms, params}, ...]`` is
    AgentKernelArena's eval-script convention (``task_runner.py
    performance``, ``cal_kernel_perf.py``). GEAK consumes this file
    purely as **upstream raw data**: the GEAK preprocessor immediately
    snapshots the parsed cases into ``output_dir/benchmark_test_cases.json``
    so all downstream consumers (planner, dispatch, worker) read from a
    GEAK-canonical path inside this run's output_dir, never from the
    upstream Arena task dir.

    Returns a normalized list of test-case dicts with ``case_id``, ``ms``
    (best-effort latency), and ``params``. Returns ``None`` when the file
    is missing or malformed so callers can gracefully fall back to the
    free-text harness output.
    """
    try:
        report_path = Path(path)
    except TypeError:
        return None
    if not report_path.is_file():
        return None
    try:
        data = json.loads(report_path.read_text())
    except (OSError, ValueError):
        return None

    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("test_cases") or data.get("cases") or []
    else:
        return None

    def _first_present(raw_case: dict, keys: tuple[str, ...]) -> Any:
        # Pick the first key whose value is non-None. Skipping None matters for
        # ``torch2hip`` baseline-only output where ``opt_time`` is explicitly
        # ``None`` while ``ori_time`` carries the actual baseline latency.
        for key in keys:
            if key in raw_case and raw_case.get(key) is not None:
                return raw_case.get(key)
        return None

    def _positive_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    normalized: list[dict] = []
    for idx, raw in enumerate(cases):
        if not isinstance(raw, dict):
            continue
        case_id = raw.get("test_case_id") or raw.get("case_id") or raw.get("id") or f"case_{idx}"
        ms = _positive_float(
            _first_present(raw, ("execution_time_ms", "opt_time", "ori_time", "baseline_ms"))
        )
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        if not params and isinstance(raw.get("metadata"), dict):
            params = dict(raw.get("metadata") or {})
        if not params:
            for key in ("shape", "shapes", "input_shape", "input_shapes"):
                if key in raw:
                    params = {key: raw[key]}
                    break
        normalized.append(
            {
                "case_id": str(case_id),
                "ms": ms,
                "params": dict(params or {}),
            }
        )
    return normalized or None


# Public alias making the GEAK <-> upstream-Arena boundary explicit at
# call sites. ``parse_performance_report_json`` stays as the historical
# name because tests and external scripts already import it.
read_arena_perf_sidecar = parse_performance_report_json


def _candidate_task_runner_paths(performance_command: Any) -> list[Path]:
    """Pick task-runner / cal_kernel_perf script paths out of a perf command.

    AgentKernelArena tasks usually invoke the perf command via either
    ``python3 scripts/task_runner.py performance`` or
    ``python3 eval_tools/cal_kernel_perf.py``. The script's parent
    directory's parent is the task root, and that is where ``build/``
    lives. This helper extracts those file paths so the preprocessor can
    walk the right ancestor chain when searching for the report.
    """
    if not performance_command:
        return []
    text = performance_command if isinstance(performance_command, str) else " ".join(
        str(item) for item in performance_command
    )
    candidates: list[Path] = []
    for token in re.findall(r"[^\s'\"]+\.py", text):
        try:
            p = Path(token)
        except (OSError, ValueError):
            continue
        if p.name in {"task_runner.py", "cal_kernel_perf.py", "compile.py", "correctness_check.py"}:
            candidates.append(p)
    return candidates


def discover_performance_report(
    *,
    kernel_path: Any = None,
    repo_root: Any = None,
    output_dir: Any = None,
    performance_command: Any = None,
    extra_paths: Any = None,
    max_depth: int = 5,
) -> Path | None:
    """Locate the upstream Arena-style ``performance_report.json`` sidecar.

    This function returns a path inside the **upstream** AgentKernelArena
    task directory; the preprocessor immediately snapshots the parsed
    contents into GEAK's own ``output_dir/benchmark_test_cases.json`` so
    that downstream consumers never have to read the upstream sidecar
    again. See ``parse_performance_report_json`` for the boundary contract.

    The search order is:

    1. ``output_dir/performance_report.json`` (GEAK preprocess output).
    2. Any explicit ``extra_paths`` provided by the caller.
    3. Task-runner script paths extracted from ``performance_command``;
       for each script its parent's parent is the task dir. Relative
       command paths are also resolved against ``repo_root``.
    4. Walk up from ``kernel_path`` checking ``./build/performance_report.json``
       and ``./performance_report.json`` at each level, stopping at
       ``repo_root`` or after ``max_depth`` ascents.
    5. Legacy fallback: ``repo_root/build/performance_report.json``.

    Returns ``None`` when no candidate file exists.
    """
    seen: set[Path] = set()

    def _check(p: Any) -> Path | None:
        if p is None:
            return None
        try:
            candidate = Path(p)
        except (OSError, ValueError, TypeError):
            return None
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            return None
        seen.add(resolved)
        if candidate.is_file():
            return candidate
        return None

    if output_dir is not None:
        hit = _check(Path(output_dir) / "performance_report.json")
        if hit is not None:
            return hit

    if extra_paths:
        for ep in extra_paths:
            hit = _check(ep)
            if hit is not None:
                return hit

    for runner in _candidate_task_runner_paths(performance_command):
        # Real performance commands often use repo-relative paths like
        # ``python3 tasks/.../scripts/task_runner.py performance``. When the
        # current working directory is not the repo root the relative
        # ancestors won't resolve to anything useful, so always also probe
        # the path joined to ``repo_root`` when one is supplied.
        runner_candidates: list[Path] = [runner]
        if not runner.is_absolute() and repo_root:
            try:
                runner_candidates.append(Path(repo_root) / runner)
            except (OSError, ValueError, TypeError):
                pass
        for runner_path in runner_candidates:
            for ancestor in (runner_path.parent, runner_path.parent.parent):
                try:
                    if not ancestor:
                        continue
                    hit = _check(ancestor / "build" / "performance_report.json")
                    if hit is not None:
                        return hit
                    hit = _check(ancestor / "performance_report.json")
                    if hit is not None:
                        return hit
                except (OSError, ValueError):
                    continue

    if kernel_path is not None:
        try:
            kernel = Path(kernel_path)
        except (OSError, ValueError, TypeError):
            kernel = None
        if kernel is not None:
            try:
                root = Path(repo_root).resolve() if repo_root else None
            except (OSError, ValueError):
                root = None
            current = kernel.parent
            for _ in range(max_depth):
                hit = _check(current / "build" / "performance_report.json")
                if hit is not None:
                    return hit
                hit = _check(current / "performance_report.json")
                if hit is not None:
                    return hit
                if current == current.parent:
                    break
                if root is not None:
                    try:
                        if current.resolve() == root:
                            break
                    except OSError:
                        pass
                current = current.parent

    if repo_root is not None:
        hit = _check(Path(repo_root) / "build" / "performance_report.json")
        if hit is not None:
            return hit

    return None


def parse_shape_latencies_ms(output: str) -> dict[str, float]:
    """Extract per-shape latencies from harness benchmark output.

    Expected formats:
        ``(32,4096): 0.0503 ms``
        ``pa_decode_small: 0.1759 ms``
    """
    shape_latencies: dict[str, float] = {}
    for m in re.finditer(r"^\s*(\([^)]*\)):\s*([\d.]+(?:e[+-]?\d+)?)\s*ms\s*$", output, re.MULTILINE):
        shape_latencies[m.group(1)] = float(m.group(2))
    for m in re.finditer(
        r"^\s*([A-Za-z][A-Za-z0-9_.\-/]*)\s*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms\s*$",
        output,
        re.MULTILINE,
    ):
        case_id = m.group(1)
        # Skip generic summary labels that are already handled by explicit
        # latency parsers; per-case IDs should remain as shape keys.
        if case_id.lower() in {"median", "geomean", "mean", "total", "latency"}:
            continue
        shape_latencies[case_id] = float(m.group(2))
    return shape_latencies


def _parse_shape_total_latency_ms(output: str) -> float | None:
    """Return total latency across parsed per-shape benchmark lines."""
    shape_latencies = parse_shape_latencies_ms(output)
    if not shape_latencies:
        return None
    total = sum(shape_latencies.values())
    return total if total > 0 else None


def _geomean_ms(shape_latencies: dict[str, float]) -> float | None:
    """Return geometric mean latency for positive per-shape timings."""
    values = [value for value in shape_latencies.values() if value > 0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def extract_benchmark_config_lines(output: str) -> list[str] | None:
    """Extract benchmark config fingerprint lines from harness output.

    Captures the config/shape identifiers from each benchmark line,
    stripping timing numbers so only the problem description remains.
    This allows comparing whether baseline and candidate ran on the
    same benchmark configurations, regardless of kernel language or
    variable naming conventions.

    Works by finding lines that contain timing data (e.g. '0.0342ms')
    and extracting the config prefix before the first timing number.

    Examples of lines matched:
        'B=1 H=32 NQ=16 N_CTX=[512] ...  2.11ms   0.10ms  21.37x *'
        '(1, 16), k=2       0.0196ms   0.0335ms     0.58x'
        'Config (B=256,H=1024)   0.072ms  ...'

    Returns a sorted list of config identifiers, or None if no configs found.
    """
    configs: list[str] = []
    # Match lines with at least one timing value: "0.0342ms", "0.0342 ms", or
    # bare floats like "0.0342" in columns (common in table-formatted output).
    timing_pattern = re.compile(r"\d+\.\d+(?:ms|us|µs|s|x)?")
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith(("-", "=", "#", "Status", "Geometric", "GEAK_")):
            continue
        if not timing_pattern.search(line):
            continue
        # Skip header/summary lines
        if any(kw in line.lower() for kw in ("comparing", "running", "warmup", "median", "geomean", "mean")):
            continue
        # Extract config prefix: everything before the first timing value.
        # Handles multiple output formats:
        #   "M=128, N=16  0.0747  0.0474  1.58x"   → "M=128, N=16"
        #   "B=1 H=32 ... 2.11ms 0.10ms 21.37x"    → "B=1 H=32 ..."
        #   "(2, 4, 64): kernel=0.0411 ms | ref=..."→ "(2, 4, 64)"
        # Split on: =<float>, :<whitespace><float>, or <whitespace><float>
        config_part = re.split(r"(?<=[=:])\s*\d+\.\d+|\s+\d+\.\d+", line)[0].strip()
        # Clean trailing separators and labels that precede timing values
        config_part = re.sub(r"[\s:|]+$", "", config_part)
        config_part = re.sub(r"\s*\|\s*\w+$", "", config_part)
        config_part = re.sub(r":\s*\w+=$", "", config_part)
        if config_part and len(config_part) > 3:
            configs.append(config_part)
    return sorted(configs) if configs else None


def _universal_latency_fallback(text: str) -> float | None:
    """Last-resort: find a number near latency-related keywords in the last
    30 lines of output. Handles formats like 'Overall Median: 0.052ms'."""
    keywords = {"median", "overall", "geomean", "latency", "total"}
    candidates: list[float] = []
    lines = text.strip().splitlines()
    for line in lines[-30:]:
        lower = line.lower()
        if not any(kw in lower for kw in keywords):
            continue
        for m in re.finditer(r"([\d.]+(?:e[+-]?\d+)?)\s*ms", line):
            val = float(m.group(1))
            if 0.0001 < val < 100000:
                candidates.append(val)
    return candidates[-1] if candidates else None


def _extract_latency(text: str) -> float | None:
    """Extract latency from benchmark output.

    Priority:
    1. GEAK_RESULT_LATENCY_MS=<number> (standardized marker, always correct)
    2. Legacy format parsers (TOTAL_KERNEL_TIME_MS, BENCHMARK_METRIC, etc.)
    3. Universal fallback: last number near latency keywords in output
    """
    m = re.search(r"GEAK_RESULT_LATENCY_MS=([\d.]+(?:e[+-]?\d+)?)", text)
    if m:
        return float(m.group(1))

    val = parse_total_kernel_time_ms(text)
    if val is not None:
        return val
    val = _parse_benchmark_metric(text)
    if val is not None:
        return val
    val = parse_median_latency_ms(text)
    if val is not None:
        return val
    val = parse_google_benchmark_ms(text)
    if val is not None:
        return val
    val = _parse_shape_total_latency_ms(text)
    if val is not None:
        return val

    return _universal_latency_fallback(text)


def extract_latency_ms(text: str) -> float | None:
    """Public wrapper for standardized latency extraction."""
    return _extract_latency(text)


def extract_reported_speedup(text: str) -> float | None:
    """Extract a reported speedup scalar from benchmark output.

    Supported markers include:
    - ``GEAK_RESULT_GEOMEAN_SPEEDUP=<number>``
    - ``GEAK_RESULT_SPEEDUP=<number>``
    - ``Geometric mean speedup: <number>x``
    - ``Speedup (geomean): <number>x``
    """

    for pat in (
        r"GEAK_RESULT_GEOMEAN_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)",
        r"GEAK_RESULT_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)",
        r"Geometric mean speedup:\s*([\d.]+(?:e[+-]?\d+)?)x",
        r"Speedup\s*\(geomean\)\s*:\s*([\d.]+(?:e[+-]?\d+)?)x",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def compute_shape_speedups(
    baseline_shapes_ms: dict[str, float],
    candidate_shapes_ms: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute per-shape speedups for the overlap between baseline and candidate."""
    results: dict[str, dict[str, float]] = {}
    for shape, baseline_ms in baseline_shapes_ms.items():
        candidate_ms = candidate_shapes_ms.get(shape)
        if candidate_ms is None or baseline_ms <= 0 or candidate_ms <= 0:
            continue
        results[shape] = {
            "baseline_ms": round(baseline_ms, 6),
            "candidate_ms": round(candidate_ms, 6),
            "speedup": round(baseline_ms / candidate_ms, 6),
        }
    return results


def _has_significant_shape_regression(
    shape_speedups: dict[str, dict[str, float]],
    *,
    floor: float = _PER_SHAPE_REGRESSION_SPEEDUP_FLOOR,
) -> bool:
    """Return True when any parsed shape regresses beyond the noise floor."""
    return any((info.get("speedup") or 0.0) < floor for info in shape_speedups.values())


def _added_lines(patch_text: str) -> list[str]:
    return [line[1:].strip() for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++")]


_BACKUP_FILE_SUFFIXES = (".bak", ".backup", ".orig", ".tmp", "~")


def _patch_touches_backup_file(patch_text: str) -> bool:
    for raw_line in patch_text.splitlines():
        paths: list[str] = []
        if raw_line.startswith("diff --git "):
            parts = raw_line.split()
            paths.extend(parts[2:4])
        elif raw_line.startswith(("+++ ", "--- ")):
            paths.append(raw_line[4:].strip())

        for raw_path in paths:
            path = raw_path
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path == "/dev/null":
                continue
            name = Path(path).name.lower()
            if name.endswith(_BACKUP_FILE_SUFFIXES):
                return True
    return False


def _identifier_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")


def _contains_identifier(text: str, symbol: str) -> bool:
    return bool(_identifier_pattern(symbol).search(text))


def _patch_touches_any_target_symbol(patch_text: str, required_symbols: list[str]) -> bool:
    code_text = "\n".join(_added_lines(patch_text))
    return any(_contains_identifier(code_text, symbol) for symbol in required_symbols)


def _patch_touches_forbidden_target_symbol(patch_text: str, forbidden_symbols: list[str]) -> str | None:
    return patch_touches_forbidden_target_symbol(patch_text, forbidden_symbols)


def _has_added_generic_gluon_dot(patch_text: str) -> bool:
    for line in _added_lines(patch_text):
        if not line or line.startswith("#"):
            continue
        if re.search(r"(?<![A-Za-z0-9_])(?:gl|ttgl)\.dot\s*\(", line):
            return True
    return False


def _has_added_plain_exception_fallback(patch_text: str) -> bool:
    lines = [line for line in _added_lines(patch_text) if line and not line.startswith("#")]
    if not any(re.match(r"except\s+Exception\b", line) for line in lines):
        return False
    launch_pattern = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\[[^\]]+\]\s*\(")
    for line in lines:
        match = launch_pattern.search(line)
        if match and "gluon" not in match.group(1).lower():
            return True
    return False


def _task_requires_matrix_lowering_contract(task_meta: dict[str, Any] | None = None) -> bool:
    task_meta = task_meta or {}
    text = "\n".join(
        str(task_meta.get(key) or "")
        for key in (
            "required_amd_gluon_contract_tags",
            "label",
            "gluon_doc_profile",
            "optimization_direction",
            "performance_hypothesis",
            "allowed_change",
            "gluon_overlay_reason",
        )
    ).lower()
    return any(
        marker in text
        for marker in (
            "matrix",
            "dotoperandlayout",
            "operand layout",
            "mfma",
            "wmma",
            "matrix_lowering",
        )
    )


def _required_amd_gluon_static_contract_error(
    patch_text: str,
    required_output_dialect: str,
    task_meta: dict[str, Any] | None = None,
) -> str | None:
    if str(required_output_dialect or "").strip().lower() != "amd_gluon":
        return None
    if _has_added_plain_exception_fallback(patch_text):
        return "required AMD Gluon patches must not add broad exception fallback to a plain Triton launcher"
    if _task_requires_matrix_lowering_contract(task_meta) and _has_added_generic_gluon_dot(patch_text):
        return "required AMD Gluon patches must not use generic gl.dot as a mechanical dot rewrite"
    return None


def _extract_added_gluon_jit_defs(patch_text: str) -> list[str]:
    defs: list[str] = []
    pending_gluon_jit = False
    for line in _added_lines(patch_text):
        if line.startswith("@gluon.jit"):
            pending_gluon_jit = True
            continue
        match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if match:
            if pending_gluon_jit:
                defs.append(match.group(1))
            pending_gluon_jit = False
            continue
        if line and not line.startswith("@"):
            pending_gluon_jit = False
    return defs


def _patch_calls_symbol(patch_text: str, symbol: str) -> bool:
    call_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])\s*(?:\[|\()")
    for raw_line in patch_text.splitlines():
        if raw_line.startswith("-") and not raw_line.startswith("---"):
            continue
        line = raw_line[1:].strip() if raw_line.startswith(("+", " ")) else raw_line.strip()
        if not line or line.startswith(("#", "@", "def ", "class ", "import ", "from ")):
            continue
        if call_pattern.search(line):
            return True
    return False


def _gluon_execution_contract_satisfied(
    patch_text: str,
    *,
    required_output_dialect: str,
) -> bool:
    if required_output_dialect not in {"amd_gluon", "mixed"}:
        return True
    gluon_defs = _extract_added_gluon_jit_defs(patch_text)
    if not gluon_defs:
        return True
    return any(_patch_calls_symbol(patch_text, helper) for helper in gluon_defs)


def classify_patch_output_dialect(patch_text: str) -> str:
    """Classify whether a patch appears to implement plain Triton, AMD Gluon, or both."""
    return _postprocess_classify_patch_output_dialect(patch_text)


def _find_task_metadata_for_patch_dir(patch_dir: Path) -> dict[str, Any]:
    """Best-effort lookup of the task frontmatter associated with a result dir."""
    label = patch_dir.name
    round_dir = patch_dir.parent.name
    output_root = patch_dir.parent.parent.parent if len(patch_dir.parents) >= 3 else None
    if output_root is None:
        return {}
    tasks_dir = output_root / "tasks" / round_dir
    if not tasks_dir.is_dir():
        return {}
    candidates = list(tasks_dir.glob(f"*_{label}.md")) + list(tasks_dir.glob(f"{label}.md"))
    if not candidates:
        candidates = [p for p in tasks_dir.glob("*.md") if p.stem.endswith(label)]
    if not candidates:
        return {}
    try:
        from minisweagent.run.task_file import read_task_file

        meta, body = read_task_file(candidates[0])
        inferred_targets = _infer_required_patch_target_symbols(body, meta)
        inferred_forbidden_targets = _infer_forbidden_patch_target_symbols(body, meta)
        optional_meta: dict[str, Any] = {}
        for key, tag in _OPTIONAL_GLUON_RESULT_METADATA:
            value = meta.get(key)
            if value in (None, ""):
                value = _parse_tagged_task_value(body, tag)
            normalized = _normalize_optional_result_metadata(value)
            if normalized not in (None, ""):
                optional_meta[key] = normalized
        implementation_layer = _parse_tagged_task_value(body, "Implementation layer")
        extension_layer = _parse_tagged_task_value(body, "Extension layer")
        optimization_direction = _parse_tagged_task_value(body, "Optimization direction")
        performance_hypothesis = _parse_tagged_task_value(body, "Performance hypothesis")
        allowed_change = _parse_tagged_task_value(body, "Allowed change")
        gluon_overlay_reason = _parse_tagged_task_value(body, "Gluon overlay reason")
        if inferred_targets:
            meta = dict(meta)
            meta["required_patch_target_symbols"] = inferred_targets
        if inferred_forbidden_targets:
            meta = dict(meta)
            meta["forbidden_patch_target_symbols"] = inferred_forbidden_targets
        if (
            implementation_layer
            or extension_layer
            or optimization_direction
            or performance_hypothesis
            or allowed_change
            or gluon_overlay_reason
            or optional_meta
        ):
            meta = dict(meta)
            if implementation_layer:
                meta["implementation_layer"] = implementation_layer
            if extension_layer:
                meta["extension_layer"] = extension_layer
            if optimization_direction:
                meta["optimization_direction"] = optimization_direction
            if performance_hypothesis:
                meta["performance_hypothesis"] = performance_hypothesis
            if allowed_change:
                meta["allowed_change"] = allowed_change
            if gluon_overlay_reason:
                meta["gluon_overlay_reason"] = gluon_overlay_reason
            meta.update(optional_meta)
        return meta
    except Exception as exc:
        logger.debug("Could not read task metadata for %s: %s", patch_dir, exc)
        return {}


def _dialect_contract_satisfied(required: str, actual: str) -> bool:
    required = str(required or "any").strip().lower()
    actual = str(actual or "unknown").strip().lower()
    if required in {"", "any"}:
        return True
    if required == "mixed":
        return actual == "mixed"
    if required == "amd_gluon":
        return actual == "amd_gluon"
    if required == "plain_triton":
        return actual in {"plain_triton", "unknown"}
    return actual == required


def _parse_tagged_task_value(task_body: str, tag: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(tag)}\s*:\s*(.+?)\s*$", task_body, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip("`")


def _parse_tagged_task_values(task_body: str, tag: str) -> list[str]:
    return [
        match.group(1).strip().strip("`")
        for match in re.finditer(rf"^\s*{re.escape(tag)}\s*:\s*(.+?)\s*$", task_body, re.IGNORECASE | re.MULTILINE)
    ]


def _metadata_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _infer_required_patch_target_symbols(task_body: str, task_meta: dict[str, Any]) -> list[str]:
    symbols = _metadata_list(task_meta.get("required_patch_target_symbols"))
    for match in re.finditer(
        r"^\s*(?:Target symbol|Target component|Required patch target symbols?)\s*:\s*(.+?)\s*$",
        task_body,
        re.IGNORECASE | re.MULTILINE,
    ):
        value = match.group(1)
        symbols.extend(part.strip().strip("`") for part in value.split(","))
        symbols.extend(target_symbols_from_scoped_text(value))
    for field in ("Allowed change", "Target component"):
        value = _parse_tagged_task_value(task_body, field)
        if value:
            symbols.extend(target_symbols_from_scoped_text(value))

    unique: list[str] = []
    for symbol in symbols:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol) and symbol not in unique:
            unique.append(symbol)
    return unique


def _infer_forbidden_patch_target_symbols(task_body: str, task_meta: dict[str, Any]) -> list[str]:
    symbols = _metadata_list(task_meta.get("forbidden_patch_target_symbols"))
    for key in ("forbidden_change", "forbidden_changes"):
        symbols.extend(forbidden_symbols_from_scoped_text(task_meta.get(key)))
    for field in ("Forbidden change", "Reject if"):
        for value in _parse_tagged_task_values(task_body, field):
            symbols.extend(forbidden_symbols_from_scoped_text(value))
    for value in do_not_clauses(task_body):
        symbols.extend(forbidden_symbols_from_scoped_text(value))

    unique: list[str] = []
    for symbol in symbols:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol) and symbol not in unique:
            unique.append(symbol)
    return unique


_OPTIONAL_GLUON_RESULT_METADATA: tuple[tuple[str, str], ...] = (
    ("source_origin", "Source origin"),
    ("gluon_tl_policy", "Gluon TL policy"),
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
    ("extension_intent", "Extension intent"),
    ("expected_outcome", "Expected outcome"),
    ("not_viable_for_l1_if_slower_than_base", "Not viable for L1 if slower than Base"),
    ("overhead_source_to_record", "Overhead source to record"),
    ("minimum_executable_unit", "Minimum executable unit"),
    ("target_symbol", "Target symbol"),
    ("target_component", "Target component"),
    ("forbidden_change", "Forbidden change"),
    ("allowed_execution_path", "Allowed execution path"),
    ("scope_infeasible_policy", "Scope infeasible policy"),
    ("whole_kernel_required_reason", "Whole kernel required reason"),
    ("scope_infeasible_reported", "Scope infeasible reported"),
)


def _normalize_optional_result_metadata(value: Any) -> Any:
    if isinstance(value, bool):
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


def _required_output_dialect(task_meta: dict[str, Any], *, label: str) -> str:
    required = str(task_meta.get("required_output_dialect") or "").strip().lower()
    implementation_layer = str(task_meta.get("implementation_layer") or "").strip().lower()
    extension_layer = str(task_meta.get("extension_layer") or "").strip().lower()
    layer_required = ""
    label_text = label.lower()
    if "hybrid" in label_text or "mixed" in label_text:
        layer_required = "mixed"
    elif "mixed" in implementation_layer or "hybrid" in implementation_layer or extension_layer == "hybrid":
        layer_required = "mixed"
    elif "amd_gluon" in implementation_layer or extension_layer in {"l0", "l1"}:
        layer_required = "amd_gluon"
    if required:
        if required in {"any", "plain_triton"} and layer_required in {"amd_gluon", "mixed"}:
            return layer_required
        return required
    return layer_required or "any"


def _gluon_l1_anchor_viability(
    *,
    required_output_dialect: str,
    actual_output_dialect: str,
    dialect_contract_satisfied: bool,
    gluon_execution_contract_satisfied: bool,
    speedup: float,
    has_shape_regression: bool,
    task_meta: dict[str, Any],
) -> str:
    if task_meta.get("scope_infeasible_reported") is True:
        return "not_applicable"
    if (
        required_output_dialect not in {"amd_gluon", "mixed"}
        or actual_output_dialect not in {"amd_gluon", "mixed"}
        or not dialect_contract_satisfied
        or not gluon_execution_contract_satisfied
    ):
        return "not_applicable"
    if speedup >= 1.0 and not has_shape_regression:
        return "viable_for_l1"
    extension_intent = str(task_meta.get("extension_intent") or "").strip().lower()
    if extension_intent == "performance_candidate":
        return "neutral_or_slow_anchor"
    if (
        extension_intent == "execution_anchor"
        or task_meta.get("not_viable_for_l1_if_slower_than_base") is True
        or has_shape_regression
    ):
        return "not_viable_for_l1"
    if speedup < 0.5:
        return "not_viable_for_l1"
    return "neutral_or_slow_anchor"


def _find_original_baseline_ms(patch_dir: Path) -> float | None:
    """Walk up from patch_dir to find benchmark_baseline.txt (the canonical baseline).

    The preprocessing phase writes benchmark_baseline.txt at the kernel
    output root (e.g. patches/exp0/rope/benchmark_baseline.txt).  Task dirs
    are nested under results/round_N/strategy_name, so we walk upward.
    """
    d = patch_dir
    for _ in range(8):
        bl = d / "benchmark_baseline.txt"
        if bl.is_file():
            text = bl.read_text()
            lat = _extract_latency(text)
            if lat is not None and lat > 0:
                return lat
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def compute_best_patch(patch_dir: Path) -> dict[str, Any] | None:
    """Deterministically select the best non-empty patch from a task directory.

    Uses ``benchmark_baseline.txt`` as the canonical (unmodified) baseline rather
    than ``patch_0_test.txt`` which is the agent's first attempt.  Only
    returns a result if a patch genuinely beats the true baseline (>1.0x).
    """
    original_bl = _find_original_baseline_ms(patch_dir)

    baseline_file = patch_dir / "patch_0_test.txt"
    baseline_text = ""
    baseline_shape_latencies: dict[str, float] = {}
    if original_bl is not None:
        baseline_ms = original_bl
        baseline_source = "benchmark_baseline.txt"
        baseline_file_path = next(
            (p for p in [patch_dir, *patch_dir.parents] if (p / "benchmark_baseline.txt").is_file()), None
        )
        if baseline_file_path is not None:
            baseline_text = (baseline_file_path / "benchmark_baseline.txt").read_text()
            baseline_shape_latencies = parse_shape_latencies_ms(baseline_text)
    elif baseline_file.exists():
        baseline_text = baseline_file.read_text()
        baseline_ms = _extract_latency(baseline_text)
        baseline_source = "patch_0_test.txt (FALLBACK)"
        baseline_shape_latencies = parse_shape_latencies_ms(baseline_text)
    else:
        return None

    if baseline_ms is None or baseline_ms <= 0:
        return None

    best_speedup = 0.0
    best_candidate_ms: float | None = None
    best_patch_id: str | None = None
    best_patch_file: str | None = None
    best_test_file: str | None = None
    best_patch_size: int = 0
    best_actual_output_dialect = "unknown"
    best_legacy_output_dialect = "unknown"
    best_api_contract: dict[str, Any] = {
        "gluon_api_contract_status": "ok",
        "gluon_tl_policy": "strict_generated",
        "tl_symbols_seen": [],
        "forbidden_tl_symbols_seen": [],
    }
    best_layout_contract: dict[str, Any] = {
        "layout_contract_status": "ok",
        "layout_construction_policy": "host_preferred",
        "layout_symbols_seen": [],
    }
    best_shape_speedups: dict[str, dict[str, float]] = {}
    best_candidate_shape_latencies: dict[str, float] = {}
    best_candidate_shape_geomean: float | None = None
    best_has_shape_regression = False
    baseline_shape_geomean = _geomean_ms(baseline_shape_latencies)
    task_meta = _find_task_metadata_for_patch_dir(patch_dir)
    required_patch_target_symbols = _metadata_list(task_meta.get("required_patch_target_symbols"))
    forbidden_patch_target_symbols = _metadata_list(task_meta.get("forbidden_patch_target_symbols"))
    required_output_dialect = _required_output_dialect(task_meta, label=patch_dir.name)
    preserves_gluon_evidence = required_output_dialect in {"amd_gluon", "mixed"}

    for test_file in sorted(patch_dir.glob("patch_*_test.txt")):
        name = test_file.stem.replace("_test", "")

        patch_file = patch_dir / f"{name}.patch"
        if not patch_file.exists():
            continue
        psz = patch_file.stat().st_size
        if psz == 0:
            continue
        patch_text = patch_file.read_text(errors="replace")
        if _patch_touches_backup_file(patch_text):
            logger.info("Skipping %s because it touches backup or temporary files", name)
            continue
        static_contract_error = _required_amd_gluon_static_contract_error(
            patch_text,
            required_output_dialect,
            task_meta,
        )
        if static_contract_error:
            logger.info("Skipping %s because %s", name, static_contract_error)
            continue
        if required_patch_target_symbols and not _patch_touches_any_target_symbol(
            patch_text,
            required_patch_target_symbols,
        ):
            logger.info(
                "Skipping %s because it does not touch required target symbols: %s",
                name,
                required_patch_target_symbols,
            )
            continue
        forbidden_target = _patch_touches_forbidden_target_symbol(patch_text, forbidden_patch_target_symbols)
        if forbidden_target:
            logger.info("Skipping %s because it touches forbidden target scope: %s", name, forbidden_target)
            continue
        api_contract = classify_gluon_api_contract(patch_text, task_meta)
        if api_contract["gluon_api_contract_status"] in {"leftover_tl_device_api", "invalid_plain_fallback"}:
            logger.info("Skipping %s because Gluon API contract failed", name)
            continue
        layout_contract = classify_layout_contract(patch_text, task_meta)
        if layout_contract["layout_contract_status"] == "runtime_layout_object":
            logger.info("Skipping %s because layout contract failed", name)
            continue

        candidate_text = test_file.read_text()
        candidate_ms = _extract_latency(candidate_text)
        if candidate_ms is None or candidate_ms <= 0:
            continue
        candidate_shape_latencies = parse_shape_latencies_ms(candidate_text)
        actual_output_dialect = classify_patch_output_dialect(patch_text)
        if not _dialect_contract_satisfied(required_output_dialect, actual_output_dialect):
            logger.info(
                "Skipping %s because required_output_dialect=%s but patch classified as %s",
                name,
                required_output_dialect,
                actual_output_dialect,
            )
            continue
        if not _gluon_execution_contract_satisfied(
            patch_text,
            required_output_dialect=required_output_dialect,
        ):
            logger.info("Skipping %s because it defines a Gluon helper without executing it.", name)
            continue
        shape_speedups = compute_shape_speedups(baseline_shape_latencies, candidate_shape_latencies)
        has_shape_regression = _has_significant_shape_regression(shape_speedups) if shape_speedups else False
        if shape_speedups and has_shape_regression and not preserves_gluon_evidence:
            logger.info(
                "Skipping %s due to per-shape regression below %.2fx: %s",
                name,
                _PER_SHAPE_REGRESSION_SPEEDUP_FLOOR,
                shape_speedups,
            )
            continue

        speedup = baseline_ms / candidate_ms
        if speedup > best_speedup:
            best_speedup = speedup
            best_candidate_ms = candidate_ms
            best_patch_id = name
            best_patch_file = str(patch_file)
            best_test_file = str(test_file)
            best_patch_size = psz
            best_actual_output_dialect = actual_output_dialect
            best_legacy_output_dialect = _legacy_classify_patch_output_dialect(patch_text)
            best_api_contract = api_contract
            best_layout_contract = layout_contract
            best_candidate_shape_latencies = candidate_shape_latencies
            best_candidate_shape_geomean = _geomean_ms(candidate_shape_latencies)
            best_shape_speedups = shape_speedups
            best_has_shape_regression = has_shape_regression

    if best_patch_id is None:
        return None
    if best_speedup <= 1.0 and not preserves_gluon_evidence:
        return None

    dialect_ok = _dialect_contract_satisfied(
        required_output_dialect,
        best_actual_output_dialect,
    )
    gluon_execution_ok = required_output_dialect in {"amd_gluon", "mixed"}
    anchor_viability = _gluon_l1_anchor_viability(
        required_output_dialect=required_output_dialect,
        actual_output_dialect=best_actual_output_dialect,
        dialect_contract_satisfied=dialect_ok,
        gluon_execution_contract_satisfied=gluon_execution_ok,
        speedup=best_speedup,
        has_shape_regression=best_has_shape_regression,
        task_meta=task_meta,
    )
    optional_meta = {
        key: task_meta.get(key)
        for key, _tag in _OPTIONAL_GLUON_RESULT_METADATA
        if task_meta.get(key) not in (None, "")
    }
    result = {
        "best_patch_id": best_patch_id,
        "best_patch_speedup": round(best_speedup, 6),
        "best_patch_file": best_patch_file,
        "best_patch_test_output": best_test_file,
        "best_patch_size_bytes": best_patch_size,
        "baseline_latency_ms": round(baseline_ms, 6),
        "candidate_latency_ms": round(best_candidate_ms, 6),
        "baseline_source": baseline_source,
        "baseline_shape_latency_ms": baseline_shape_latencies,
        "candidate_shape_latency_ms": best_candidate_shape_latencies,
        "baseline_shape_geomean_ms": round(baseline_shape_geomean, 6) if baseline_shape_geomean else None,
        "candidate_shape_geomean_ms": (
            round(best_candidate_shape_geomean, 6) if best_candidate_shape_geomean else None
        ),
        "per_shape_speedups": best_shape_speedups,
        "objective": "total_shape_latency_ms" if best_shape_speedups else "latency_ms",
        "improves_true_baseline": best_speedup > 1.0,
        "has_significant_shape_regression": best_has_shape_regression,
        "required_output_dialect": required_output_dialect,
        "actual_output_dialect": best_actual_output_dialect,
        "legacy_output_dialect_classification": best_legacy_output_dialect,
        "dialect_contract_satisfied": dialect_ok,
        "gluon_execution_contract_satisfied": gluon_execution_ok,
        "gluon_api_contract_status": best_api_contract["gluon_api_contract_status"],
        "gluon_tl_policy": best_api_contract["gluon_tl_policy"],
        "tl_symbols_seen": best_api_contract["tl_symbols_seen"],
        "forbidden_tl_symbols_seen": best_api_contract["forbidden_tl_symbols_seen"],
        "layout_contract_status": best_layout_contract["layout_contract_status"],
        "layout_construction_policy": best_layout_contract["layout_construction_policy"],
        "layout_symbols_seen": best_layout_contract["layout_symbols_seen"],
        "contract_schema_version": 2,
        "gluon_l1_anchor_viability": anchor_viability,
        "not_viable_for_l1": anchor_viability == "not_viable_for_l1",
        "overhead_source": task_meta.get("overhead_source_to_record") if anchor_viability != "viable_for_l1" else None,
        "scope_compliant": True,
        "forbidden_scope_violation": None,
        "scope_infeasible_reported": False,
        "required_patch_target_symbols": required_patch_target_symbols,
        "forbidden_patch_target_symbols": forbidden_patch_target_symbols,
        "fallback_used": required_output_dialect == "amd_gluon" and best_actual_output_dialect != "amd_gluon",
        "llm_selection_analysis": (
            f"Deterministic: baseline={baseline_ms:.4f}ms ({baseline_source}), "
            f"candidate={best_candidate_ms:.4f}ms from {best_patch_id}. "
            f"Speedup={best_speedup:.4f}x. Patch={best_patch_size}B."
        ),
    }
    result.update(optional_meta)
    return result


def _invalidate_existing_best_result(
    existing: dict[str, Any],
    *,
    reason: str,
    required_output_dialect: str = "any",
    actual_output_dialect: str = "unknown",
    forbidden_scope_violation: str | None = None,
) -> dict[str, Any]:
    existing["best_patch_id"] = None
    existing["best_patch_file"] = None
    existing["best_patch_test_output"] = existing.get("best_patch_test_output")
    existing["best_patch_speedup"] = 0.0
    existing["required_output_dialect"] = required_output_dialect
    existing["actual_output_dialect"] = actual_output_dialect
    existing["dialect_contract_satisfied"] = False
    existing["fallback_used"] = required_output_dialect == "amd_gluon"
    existing["invalidated"] = True
    existing["invalid_reason"] = reason
    existing["scope_compliant"] = False
    existing["forbidden_scope_violation"] = forbidden_scope_violation
    existing.setdefault("scope_infeasible_reported", False)
    existing["llm_selection_analysis"] = (
        existing.get("llm_selection_analysis") or ""
    ) + f" [Invalidated: {reason}]"
    return existing


def rewrite_best_results(patch_dir: Path) -> dict[str, Any] | None:
    """Overwrite ``best_results.json`` with deterministic selection if possible.

    Uses the canonical baseline from benchmark_baseline.txt.  If no patch
    genuinely improves on the true baseline, clamps any LLM-reported
    speedup to 1.0x to prevent false positives.
    """
    det = compute_best_patch(patch_dir)
    existing_path = patch_dir / "best_results.json"
    original_bl = _find_original_baseline_ms(patch_dir)

    if det is not None:
        existing_path.write_text(json.dumps(det, indent=2))
        logger.info(
            "Deterministic best_results for %s: %s (%.4fx)",
            patch_dir.name,
            det["best_patch_id"],
            det["best_patch_speedup"],
        )
        return det

    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
            pf = existing.get("best_patch_file")
            task_meta = _find_task_metadata_for_patch_dir(patch_dir)
            required_output_dialect = _required_output_dialect(task_meta, label=patch_dir.name)
            required_patch_target_symbols = _metadata_list(task_meta.get("required_patch_target_symbols"))
            forbidden_patch_target_symbols = _metadata_list(task_meta.get("forbidden_patch_target_symbols"))
            for key, _tag in _OPTIONAL_GLUON_RESULT_METADATA:
                if task_meta.get(key) not in (None, ""):
                    existing[key] = task_meta.get(key)
            existing["required_output_dialect"] = required_output_dialect
            existing["required_patch_target_symbols"] = required_patch_target_symbols
            existing["forbidden_patch_target_symbols"] = forbidden_patch_target_symbols
            existing.setdefault("scope_compliant", True)
            existing.setdefault("forbidden_scope_violation", None)
            existing["gluon_execution_contract_satisfied"] = (
                bool(existing.get("gluon_execution_contract_satisfied"))
                if required_output_dialect in {"amd_gluon", "mixed"}
                else False
            )

            if not pf:
                invalid = _invalidate_existing_best_result(
                    existing,
                    reason="best_results has no patch file",
                    required_output_dialect=required_output_dialect,
                )
                existing_path.write_text(json.dumps(invalid, indent=2))
                return invalid

            patch_path = Path(pf)
            if not patch_path.exists():
                invalid = _invalidate_existing_best_result(
                    existing,
                    reason=f"best patch file does not exist: {pf}",
                    required_output_dialect=required_output_dialect,
                )
                existing_path.write_text(json.dumps(invalid, indent=2))
                return invalid

            if patch_path.stat().st_size == 0:
                invalid = _invalidate_existing_best_result(
                    existing,
                    reason="best patch is empty (0 bytes)",
                    required_output_dialect=required_output_dialect,
                    actual_output_dialect="empty",
                )
                existing_path.write_text(json.dumps(invalid, indent=2))
                return invalid

            patch_text = patch_path.read_text(errors="replace")
            actual_output_dialect = classify_patch_output_dialect(patch_text)
            if required_patch_target_symbols and not _patch_touches_any_target_symbol(
                patch_text,
                required_patch_target_symbols,
            ):
                invalid = _invalidate_existing_best_result(
                    existing,
                    reason="selected patch did not touch required target symbols",
                    required_output_dialect=required_output_dialect,
                    actual_output_dialect=actual_output_dialect,
                )
                existing_path.write_text(json.dumps(invalid, indent=2))
                return invalid
            forbidden_target = _patch_touches_forbidden_target_symbol(patch_text, forbidden_patch_target_symbols)
            if forbidden_target:
                invalid = _invalidate_existing_best_result(
                    existing,
                    reason=f"selected patch touched forbidden target scope {forbidden_target}",
                    required_output_dialect=required_output_dialect,
                    actual_output_dialect=actual_output_dialect,
                    forbidden_scope_violation=forbidden_target,
                )
                existing_path.write_text(json.dumps(invalid, indent=2))
                return invalid
            if not _dialect_contract_satisfied(required_output_dialect, actual_output_dialect):
                invalid = _invalidate_existing_best_result(
                    existing,
                    reason=(
                        f"required_output_dialect={required_output_dialect} "
                        f"but patch classified as {actual_output_dialect}"
                    ),
                    required_output_dialect=required_output_dialect,
                    actual_output_dialect=actual_output_dialect,
                )
                existing_path.write_text(json.dumps(invalid, indent=2))
                return invalid
            if not _gluon_execution_contract_satisfied(
                patch_text,
                required_output_dialect=required_output_dialect,
            ):
                invalid = _invalidate_existing_best_result(
                    existing,
                    reason="Gluon helper was defined but not executed",
                    required_output_dialect=required_output_dialect,
                    actual_output_dialect=actual_output_dialect,
                )
                existing_path.write_text(json.dumps(invalid, indent=2))
                return invalid

            if original_bl is not None:
                existing["best_patch_speedup"] = 1.0
                existing["baseline_latency_ms"] = original_bl
                existing["baseline_source"] = "benchmark_baseline.txt"
                existing["llm_selection_analysis"] = (
                    existing.get("llm_selection_analysis") or ""
                ) + f" [Clamped: no patch beat true baseline {original_bl:.4f}ms]"
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            return existing
        except (json.JSONDecodeError, ValueError):
            pass

    return None
