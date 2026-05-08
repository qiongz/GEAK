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
    added = _added_lines(patch_text)
    added_text = "\n".join(added) if added else patch_text
    has_gluon = any(marker in added_text for marker in _AMD_GLUON_PATCH_MARKERS) or bool(
        re.search(r"\bgl\.(?:load|store|program_id|arange|zeros|full)\b", added_text)
    )
    has_plain_triton = any(marker in added_text for marker in _PLAIN_TRITON_PATCH_MARKERS)
    if has_gluon and has_plain_triton:
        return "mixed"
    if has_gluon:
        return "amd_gluon"
    if has_plain_triton:
        return "plain_triton"
    return "unknown"


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
        implementation_layer = _parse_tagged_task_value(body, "Implementation layer")
        extension_layer = _parse_tagged_task_value(body, "Extension layer")
        if implementation_layer or extension_layer:
            meta = dict(meta)
            if implementation_layer:
                meta["implementation_layer"] = implementation_layer
            if extension_layer:
                meta["extension_layer"] = extension_layer
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
    return actual == required


def _parse_tagged_task_value(task_body: str, tag: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(tag)}\s*:\s*(.+?)\s*$", task_body, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip("`")


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
    best_shape_speedups: dict[str, dict[str, float]] = {}
    best_candidate_shape_latencies: dict[str, float] = {}
    best_candidate_shape_geomean: float | None = None
    baseline_shape_geomean = _geomean_ms(baseline_shape_latencies)
    task_meta = _find_task_metadata_for_patch_dir(patch_dir)
    required_output_dialect = _required_output_dialect(task_meta, label=patch_dir.name)

    for test_file in sorted(patch_dir.glob("patch_*_test.txt")):
        name = test_file.stem.replace("_test", "")

        patch_file = patch_dir / f"{name}.patch"
        if not patch_file.exists():
            continue
        psz = patch_file.stat().st_size
        if psz == 0:
            continue
        patch_text = patch_file.read_text(errors="replace")

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
        if shape_speedups and _has_significant_shape_regression(shape_speedups):
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
            best_candidate_shape_latencies = candidate_shape_latencies
            best_candidate_shape_geomean = _geomean_ms(candidate_shape_latencies)
            best_shape_speedups = shape_speedups

    if best_patch_id is None or best_speedup <= 1.0:
        return None

    return {
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
        "required_output_dialect": required_output_dialect,
        "actual_output_dialect": best_actual_output_dialect,
        "dialect_contract_satisfied": _dialect_contract_satisfied(
            required_output_dialect,
            best_actual_output_dialect,
        ),
        "fallback_used": required_output_dialect == "amd_gluon" and best_actual_output_dialect != "amd_gluon",
        "llm_selection_analysis": (
            f"Deterministic: baseline={baseline_ms:.4f}ms ({baseline_source}), "
            f"candidate={best_candidate_ms:.4f}ms from {best_patch_id}. "
            f"Speedup={best_speedup:.4f}x. Patch={best_patch_size}B."
        ),
    }


def _invalidate_existing_best_result(
    existing: dict[str, Any],
    *,
    reason: str,
    required_output_dialect: str = "any",
    actual_output_dialect: str = "unknown",
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
