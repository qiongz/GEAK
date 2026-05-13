"""Deterministic benchmark output parsing and patch selection.

Provides regex-based extraction of latency metrics from harness output
and a ``compute_best_patch()`` function that selects the best non-empty
patch by comparing benchmark numbers -- no LLM involved.

Measurement methodology:
- Uses ``benchmark_baseline.txt`` (the canonical unmodified baseline benchmark)
- Prioritizes ``GEAK_RESULT_LATENCY_MS=<number>`` marker (standardized)
- Falls back to per-case total latency for ``case_id: <ms> ms`` benchmark output
- Reports actual speedup against the true baseline, including regressions (< 1.0)
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
    filter_abstract_target_symbols,
    forbidden_symbols_from_scoped_text,
    patch_touches_forbidden_target_symbol,
    target_symbols_from_scoped_text,
)

logger = logging.getLogger(__name__)
_AMD_GLUON_PATCH_MARKERS = (
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
_CONTRACT_SCHEMA_VERSION = 2
_VALID_SOURCE_ORIGINS = {
    "generated_overlay",
    "existing_amd_gluon_operator",
    "nv_gluon_translation",
    "unknown",
}
_VALID_GLUON_TL_POLICIES = {
    "strict_generated",
    "preserve_existing_allowed",
    "production_source_allowed",
}
_VALID_LAYOUT_POLICIES = {
    "host_preferred",
    "constexpr_in_kernel_allowed",
    "source_preserve",
}
_STRICT_ALLOWED_TL_SYMBOLS = {
    "tl.constexpr",
    "tl.range",
    "tl.float32",
    "tl.float16",
    "tl.bfloat16",
    "tl.int1",
    "tl.int8",
    "tl.int16",
    "tl.int32",
    "tl.int64",
    "tl.uint8",
    "tl.uint16",
    "tl.uint32",
    "tl.uint64",
}
_CONDITIONAL_SOURCE_PRESERVE_TL_SYMBOLS = {
    "tl.program_id",
    "tl.cdiv",
    "tl.where",
}
_DEFAULT_FORBIDDEN_TL_SYMBOLS = {
    "tl.arange",
    "tl.load",
    "tl.store",
    "tl.zeros",
    "tl.full",
    "tl.dot",
}
_TL_SYMBOL_RE = re.compile(r"\btl\.([A-Za-z_][A-Za-z0-9_]*)")
_GLUON_LAYOUT_CONSTRUCTOR_RE = re.compile(
    r"\b(?:gl|ttgl)\.(?:BlockedLayout|SliceLayout|DotOperandLayout|DistributedLinearLayout|"
    r"SwizzledSharedLayout|PaddedSharedLayout|PartitionedSharedLayout)\s*\("
    r"|\b(?:gl|ttgl)\.amd\.(?:AMDMFMALayout|AMDWMMALayout)\s*\("
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


def _parse_shape_latencies_json_marker(output: str) -> dict[str, float]:
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("GEAK_BENCHMARK_RESULTS_MS="):
            continue
        payload = stripped.split("=", 1)[1].strip()
        try:
            raw = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(raw, dict):
            continue
        parsed: dict[str, float] = {}
        for key, value in raw.items():
            try:
                latency = float(value)
            except (TypeError, ValueError):
                parsed = {}
                break
            if latency <= 0:
                parsed = {}
                break
            parsed[str(key)] = latency
        if parsed:
            return parsed
    return {}


def parse_shape_latencies_ms(output: str) -> dict[str, float]:
    """Extract per-shape latencies from harness benchmark output.

    Expected formats:
        ``GEAK_BENCHMARK_RESULTS_MS={"case_a": 0.1, "case_b": 0.2}``
        ``(32,4096): 0.0503 ms``
        ``mla_decode_small: 0.0580 ms``
    """
    marker_latencies = _parse_shape_latencies_json_marker(output)
    if marker_latencies:
        return marker_latencies

    shape_latencies: dict[str, float] = {}
    for m in re.finditer(r"^\s*(\([^)]*\)):\s*([\d.]+(?:e[+-]?\d+)?)\s*ms\s*$", output, re.MULTILINE):
        shape_latencies[m.group(1)] = float(m.group(2))
    for m in re.finditer(
        r"^\s*([A-Za-z][A-Za-z0-9_.\-/]*)\s*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms\s*$",
        output,
        re.MULTILINE,
    ):
        case_id = m.group(1)
        if case_id.lower() in {"median", "geomean", "mean", "total", "latency"}:
            continue
        shape_latencies[case_id] = float(m.group(2))
    return shape_latencies


def _parse_shape_total_latency_ms(output: str) -> float | None:
    shape_latencies = parse_shape_latencies_ms(output)
    if not shape_latencies:
        return None
    total = sum(shape_latencies.values())
    return total if total > 0 else None


def _geomean_ms(shape_latencies: dict[str, float]) -> float | None:
    values = [value for value in shape_latencies.values() if value > 0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def extract_benchmark_config_lines(output: str) -> list[str]:
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
    # Match lines with at least one timing value that includes a unit suffix.
    # Requiring a unit avoids false positives on lines that happen to contain
    # bare decimals (e.g. version numbers, shape descriptions with floats).
    timing_pattern = re.compile(r"\d+\.\d+\s*(?:ms|us|µs|s|x)")
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith(("-", "=", "#", "Status", "Geometric", "GEAK_")):
            continue
        if not timing_pattern.search(line):
            continue
        if any(kw in line.lower() for kw in ("comparing", "running", "warmup", "median", "geomean", "mean")):
            continue
        # Extract config prefix: everything before the first timing value.
        # Handles multiple output formats:
        #   "M=128, N=16  0.0747ms  0.0474ms  1.58x" → "M=128, N=16"
        #   "B=1 H=32 ... 2.11ms 0.10ms 21.37x"      → "B=1 H=32 ..."
        #   "(2, 4, 64): kernel=0.0411 ms | ref=..."   → "(2, 4, 64)"
        config_part = re.split(r"(?<=[=:])\s*\d+\.\d+|\s+\d+\.\d+", line)[0].strip()
        config_part = re.sub(r"[\s:|]+$", "", config_part)
        config_part = re.sub(r"\s*\|\s*\w+$", "", config_part)
        config_part = re.sub(r":\s*\w+=$", "", config_part)
        if config_part:
            configs.append(config_part)
    return sorted(configs)


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


def extract_latency_ms(text: str) -> float | None:
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
    floor: float = 0.95,
) -> bool:
    return any((info.get("speedup") or 0.0) < floor for info in shape_speedups.values())


def _shape_latencies_comparable(
    baseline_shapes_ms: dict[str, float],
    candidate_shapes_ms: dict[str, float],
) -> bool:
    return bool(baseline_shapes_ms and candidate_shapes_ms and set(baseline_shapes_ms) == set(candidate_shapes_ms))


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
            lat = extract_latency_ms(text)
            if lat is not None and lat > 0:
                return lat
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def _find_original_baseline_text(patch_dir: Path) -> str | None:
    d = patch_dir
    for _ in range(8):
        bl = d / "benchmark_baseline.txt"
        if bl.is_file():
            try:
                return bl.read_text()
            except OSError:
                return None
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def _find_task_metadata_for_patch_dir(patch_dir: Path) -> dict[str, Any]:
    label = patch_dir.name
    task_dirs: list[Path] = []
    for root in (patch_dir, *patch_dir.parents):
        tasks_root = root / "tasks"
        if not tasks_root.is_dir():
            continue
        task_dirs.extend(sorted(path for path in tasks_root.glob("round_*") if path.is_dir()))
        if tasks_root.is_dir():
            task_dirs.append(tasks_root)
        break
    if not task_dirs:
        return {}
    candidates: list[Path] = []
    for tasks_dir in task_dirs:
        candidates.extend(list(tasks_dir.glob(f"*_{label}.md")) + list(tasks_dir.glob(f"{label}.md")))
    if not candidates:
        for tasks_dir in task_dirs:
            candidates.extend([p for p in tasks_dir.glob("*.md") if p.stem.endswith(label)])
    if not candidates:
        return {}
    try:
        from minisweagent.run.task_file import read_task_file

        meta, body = read_task_file(candidates[0])
        inferred_targets = _infer_required_patch_target_symbols(body, meta)
        inferred_forbidden_targets = _infer_forbidden_patch_target_symbols(body, meta)
        comparison_target = _parse_tagged_task_value(body, "Comparison target")
        safe_anchor = _parse_tagged_task_value(body, "Safe anchor")
        implementation_layer = _parse_tagged_task_value(body, "Implementation layer")
        extension_layer = _parse_tagged_task_value(body, "Extension layer")
        optimization_direction = _parse_tagged_task_value(body, "Optimization direction")
        performance_hypothesis = _parse_tagged_task_value(body, "Performance hypothesis")
        allowed_change = _parse_tagged_task_value(body, "Allowed change")
        gluon_overlay_reason = _parse_tagged_task_value(body, "Gluon overlay reason")
        optional_meta: dict[str, Any] = {}
        for key, tag in _OPTIONAL_GLUON_RESULT_METADATA:
            value = meta.get(key)
            if value in (None, ""):
                value = _parse_tagged_task_value(body, tag)
            normalized = _normalize_optional_result_metadata(value)
            if normalized not in (None, ""):
                optional_meta[key] = normalized
        if inferred_targets:
            meta = dict(meta)
            meta["required_patch_target_symbols"] = inferred_targets
        if inferred_forbidden_targets:
            meta = dict(meta)
            meta["forbidden_patch_target_symbols"] = inferred_forbidden_targets
        if (
            comparison_target
            or safe_anchor
            or implementation_layer
            or extension_layer
            or optimization_direction
            or performance_hypothesis
            or allowed_change
            or gluon_overlay_reason
            or optional_meta
        ):
            meta = dict(meta)
            if comparison_target:
                meta["comparison_target"] = comparison_target
            if safe_anchor:
                meta["safe_anchor"] = safe_anchor
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


def _metadata_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


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


def _resolve_anchor_text(patch_dir: Path, anchor: str) -> tuple[str, str] | None:
    anchor = str(anchor or "").strip()
    if not anchor:
        return None

    candidates: list[Path] = []
    raw_path = Path(anchor)
    roots = [patch_dir, patch_dir.parent, patch_dir.parent.parent]
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(root / raw_path for root in roots)

    for candidate in list(candidates):
        if candidate.suffix == ".patch":
            candidates.append(candidate.with_name(f"{candidate.stem}_test.txt"))
        elif candidate.suffix == ".txt":
            pass
        elif candidate.name.startswith("patch_"):
            candidates.append(candidate.with_name(f"{candidate.name}_test.txt"))

    for candidate in candidates:
        if candidate.is_dir():
            best_path = candidate / "best_results.json"
            if best_path.is_file():
                try:
                    best = json.loads(best_path.read_text())
                except (json.JSONDecodeError, OSError):
                    best = {}
                for key in ("best_patch_test_output", "best_patch_file"):
                    value = best.get(key)
                    if not value:
                        continue
                    anchor_file = Path(value)
                    if key == "best_patch_file" and anchor_file.suffix == ".patch":
                        anchor_file = anchor_file.with_name(f"{anchor_file.stem}_test.txt")
                    if anchor_file.is_file():
                        return anchor_file.read_text(errors="replace"), str(anchor_file)
                patch_id = best.get("best_patch_id")
                if patch_id:
                    anchor_file = candidate / f"{patch_id}_test.txt"
                    if anchor_file.is_file():
                        return anchor_file.read_text(errors="replace"), str(anchor_file)
            continue
        if candidate.is_file():
            return candidate.read_text(errors="replace"), str(candidate)
    return None


def _infer_required_patch_target_symbols(task_body: str, task_meta: dict[str, Any]) -> list[str]:
    symbols = _metadata_list(task_meta.get("required_patch_target_symbols"))
    for key in ("target_symbol", "target_component"):
        value = task_meta.get(key)
        if isinstance(value, str) and value.strip():
            symbols.extend(part.strip().strip("`") for part in value.split(","))
            symbols.extend(target_symbols_from_scoped_text(value))
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
    return filter_abstract_target_symbols(unique)


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


def _identifier_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")


def _contains_identifier(text: str, symbol: str) -> bool:
    return bool(_identifier_pattern(symbol).search(text))


def _diff_code_lines(patch_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in patch_text.splitlines():
        if raw_line.startswith(("+++", "---", "diff ", "index ", "@@")):
            continue
        if raw_line.startswith("-") or raw_line.startswith(" "):
            continue
        line = raw_line[1:].strip() if raw_line.startswith("+") else raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _patch_touches_any_target_symbol(patch_text: str, required_symbols: list[str]) -> bool:
    code_text = "\n".join(_diff_code_lines(patch_text))
    return any(
        _contains_identifier(code_text, symbol) or _patch_edits_target_body(patch_text, symbol)
        or _patch_replaces_target_triton_jit_with_gluon_jit(patch_text, symbol)
        for symbol in required_symbols
    )


def _patch_touches_forbidden_target_symbol(patch_text: str, forbidden_symbols: list[str]) -> str | None:
    return patch_touches_forbidden_target_symbol(patch_text, forbidden_symbols)


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


def _removed_public_def_names(patch_text: str) -> list[str]:
    names: list[str] = []
    for raw_line in patch_text.splitlines():
        if not raw_line.startswith("-") or raw_line.startswith("---"):
            continue
        line = raw_line[1:].strip()
        match = re.match(r"def\s+([A-Za-z][A-Za-z0-9_]*)\s*\(", line)
        if match and not match.group(1).startswith("_") and match.group(1) not in names:
            names.append(match.group(1))
    return names


def _added_public_def_names(patch_text: str) -> set[str]:
    names: set[str] = set()
    for line in _added_lines(patch_text):
        match = re.match(r"def\s+([A-Za-z][A-Za-z0-9_]*)\s*\(", line)
        if match and not match.group(1).startswith("_"):
            names.add(match.group(1))
    return names


def _has_public_api_removal(patch_text: str) -> list[str]:
    added = _added_public_def_names(patch_text)
    return [name for name in _removed_public_def_names(patch_text) if name not in added]


def _has_unsupported_mfma_result_elem_type(patch_text: str) -> bool:
    if "AMDMFMALayout" not in patch_text:
        return False
    added_text = "\n".join(_added_lines(patch_text))
    if re.search(r"elem_type\s*=\s*gl\.(?:float16|bfloat16|float8|float8e5|float8e4nv)", added_text):
        return True
    assigned_bad_elem_type = re.search(
        r"\belem_type\s*=\s*gl\.(?:float16|bfloat16|float8|float8e5|float8e4nv)\b",
        added_text,
    )
    return bool(assigned_bad_elem_type and re.search(r"elem_type\s*=\s*elem_type\b", added_text))


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
    removed_public_defs = _has_public_api_removal(patch_text)
    if removed_public_defs:
        return (
            "required AMD Gluon patches must preserve public API wrappers; removed or renamed exported "
            + "function(s): "
            + ", ".join(removed_public_defs)
        )
    if _has_unsupported_mfma_result_elem_type(patch_text):
        return (
            "AMDMFMALayout result elem_type must use a verifier-supported accumulator/result type "
            "(for example float32/float64/int32), not fp16/bf16 input operand dtype"
        )
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


def _added_line_calls_symbol(patch_text: str, symbol: str) -> bool:
    call_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])\s*(?:\[|\()")
    for line in _added_lines(patch_text):
        if not line or line.startswith(("#", "@", "def ", "class ", "import ", "from ")):
            continue
        if call_pattern.search(line):
            return True
    return False


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


def _code_line_from_diff(raw_line: str) -> tuple[str, str] | None:
    if raw_line.startswith(("+++", "---", "diff ", "index ", "@@")):
        return None
    prefix = raw_line[0] if raw_line[:1] in {"+", "-", " "} else ""
    if prefix == "-":
        return None
    text = raw_line[1:] if prefix else raw_line
    if not text.strip() or text.lstrip().startswith("#"):
        return None
    return prefix, text.rstrip()


def _indent_width(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))


def _line_defines_symbol(text: str, symbol: str) -> bool:
    return bool(re.match(rf"\s*(?:async\s+)?def\s+{re.escape(symbol)}\s*\(", text))


def _iter_diff_hunks(patch_text: str) -> list[str]:
    parts = re.split(r"(?m)^(@@.*$)", patch_text)
    if len(parts) == 1:
        return [patch_text]
    hunks: list[str] = []
    prefix = parts[0]
    if prefix.strip():
        hunks.append(prefix)
    for idx in range(1, len(parts), 2):
        header = parts[idx]
        body = parts[idx + 1] if idx + 1 < len(parts) else ""
        hunks.append(f"{header}\n{body}")
    return hunks


def _hunk_header_defines_target(header: str, target: str) -> tuple[bool, int]:
    if not header.startswith("@@"):
        return False, 0
    remainder = header[2:].strip()
    suffix = remainder.split("@@", 1)[1].strip() if "@@" in remainder else remainder
    if _line_defines_symbol(suffix, target):
        return True, _indent_width(suffix)
    return False, 0


def _hunk_adds_inside_target_body(chunk: str, target: str, helper: str | None = None) -> bool:
    in_target = False
    target_indent = 0
    for raw_line in chunk.splitlines():
        if raw_line.startswith("@@"):
            in_target, target_indent = _hunk_header_defines_target(raw_line, target)
            continue
        parsed = _code_line_from_diff(raw_line)
        if parsed is None:
            continue
        prefix, text = parsed
        stripped = text.lstrip()
        indent = _indent_width(text)
        if _line_defines_symbol(text, target):
            in_target = True
            target_indent = indent
            if prefix == "+" and helper is None:
                return True
            continue
        if in_target and stripped and indent <= target_indent and not stripped.startswith(("@", ")", ",")):
            in_target = False
        if not in_target or prefix != "+":
            continue
        if helper is None:
            return True
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(helper)}(?![A-Za-z0-9_])\s*(?:\[|\()", text):
            return True
    return False


def _patch_edits_target_body(patch_text: str, target: str) -> bool:
    return any(_hunk_adds_inside_target_body(chunk, target) for chunk in _iter_diff_hunks(patch_text))


def _added_line_assigns_symbol(patch_text: str, left: str, right: str) -> bool:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(left)}(?![A-Za-z0-9_])\s*=\s*"
        rf"(?<![A-Za-z0-9_]){re.escape(right)}(?![A-Za-z0-9_])"
    )
    return any(pattern.search(line) for line in _added_lines(patch_text))


def _helper_name_matches_target(helper: str, target: str) -> bool:
    helper_l = helper.lower()
    target_l = target.lower()
    return (
        helper_l == target_l
        or helper_l.startswith(f"{target_l}_")
        or helper_l.endswith(f"_{target_l}")
        or helper_l == f"{target_l}gluon"
        or helper_l == f"{target_l}_gluon"
    )


def _hunk_binds_helper_to_target(patch_text: str, helper: str, target: str) -> bool:
    for chunk in _iter_diff_hunks(patch_text):
        if _hunk_adds_inside_target_body(chunk, target, helper):
            return True
    return False


def _helper_associated_with_required_symbols(
    patch_text: str,
    helper: str,
    required_symbols: list[str],
) -> bool:
    if not required_symbols:
        return True
    for symbol in required_symbols:
        if _helper_name_matches_target(helper, symbol):
            return True
        if _added_line_assigns_symbol(patch_text, symbol, helper) or _added_line_assigns_symbol(patch_text, helper, symbol):
            return True
        if _hunk_binds_helper_to_target(patch_text, helper, symbol):
            return True
    return False


def _has_added_gluon_launch_evidence(patch_text: str) -> bool:
    launch_pattern = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_]*gluon[A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*_gluon)\s*\[")
    return any(launch_pattern.search(line) for line in _added_lines(patch_text))


def _target_related_launch_pattern(target: str) -> re.Pattern[str]:
    escaped = re.escape(target)
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{escaped}(?:_?gluon|_[A-Za-z0-9_]*gluon[A-Za-z0-9_]*)|"
        rf"[A-Za-z_][A-Za-z0-9_]*_{escaped}_?gluon[A-Za-z0-9_]*)\s*\[",
        re.IGNORECASE,
    )


def _hunk_adds_target_related_gluon_launch(chunk: str, target: str) -> bool:
    launch_pattern = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_]*gluon[A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*_gluon)\s*\[")
    target_launch_pattern = _target_related_launch_pattern(target)
    in_target = False
    target_indent = 0
    for raw_line in chunk.splitlines():
        if raw_line.startswith("@@"):
            in_target, target_indent = _hunk_header_defines_target(raw_line, target)
            continue
        parsed = _code_line_from_diff(raw_line)
        if parsed is None:
            continue
        prefix, text = parsed
        stripped = text.lstrip()
        indent = _indent_width(text)
        if _line_defines_symbol(text, target):
            in_target = True
            target_indent = indent
            continue
        if in_target and stripped and indent <= target_indent and not stripped.startswith(("@", ")", ",")):
            in_target = False
        if prefix != "+":
            continue
        if target_launch_pattern.search(text):
            return True
        if in_target and launch_pattern.search(text):
            return True
    return False


def _has_target_related_gluon_launch_evidence(patch_text: str, required_symbols: list[str]) -> bool:
    if not required_symbols:
        return _has_added_gluon_launch_evidence(patch_text)
    chunks = _iter_diff_hunks(patch_text)
    for symbol in required_symbols:
        for chunk in chunks:
            if _hunk_adds_target_related_gluon_launch(chunk, symbol):
                return True
    return False


def _edits_existing_gluon_jit_context(patch_text: str) -> bool:
    return any(line.startswith(" ") and "@gluon.jit" in line for line in patch_text.splitlines())


def _hunk_replaces_target_triton_jit_with_gluon_jit(chunk: str, target: str) -> bool:
    removed_triton_jit = False
    added_gluon_jit = False
    target_def_seen = False
    for raw_line in chunk.splitlines():
        if raw_line.startswith("---"):
            continue
        if raw_line.startswith("-") and "@triton.jit" in raw_line:
            removed_triton_jit = True
            continue
        if raw_line.startswith("+") and "@gluon.jit" in raw_line:
            added_gluon_jit = True
            continue
        text = raw_line[1:] if raw_line[:1] in {"+", "-", " "} else raw_line
        if _line_defines_symbol(text, target):
            target_def_seen = True
    return removed_triton_jit and added_gluon_jit and target_def_seen


def _patch_replaces_target_triton_jit_with_gluon_jit(patch_text: str, target: str) -> bool:
    return any(_hunk_replaces_target_triton_jit_with_gluon_jit(chunk, target) for chunk in _iter_diff_hunks(patch_text))


def _patch_has_same_name_launch_evidence(patch_text: str, target: str) -> bool:
    launch_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(target)}(?![A-Za-z0-9_])\s*\[")
    for raw_line in patch_text.splitlines():
        if raw_line.startswith("-") and not raw_line.startswith("---"):
            continue
        line = raw_line[1:].strip() if raw_line.startswith(("+", " ")) else raw_line.strip()
        if not line or line.startswith(("#", "@", "def ", "class ", "import ", "from ")):
            continue
        if launch_pattern.search(line):
            return True
    return False


def _has_same_name_gluon_replacement_execution_evidence(patch_text: str, required_symbols: list[str]) -> bool:
    for symbol in required_symbols:
        if _patch_replaces_target_triton_jit_with_gluon_jit(
            patch_text,
            symbol,
        ) and _patch_has_same_name_launch_evidence(patch_text, symbol):
            return True
    return False


def _has_added_gluon_marker(patch_text: str) -> bool:
    added_text = "\n".join(_added_lines(patch_text))
    return any(marker in added_text for marker in _AMD_GLUON_PATCH_MARKERS) or bool(
        re.search(r"\bgl\.(?:load|store|program_id|arange|zeros|full|cdiv|maximum|minimum|exp)\b", added_text)
    )


def _gluon_execution_contract_satisfied(
    patch_text: str,
    *,
    required_symbols: list[str],
    required_output_dialect: str,
) -> bool:
    """Reject helper-only Gluon patches for stage-specific required targets.

    A patch that only defines ``target_gluon`` while the host still dispatches the
    plain target path should not satisfy a stage-specific AMD Gluon task.
    """
    if required_output_dialect not in {"amd_gluon", "mixed"}:
        return True
    gluon_defs = _extract_added_gluon_jit_defs(patch_text)
    if not gluon_defs:
        if _has_same_name_gluon_replacement_execution_evidence(patch_text, required_symbols):
            return True
        if _has_target_related_gluon_launch_evidence(patch_text, required_symbols):
            return True
        if not required_symbols and _edits_existing_gluon_jit_context(patch_text):
            return True
        return not _has_added_gluon_marker(patch_text)
    for helper in gluon_defs:
        if not _helper_associated_with_required_symbols(patch_text, helper, required_symbols):
            continue
        if _patch_calls_symbol(patch_text, helper) or _added_line_calls_symbol(patch_text, helper):
            return True
    return False


def _scope_escalation_violation(patch_text: str, task_meta: dict[str, Any] | None = None) -> str | None:
    """Detect patches that widen an inline-scoped L0 task to a whole-kernel path."""
    task_meta = task_meta or {}
    allowed_path = str(task_meta.get("allowed_execution_path") or "").strip().lower()
    minimum_unit = str(task_meta.get("minimum_executable_unit") or "").strip().lower()
    if allowed_path != "inline_scoped_helper" and minimum_unit != "inline_scoped_helper":
        return None

    target_symbol = str(task_meta.get("target_symbol") or "").strip()
    gluon_defs = _extract_added_gluon_jit_defs(patch_text)
    if target_symbol:
        for helper in gluon_defs:
            if helper == target_symbol or helper.startswith(f"{target_symbol}_"):
                return f"scope escalation: inline_scoped_helper -> whole_jit_kernel via `{helper}`"
        launch_re = re.compile(rf"\b{re.escape(target_symbol)}[A-Za-z0-9_]*_gluon\s*\[")
        if any(line.startswith("+") and launch_re.search(line) for line in patch_text.splitlines()):
            return f"scope escalation: inline_scoped_helper rerouted `{target_symbol}` to a whole Gluon kernel"

    added = "\n".join(_added_lines(patch_text)).lower()
    if "whole-kernel @gluon.jit" in added or "whole kernel @gluon.jit" in added:
        return "scope escalation: inline_scoped_helper patch declares a whole-kernel Gluon conversion"
    return None


def _gluon_evidence_summary(
    *,
    required_output_dialect: str,
    actual_output_dialect: str = "unknown",
    gluon_execution_contract_satisfied: bool = False,
    speedup: float = 0.0,
    has_shape_regression: bool = False,
    per_shape_speedups: dict[str, Any] | None = None,
    scope_escalation: str | None = None,
) -> str:
    required = str(required_output_dialect or "any").strip().lower()
    if required not in {"amd_gluon", "mixed"}:
        return "not_applicable"
    if scope_escalation:
        return "scope_escalation"
    if actual_output_dialect not in {"amd_gluon", "mixed"} or not gluon_execution_contract_satisfied:
        return "compile_failed_or_not_executed"
    shape_values = [
        float(info.get("speedup") or 0.0)
        for info in (per_shape_speedups or {}).values()
        if isinstance(info, dict)
    ]
    if has_shape_regression:
        return "executed_slower"
    if speedup >= 1.0:
        return "executed_win"
    if any(value > 1.0 for value in shape_values):
        return "executed_win_by_shape"
    return "executed_slower"


def _metadata_value(task_meta: dict[str, Any] | None, key: str, default: str = "") -> str:
    if not task_meta:
        return default
    value = task_meta.get(key)
    if value in (None, ""):
        return default
    return str(value).strip()


def _normalize_source_origin(task_meta: dict[str, Any] | None = None) -> str:
    value = _metadata_value(task_meta, "source_origin", "unknown").strip().lower()
    return value if value in _VALID_SOURCE_ORIGINS else "unknown"


def _normalize_gluon_tl_policy(task_meta: dict[str, Any] | None = None) -> str:
    explicit = _metadata_value(task_meta, "gluon_tl_policy", "").strip().lower()
    if explicit in _VALID_GLUON_TL_POLICIES:
        return explicit
    source_origin = _normalize_source_origin(task_meta)
    if source_origin == "existing_amd_gluon_operator":
        return "production_source_allowed"
    if source_origin == "nv_gluon_translation":
        return "preserve_existing_allowed"
    return "strict_generated"


def _normalize_layout_policy(task_meta: dict[str, Any] | None = None) -> str:
    explicit = _metadata_value(task_meta, "layout_construction_policy", "").strip().lower()
    if explicit in _VALID_LAYOUT_POLICIES:
        return explicit
    source_origin = _normalize_source_origin(task_meta)
    if source_origin == "existing_amd_gluon_operator":
        return "source_preserve"
    return "host_preferred"


def _metadata_symbol_set(task_meta: dict[str, Any] | None, key: str, defaults: set[str]) -> set[str]:
    if not task_meta or key not in task_meta:
        return set(defaults)
    raw = task_meta.get(key)
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        values = []
    return {value if value.startswith("tl.") else f"tl.{value}" for value in values} or set(defaults)


def _diff_code_entries(patch_text: str) -> list[tuple[str, str]]:
    """Return (change_kind, code_line) for non-header, non-comment diff lines."""
    entries: list[tuple[str, str]] = []
    has_diff_prefix = any(line.startswith(("+", "-", " ")) for line in patch_text.splitlines())
    for raw_line in patch_text.splitlines():
        if raw_line.startswith(("+++", "---", "diff ", "index ", "@@")):
            continue
        prefix = raw_line[:1] if raw_line[:1] in {"+", "-", " "} else ""
        if prefix == "-":
            continue
        text = raw_line[1:] if prefix else raw_line
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if prefix == "+":
            change_kind = "added"
        elif prefix == " ":
            change_kind = "preserved"
        else:
            change_kind = "added" if not has_diff_prefix else "preserved"
        entries.append((change_kind, text.rstrip()))
    return entries


def _gluon_jit_code_entries(patch_text: str) -> list[tuple[str, str]]:
    entries = _diff_code_entries(patch_text)
    result: list[tuple[str, str]] = []
    pending_gluon_jit = False
    in_gluon = False
    body_indent = 0
    for change_kind, text in entries:
        stripped = text.lstrip()
        indent = _indent_width(text)
        if stripped.startswith("@gluon.jit"):
            pending_gluon_jit = True
            in_gluon = False
            continue
        if re.match(r"(?:async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped):
            if pending_gluon_jit:
                in_gluon = True
                body_indent = indent
                pending_gluon_jit = False
            else:
                in_gluon = False
            continue
        if pending_gluon_jit and stripped and not stripped.startswith("@"):
            pending_gluon_jit = False
        if in_gluon and stripped and indent <= body_indent and not stripped.startswith(("@", ")", ",")):
            in_gluon = False
        if in_gluon:
            result.append((change_kind, text))
    return result


def classify_gluon_api_contract(
    patch_text: str,
    task_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify tl.* usage inside @gluon.jit independently from output dialect."""
    policy = _normalize_gluon_tl_policy(task_meta)
    source_origin = _normalize_source_origin(task_meta)
    allowed_symbols = _metadata_symbol_set(task_meta, "allowed_tl_symbols", _STRICT_ALLOWED_TL_SYMBOLS)
    forbidden_symbols = _metadata_symbol_set(task_meta, "forbidden_tl_symbols", _DEFAULT_FORBIDDEN_TL_SYMBOLS)
    seen: list[dict[str, str]] = []
    forbidden_seen: list[dict[str, str]] = []
    for change_kind, line in _gluon_jit_code_entries(patch_text):
        for match in _TL_SYMBOL_RE.finditer(line):
            symbol = f"tl.{match.group(1)}"
            entry = {
                "symbol": symbol,
                "location": "inside_gluon_jit",
                "change_kind": change_kind,
            }
            seen.append(entry)
            if symbol in forbidden_symbols:
                forbidden_seen.append(entry)
                continue
            if symbol in allowed_symbols:
                continue
            if symbol in _CONDITIONAL_SOURCE_PRESERVE_TL_SYMBOLS:
                if policy in {"production_source_allowed", "preserve_existing_allowed"} and change_kind != "added":
                    continue
                forbidden_seen.append(entry)
                continue
            if change_kind == "added" or policy == "strict_generated":
                forbidden_seen.append(entry)
    if forbidden_seen:
        status = "leftover_tl_device_api"
    elif seen:
        status = "allowed_tl_only"
    else:
        status = "ok"
    return {
        "gluon_api_contract_status": status,
        "tl_symbols_seen": seen,
        "forbidden_tl_symbols_seen": forbidden_seen,
        "gluon_tl_policy": policy,
        "source_origin": source_origin,
        "contract_schema_version": _CONTRACT_SCHEMA_VERSION,
    }


def classify_layout_contract(
    patch_text: str,
    task_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = _normalize_layout_policy(task_meta)
    runtime_layout_entries: list[dict[str, str]] = []
    layout_entries: list[dict[str, str]] = []
    for change_kind, line in _gluon_jit_code_entries(patch_text):
        if not _GLUON_LAYOUT_CONSTRUCTOR_RE.search(line):
            continue
        entry = {
            "line": line.strip(),
            "location": "inside_gluon_jit",
            "change_kind": change_kind,
        }
        layout_entries.append(entry)
        if "constexpr" not in line:
            runtime_layout_entries.append(entry)
    if runtime_layout_entries:
        status = "runtime_layout_object"
    elif policy == "host_preferred" and any(entry["change_kind"] == "added" for entry in layout_entries):
        status = "missing_layout_contract"
    else:
        status = "ok"
    return {
        "layout_contract_status": status,
        "layout_construction_policy": policy,
        "layout_symbols_seen": layout_entries,
        "contract_schema_version": _CONTRACT_SCHEMA_VERSION,
    }


def _has_host_side_mixed_dispatch(patch_text: str) -> bool:
    added_text = "\n".join(_added_lines(patch_text)) if _added_lines(patch_text) else patch_text
    lowered = added_text.lower()
    if not ("gluon" in lowered and ("triton" in lowered or "plain" in lowered)):
        return False
    has_gluon_launch = bool(
        re.search(r"\b(?:gluon[A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*_?gluon[A-Za-z0-9_]*)\s*\[", added_text)
    )
    has_plain_launch = bool(
        re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(?:triton|plain)[A-Za-z0-9_]*\s*\[", added_text)
        or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*_kernel\s*\[", added_text)
    )
    dispatch_condition = any(
        marker in lowered
        for marker in (
            "if ",
            "elif ",
            "else:",
            "dispatch",
            "use_gluon",
            "use_triton",
            "fallback",
            "shape",
            "bucket",
        )
    )
    return has_gluon_launch and has_plain_launch and dispatch_condition


def _legacy_classify_patch_output_dialect(patch_text: str) -> str:
    added = _added_lines(patch_text)
    added_text = "\n".join(added) if added else patch_text
    has_gluon = any(marker in added_text for marker in _AMD_GLUON_PATCH_MARKERS) or bool(
        re.search(r"\bgl\.(?:load|store|program_id|arange|zeros|full|cdiv|maximum|minimum|exp)\b", added_text)
    )
    has_plain_triton = any(marker in added_text for marker in _PLAIN_TRITON_PATCH_MARKERS)
    if has_gluon and has_plain_triton:
        return "mixed"
    if has_gluon:
        return "amd_gluon"
    if has_plain_triton:
        return "plain_triton"
    return "unknown"


def classify_patch_output_dialect(patch_text: str) -> str:
    added = _added_lines(patch_text)
    added_text = "\n".join(added) if added else patch_text
    has_gluon = any(marker in added_text for marker in _AMD_GLUON_PATCH_MARKERS) or bool(
        re.search(r"\bgl\.(?:load|store|program_id|arange|zeros|full|cdiv|maximum|minimum|exp)\b", added_text)
    )
    has_plain_triton = any(marker in added_text for marker in _PLAIN_TRITON_PATCH_MARKERS)
    if has_gluon and has_plain_triton and _has_host_side_mixed_dispatch(patch_text):
        return "mixed"
    if has_gluon:
        return "amd_gluon"
    if has_plain_triton:
        return "plain_triton"
    return "unknown"


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


def _layer_required_output_dialect(task_meta: dict[str, Any], *, label: str) -> str:
    label_text = label.lower()
    implementation_layer = str(task_meta.get("implementation_layer") or "").strip().lower()
    extension_layer = str(task_meta.get("extension_layer") or "").strip().lower()
    if "hybrid" in label_text or "mixed" in label_text:
        return "mixed"
    if "mixed" in implementation_layer or "hybrid" in implementation_layer or extension_layer == "hybrid":
        return "mixed"
    if "amd_gluon" in implementation_layer or extension_layer in {"l0", "l1"}:
        return "amd_gluon"
    if (
        label_text.startswith(("ext-", "extension-"))
        or "extension-l" in label_text
        or "ext_l" in label_text
    ) and ("gluon" in label_text or "amd-gluon" in label_text or "amd_gluon" in label_text):
        return "amd_gluon"
    return ""


def _required_output_dialect(task_meta: dict[str, Any], *, label: str) -> str:
    required = str(task_meta.get("required_output_dialect") or "").strip().lower()
    layer_required = _layer_required_output_dialect(task_meta, label=label)
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
    block_l1 = task_meta.get("not_viable_for_l1_if_slower_than_base")
    if extension_intent == "performance_candidate" and speedup < 1.0:
        return "neutral_or_slow_anchor"
    if extension_intent == "execution_anchor" and (block_l1 is True or speedup < 1.0 or has_shape_regression):
        return "not_viable_for_l1"
    if speedup < 0.5 or has_shape_regression:
        return "not_viable_for_l1"
    return "neutral_or_slow_anchor"


_COARSE_OR_MISSING_OVERHEAD_SOURCES = {"", "unknown", "tiny_stage_overhead"}


def _overhead_attribution_incomplete(anchor_viability: str, overhead_source: Any) -> tuple[bool, str | None]:
    if anchor_viability != "not_viable_for_l1":
        return False, None
    normalized = str(overhead_source or "").strip().lower()
    if normalized not in _COARSE_OR_MISSING_OVERHEAD_SOURCES:
        return False, None
    return (
        True,
        "not_viable_for_l1 needs a concrete removable-overhead attribution before treating the L0 anchor as exhausted",
    )


def compute_best_patch(patch_dir: Path) -> dict[str, Any] | None:
    """Deterministically select the best non-empty patch from a task directory.

    Uses ``benchmark_baseline.txt`` as the canonical (unmodified) baseline rather
    than ``patch_0_test.txt`` which is the agent's first attempt.
    """
    original_bl = _find_original_baseline_ms(patch_dir)

    baseline_text = _find_original_baseline_text(patch_dir) or ""
    baseline_shape_latencies: dict[str, float] = {}
    if original_bl is None or not baseline_text:
        logger.debug("compute_best_patch(%s): no canonical benchmark_baseline.txt found; returning None.", patch_dir.name)
        return None
    baseline_ms = original_bl
    baseline_source = "benchmark_baseline.txt"
    logger.debug("compute_best_patch(%s): using benchmark_baseline.txt (%.4f ms).", patch_dir.name, original_bl)
    baseline_shape_latencies = parse_shape_latencies_ms(baseline_text)

    if baseline_ms is None or baseline_ms <= 0:
        logger.debug("compute_best_patch(%s): invalid baseline_ms=%s; returning None.", patch_dir.name, baseline_ms)
        return None

    best_speedup = 0.0
    best_candidate_ms: float | None = None
    best_patch_id: str | None = None
    best_patch_file: str | None = None
    best_test_file: str | None = None
    best_patch_size: int = 0
    best_shape_speedups: dict[str, dict[str, float]] = {}
    best_candidate_shape_latencies: dict[str, float] = {}
    best_candidate_shape_geomean: float | None = None
    best_has_shape_regression = False
    best_shape_evidence_missing = False
    best_baseline_comparison_ms = baseline_ms
    best_actual_output_dialect = "unknown"
    best_legacy_output_dialect = "unknown"
    best_api_contract: dict[str, Any] = {
        "gluon_api_contract_status": "ok",
        "gluon_tl_policy": _normalize_gluon_tl_policy({}),
        "tl_symbols_seen": [],
        "forbidden_tl_symbols_seen": [],
    }
    best_layout_contract: dict[str, Any] = {
        "layout_contract_status": "ok",
        "layout_construction_policy": _normalize_layout_policy({}),
        "layout_symbols_seen": [],
    }
    baseline_shape_geomean = _geomean_ms(baseline_shape_latencies)
    task_meta = _find_task_metadata_for_patch_dir(patch_dir)
    required_patch_target_symbols = _metadata_list(task_meta.get("required_patch_target_symbols"))
    forbidden_patch_target_symbols = _metadata_list(task_meta.get("forbidden_patch_target_symbols"))
    required_output_dialect = _required_output_dialect(task_meta, label=patch_dir.name)
    legacy_output_dialect_classification = "unknown"
    true_baseline_ms = baseline_ms
    true_baseline_shape_latencies = dict(baseline_shape_latencies)
    comparison_target = str(task_meta.get("comparison_target") or "true_baseline").strip().lower()
    safe_anchor_ref = str(task_meta.get("safe_anchor") or "").strip()
    safe_anchor_source: str | None = None
    safe_anchor_ms: float | None = None
    if comparison_target == "safe_anchor":
        resolved_anchor = _resolve_anchor_text(patch_dir, safe_anchor_ref)
        if resolved_anchor is None:
            logger.debug("compute_best_patch(%s): Comparison target is safe_anchor but anchor was not found: %s", patch_dir.name, safe_anchor_ref)
            return None
        anchor_text, safe_anchor_source = resolved_anchor
        safe_anchor_ms = extract_latency_ms(anchor_text)
        if safe_anchor_ms is None or safe_anchor_ms <= 0:
            logger.debug("compute_best_patch(%s): safe_anchor has no valid latency: %s", patch_dir.name, safe_anchor_source)
            return None
        baseline_ms = safe_anchor_ms
        baseline_source = f"safe_anchor:{safe_anchor_ref or safe_anchor_source}"
        baseline_shape_latencies = parse_shape_latencies_ms(anchor_text)
        baseline_shape_geomean = _geomean_ms(baseline_shape_latencies)

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
        scope_escalation = _scope_escalation_violation(patch_text, task_meta)
        if scope_escalation:
            logger.info("Skipping %s because scope contract failed: %s", name, scope_escalation)
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
            logger.info(
                "Skipping %s because Gluon API contract failed: %s",
                name,
                api_contract["forbidden_tl_symbols_seen"],
            )
            continue
        layout_contract = classify_layout_contract(patch_text, task_meta)
        if layout_contract["layout_contract_status"] == "runtime_layout_object":
            logger.info("Skipping %s because layout contract failed: runtime layout object", name)
            continue
        actual_output_dialect = classify_patch_output_dialect(patch_text)
        legacy_output_dialect_classification = _legacy_classify_patch_output_dialect(patch_text)
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
            required_symbols=required_patch_target_symbols,
            required_output_dialect=required_output_dialect,
        ):
            logger.info(
                "Skipping %s because it defines a Gluon helper without executing it for target symbols: %s",
                name,
                required_patch_target_symbols,
            )
            continue

        candidate_text = test_file.read_text()
        candidate_ms = extract_latency_ms(candidate_text)
        if candidate_ms is None or candidate_ms <= 0:
            continue
        candidate_shape_latencies = parse_shape_latencies_ms(candidate_text)
        shape_speedups = compute_shape_speedups(baseline_shape_latencies, candidate_shape_latencies)
        has_shape_regression = _has_significant_shape_regression(shape_speedups) if shape_speedups else False
        shape_evidence_missing = bool(len(baseline_shape_latencies) > 1 and not candidate_shape_latencies)
        baseline_comparison_ms = baseline_ms
        candidate_comparison_ms = candidate_ms
        if _shape_latencies_comparable(baseline_shape_latencies, candidate_shape_latencies):
            baseline_comparison_ms = sum(baseline_shape_latencies.values())
            candidate_comparison_ms = sum(candidate_shape_latencies.values())
        elif shape_evidence_missing:
            has_shape_regression = True
        if comparison_target == "safe_anchor" and has_shape_regression:
            logger.info(
                "Skipping %s because it regresses at least one shape versus safe anchor: %s",
                name,
                shape_speedups,
            )
            continue

        speedup = baseline_comparison_ms / candidate_comparison_ms
        if speedup > best_speedup:
            best_speedup = speedup
            best_candidate_ms = candidate_comparison_ms
            best_baseline_comparison_ms = baseline_comparison_ms
            best_patch_id = name
            best_patch_file = str(patch_file)
            best_test_file = str(test_file)
            best_patch_size = psz
            best_candidate_shape_latencies = candidate_shape_latencies
            best_candidate_shape_geomean = _geomean_ms(candidate_shape_latencies)
            best_shape_speedups = shape_speedups
            best_has_shape_regression = has_shape_regression
            best_shape_evidence_missing = shape_evidence_missing
            best_actual_output_dialect = actual_output_dialect
            best_api_contract = api_contract
            best_layout_contract = layout_contract
            best_legacy_output_dialect = legacy_output_dialect_classification

    if best_patch_id is None or (comparison_target == "safe_anchor" and best_speedup <= 1.0):
        logger.debug(
            "compute_best_patch(%s): no eligible patch found.",
            patch_dir.name,
        )
        return None
    dialect_ok = _dialect_contract_satisfied(required_output_dialect, best_actual_output_dialect)
    gluon_execution_ok = required_output_dialect in {"amd_gluon", "mixed"}
    anchor_shape_regression = best_has_shape_regression or best_shape_evidence_missing
    anchor_viability = _gluon_l1_anchor_viability(
        required_output_dialect=required_output_dialect,
        actual_output_dialect=best_actual_output_dialect,
        dialect_contract_satisfied=dialect_ok,
        gluon_execution_contract_satisfied=gluon_execution_ok,
        speedup=best_speedup,
        has_shape_regression=anchor_shape_regression,
        task_meta=task_meta,
    )
    optional_meta = {
        key: task_meta.get(key)
        for key, _tag in _OPTIONAL_GLUON_RESULT_METADATA
        if task_meta.get(key) not in (None, "")
    }
    overhead_source = task_meta.get("overhead_source_to_record") if anchor_viability != "viable_for_l1" else None
    overhead_attribution_incomplete, overhead_attribution_warning = _overhead_attribution_incomplete(
        anchor_viability,
        overhead_source,
    )

    result = {
        "best_patch_id": best_patch_id,
        "best_patch_speedup": round(best_speedup, 6),
        "best_patch_file": best_patch_file,
        "best_patch_test_output": best_test_file,
        "best_patch_size_bytes": best_patch_size,
        "baseline_latency_ms": round(best_baseline_comparison_ms, 6),
        "candidate_latency_ms": round(best_candidate_ms, 6),
        "baseline_source": baseline_source,
        "comparison_target": comparison_target,
        "safe_anchor": safe_anchor_ref or None,
        "safe_anchor_source": safe_anchor_source,
        "safe_anchor_latency_ms": round(safe_anchor_ms, 6) if safe_anchor_ms else None,
        "true_baseline_latency_ms": round(true_baseline_ms, 6),
        "true_baseline_shape_latency_ms": true_baseline_shape_latencies,
        "baseline_shape_latency_ms": baseline_shape_latencies,
        "candidate_shape_latency_ms": best_candidate_shape_latencies,
        "baseline_shape_geomean_ms": round(baseline_shape_geomean, 6) if baseline_shape_geomean else None,
        "candidate_shape_geomean_ms": (
            round(best_candidate_shape_geomean, 6) if best_candidate_shape_geomean else None
        ),
        "per_shape_speedups": best_shape_speedups,
        "objective": "total_shape_latency_ms" if best_shape_speedups else "latency_ms",
        "improves_true_baseline": (
            (sum(true_baseline_shape_latencies.values()) / sum(best_candidate_shape_latencies.values())) > 1.0
            if _shape_latencies_comparable(true_baseline_shape_latencies, best_candidate_shape_latencies)
            else (true_baseline_ms / best_candidate_ms) > 1.0 if best_candidate_ms else False
        ),
        "has_significant_shape_regression": anchor_shape_regression,
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
        "contract_schema_version": _CONTRACT_SCHEMA_VERSION,
        "gluon_l1_anchor_viability": anchor_viability,
        "not_viable_for_l1": anchor_viability == "not_viable_for_l1",
        "overhead_source": overhead_source,
        "overhead_attribution_incomplete": overhead_attribution_incomplete,
        "overhead_attribution_warning": overhead_attribution_warning,
        "gluon_evidence_summary": _gluon_evidence_summary(
            required_output_dialect=required_output_dialect,
            actual_output_dialect=best_actual_output_dialect,
            gluon_execution_contract_satisfied=gluon_execution_ok,
            speedup=best_speedup,
            has_shape_regression=anchor_shape_regression,
            per_shape_speedups=best_shape_speedups,
        ),
        "scope_compliant": True,
        "forbidden_scope_violation": None,
        "scope_escalation_violation": None,
        "scope_infeasible_reported": False,
        "required_patch_target_symbols": required_patch_target_symbols,
        "forbidden_patch_target_symbols": forbidden_patch_target_symbols,
        "llm_selection_analysis": (
            f"Deterministic: baseline={best_baseline_comparison_ms:.4f}ms ({baseline_source}), "
            f"candidate={best_candidate_ms:.4f}ms from {best_patch_id}. "
            f"Speedup={best_speedup:.4f}x. Patch={best_patch_size}B."
        ),
    }
    result.update(optional_meta)
    return result


def rewrite_best_results(patch_dir: Path) -> dict[str, Any] | None:
    """Overwrite ``best_results.json`` with deterministic selection if possible.

    Uses the canonical baseline from benchmark_baseline.txt.  Empty patches or
    patches that miss explicit target-symbol metadata are invalidated.
    """
    det = compute_best_patch(patch_dir)
    existing_path = patch_dir / "best_results.json"
    original_bl = _find_original_baseline_ms(patch_dir)
    task_meta = _find_task_metadata_for_patch_dir(patch_dir)
    required_patch_target_symbols = _metadata_list(task_meta.get("required_patch_target_symbols"))
    forbidden_patch_target_symbols = _metadata_list(task_meta.get("forbidden_patch_target_symbols"))
    required_output_dialect = _required_output_dialect(task_meta, label=patch_dir.name)
    comparison_target = str(task_meta.get("comparison_target") or "true_baseline").strip().lower()
    safe_anchor = str(task_meta.get("safe_anchor") or "").strip()

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
            pf = existing.get("best_patch_file")
            if not pf:
                for candidate_patch in sorted(patch_dir.glob("patch_*.patch")):
                    patch_text = candidate_patch.read_text(errors="replace")
                    scope_escalation = _scope_escalation_violation(patch_text, task_meta)
                    if scope_escalation:
                        existing["best_patch_id"] = None
                        existing["best_patch_file"] = None
                        existing["best_patch_speedup"] = 0.0
                        existing["required_output_dialect"] = required_output_dialect
                        existing["scope_compliant"] = False
                        existing["scope_escalation_violation"] = scope_escalation
                        existing["gluon_execution_contract_satisfied"] = False
                        existing["gluon_evidence_summary"] = _gluon_evidence_summary(
                            required_output_dialect=required_output_dialect,
                            scope_escalation=scope_escalation,
                        )
                        existing["llm_selection_analysis"] = (
                            existing.get("llm_selection_analysis") or ""
                        ) + f" [Invalidated: {scope_escalation}]"
                        existing_path.write_text(json.dumps(existing, indent=2))
                        return existing

            if pf and Path(pf).exists():
                patch_text = Path(pf).read_text(errors="replace")
                if _patch_touches_backup_file(patch_text):
                    logger.warning(
                        "rewrite_best_results(%s): selected patch touches backup or temporary files.",
                        patch_dir.name,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["scope_compliant"] = False
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + " [Invalidated: selected patch touched backup or temporary files]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing
                static_contract_error = _required_amd_gluon_static_contract_error(
                    patch_text,
                    required_output_dialect,
                    task_meta,
                )
                if static_contract_error:
                    logger.warning(
                        "rewrite_best_results(%s): selected patch violates static AMD Gluon contract: %s.",
                        patch_dir.name,
                        static_contract_error,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["required_output_dialect"] = required_output_dialect
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + f" [Invalidated: {static_contract_error}]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing
                scope_escalation = _scope_escalation_violation(patch_text, task_meta)
                if scope_escalation:
                    logger.warning(
                        "rewrite_best_results(%s): selected patch violates scope contract: %s.",
                        patch_dir.name,
                        scope_escalation,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["required_output_dialect"] = required_output_dialect
                    existing["scope_compliant"] = False
                    existing["scope_escalation_violation"] = scope_escalation
                    existing["gluon_execution_contract_satisfied"] = False
                    existing["gluon_evidence_summary"] = _gluon_evidence_summary(
                        required_output_dialect=required_output_dialect,
                        scope_escalation=scope_escalation,
                    )
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + f" [Invalidated: {scope_escalation}]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing

            if pf and Path(pf).exists() and required_patch_target_symbols:
                patch_text = Path(pf).read_text(errors="replace")
                if not _patch_touches_any_target_symbol(patch_text, required_patch_target_symbols):
                    logger.warning(
                        "rewrite_best_results(%s): selected patch misses required target symbols %s.",
                        patch_dir.name,
                        required_patch_target_symbols,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["required_patch_target_symbols"] = required_patch_target_symbols
                    existing["scope_compliant"] = False
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + " [Invalidated: selected patch did not touch required target symbols]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing
                forbidden_target = _patch_touches_forbidden_target_symbol(patch_text, forbidden_patch_target_symbols)
                if forbidden_target:
                    logger.warning(
                        "rewrite_best_results(%s): selected patch touches forbidden target scope %s.",
                        patch_dir.name,
                        forbidden_target,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["forbidden_patch_target_symbols"] = forbidden_patch_target_symbols
                    existing["scope_compliant"] = False
                    existing["forbidden_scope_violation"] = forbidden_target
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + f" [Invalidated: selected patch touched forbidden target scope {forbidden_target}]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing

            if pf and Path(pf).exists():
                patch_text = Path(pf).read_text(errors="replace")
                existing["legacy_output_dialect_classification"] = _legacy_classify_patch_output_dialect(patch_text)
                api_contract = classify_gluon_api_contract(patch_text, task_meta)
                layout_contract = classify_layout_contract(patch_text, task_meta)
                existing.update(api_contract)
                existing.update(layout_contract)
                if api_contract["gluon_api_contract_status"] in {"leftover_tl_device_api", "invalid_plain_fallback"}:
                    logger.warning(
                        "rewrite_best_results(%s): selected patch violates Gluon API contract: %s.",
                        patch_dir.name,
                        api_contract["forbidden_tl_symbols_seen"],
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["scope_compliant"] = False
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + " [Invalidated: selected patch violated Gluon API contract]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing
                if layout_contract["layout_contract_status"] == "runtime_layout_object":
                    logger.warning(
                        "rewrite_best_results(%s): selected patch violates layout contract.",
                        patch_dir.name,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["scope_compliant"] = False
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + " [Invalidated: selected patch created a runtime layout object]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing
                actual_output_dialect = classify_patch_output_dialect(patch_text)
                if not _dialect_contract_satisfied(required_output_dialect, actual_output_dialect):
                    logger.warning(
                        "rewrite_best_results(%s): selected patch violates required_output_dialect=%s (actual=%s).",
                        patch_dir.name,
                        required_output_dialect,
                        actual_output_dialect,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["required_output_dialect"] = required_output_dialect
                    existing["actual_output_dialect"] = actual_output_dialect
                    existing["dialect_contract_satisfied"] = False
                    existing["scope_compliant"] = False
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + " [Invalidated: selected patch violated required output dialect]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing
                if not _gluon_execution_contract_satisfied(
                    patch_text,
                    required_symbols=required_patch_target_symbols,
                    required_output_dialect=required_output_dialect,
                ):
                    logger.warning(
                        "rewrite_best_results(%s): selected patch defines Gluon helper without executing it.",
                        patch_dir.name,
                    )
                    existing["best_patch_id"] = None
                    existing["best_patch_file"] = None
                    existing["best_patch_speedup"] = 0.0
                    existing["required_patch_target_symbols"] = required_patch_target_symbols
                    existing["gluon_execution_contract_satisfied"] = False
                    existing["scope_compliant"] = False
                    existing["llm_selection_analysis"] = (
                        existing.get("llm_selection_analysis") or ""
                    ) + " [Invalidated: Gluon helper was defined but not executed for required target]"
                    existing_path.write_text(json.dumps(existing, indent=2))
                    return existing

            if pf and Path(pf).exists() and Path(pf).stat().st_size == 0:
                logger.warning("rewrite_best_results(%s): empty patch; invalidating selection.", patch_dir.name)
                existing["best_patch_id"] = None
                existing["best_patch_file"] = None
                existing["best_patch_speedup"] = 0.0
                existing["scope_compliant"] = False
                existing["llm_selection_analysis"] = (
                    existing.get("llm_selection_analysis") or ""
                ) + " [Invalidated: patch is empty (0 bytes)]"
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            if original_bl is not None:
                logger.info(
                    "rewrite_best_results(%s): no eligible patch found against %s.",
                    patch_dir.name,
                    comparison_target,
                )
                existing["best_patch_speedup"] = 0.0
                existing["best_patch_id"] = None
                existing["best_patch_file"] = None
                existing["baseline_latency_ms"] = original_bl
                existing["baseline_source"] = "benchmark_baseline.txt"
                existing["comparison_target"] = comparison_target
                if safe_anchor:
                    existing["safe_anchor"] = safe_anchor
                existing["gluon_evidence_summary"] = _gluon_evidence_summary(
                    required_output_dialect=required_output_dialect,
                    actual_output_dialect=str(existing.get("actual_output_dialect") or "unknown"),
                    gluon_execution_contract_satisfied=bool(existing.get("gluon_execution_contract_satisfied")),
                    speedup=float(existing.get("best_patch_speedup") or 0.0),
                    has_shape_regression=bool(existing.get("has_significant_shape_regression")),
                    per_shape_speedups=existing.get("per_shape_speedups") if isinstance(existing.get("per_shape_speedups"), dict) else None,
                )
                existing["llm_selection_analysis"] = (
                    existing.get("llm_selection_analysis") or ""
                ) + f" [Invalidated: no eligible patch against {comparison_target}]"
                existing_path.write_text(json.dumps(existing, indent=2))
                return existing

            return existing
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("rewrite_best_results(%s): failed to read existing best_results.json: %s", patch_dir.name, exc)

    return None
