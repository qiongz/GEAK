"""Shared dataclasses and constants for kernel/test/benchmark discovery.

These types are used across the GEAK pipeline (orchestrator, task generator,
task planner, preprocessor, etc.).  The actual discovery logic lives in
``automated_test_discovery``.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# Shared extension constants -- single source of truth
CPP_EXTENSIONS = frozenset((".cpp", ".cc", ".cu", ".hip", ".cxx"))
CPP_HEADER_EXTENSIONS = frozenset((".h", ".hpp"))
ALL_KERNEL_EXTENSIONS = frozenset((".py",)) | CPP_EXTENSIONS

PLAIN_TRITON_DIALECT = "plain_triton"
NV_GLUON_DIALECT = "nv_gluon"
AMD_GLUON_DIALECT = "amd_gluon"
ALL_INPUT_DIALECTS = frozenset((PLAIN_TRITON_DIALECT, NV_GLUON_DIALECT, AMD_GLUON_DIALECT))

GLUON_FEATURE_MODE_OFF = "off"
GLUON_FEATURE_MODE_AUTO = "auto"
GLUON_FEATURE_MODE_FORCE = "force"
ALL_GLUON_FEATURE_MODES = frozenset(
    (GLUON_FEATURE_MODE_OFF, GLUON_FEATURE_MODE_AUTO, GLUON_FEATURE_MODE_FORCE)
)

GLUON_BASELINE_PROFILE_RAW = "raw"
GLUON_BASELINE_PROFILE_MI3XX = "mi3xx"
ALL_GLUON_BASELINE_PROFILES = frozenset((GLUON_BASELINE_PROFILE_RAW, GLUON_BASELINE_PROFILE_MI3XX))

DEFAULT_TARGET_BACKEND = "hip/gfx942"
_GFX_ARCH_RE = re.compile(r"\bgfx[0-9a-zA-Z]+\b")

# Shape coverage profiles (planner-facing).
#
# Both correctness and performance evaluations on AgentKernelArena and
# similar Triton-family benchmarks now share the same multi-shape case
# set. Planner / worker prompts must classify each candidate against this
# coverage profile so that single-shape investments do not dominate the
# search budget on multi-shape benchmarks.
SHAPE_COVERAGE_UNKNOWN = "unknown"
SHAPE_COVERAGE_SINGLE = "single"
SHAPE_COVERAGE_MULTI = "multi"
SHAPE_COVERAGE_BUCKETED = "bucketed"
ALL_SHAPE_COVERAGE_PROFILES = frozenset(
    (
        SHAPE_COVERAGE_UNKNOWN,
        SHAPE_COVERAGE_SINGLE,
        SHAPE_COVERAGE_MULTI,
        SHAPE_COVERAGE_BUCKETED,
    )
)
DEFAULT_SHAPE_COVERAGE_PROFILE = SHAPE_COVERAGE_UNKNOWN
# A coverage is treated as "bucketed" when at least this many cases span
# more than one order of magnitude on at least one numeric dimension.
_SHAPE_BUCKET_MIN_CASES = 4
_SHAPE_BUCKET_MIN_RATIO = 4.0
GENERAL_SKILL_TIER = "general"
AUTHORING_SAFE_SKILL_TIER = "authoring_safe"
BENCHMARK_SAFE_SKILL_TIER = "benchmark_safe"
DEFAULT_ALLOWED_OUTPUT_DIALECTS = (PLAIN_TRITON_DIALECT,)
PLAIN_TRITON_ONLY_SEARCH_POLICY = "plain_triton_only"
COMPARE_ALLOWED_OUTPUTS_WITHOUT_BIAS_POLICY = "compare_allowed_outputs_without_bias"
PREFER_AMD_GLUON_IF_VIABLE_POLICY = "prefer_amd_gluon_if_viable_else_plain_triton"
REQUIRE_AMD_GLUON_POLICY = "require_amd_gluon"
ALL_OUTPUT_DIALECT_SEARCH_POLICIES = frozenset(
    (
        PLAIN_TRITON_ONLY_SEARCH_POLICY,
        COMPARE_ALLOWED_OUTPUTS_WITHOUT_BIAS_POLICY,
        PREFER_AMD_GLUON_IF_VIABLE_POLICY,
        REQUIRE_AMD_GLUON_POLICY,
    )
)

_GLUON_CORE_MARKERS = (
    "@gluon.jit",
    "triton.experimental.gluon",
    "from triton.experimental import gluon",
)
_AMD_GLUON_MARKERS = (
    ".amd.",
    "amd_gluon",
    "cdna3",
    "cdna4",
    "gfx942",
    "gfx950",
)
_NV_GLUON_MARKERS = (
    ".nvidia.",
    "nv_gluon",
    "hopper",
    "ampere",
    "blackwell",
    "wgmma",
    "tensordescriptor",
)


def _read_kernel_text(kernel_path: Path, content: str | None = None) -> str:
    """Return source text for dialect inference."""
    if content is not None:
        return content
    try:
        return kernel_path.read_text(errors="ignore")
    except OSError:
        return ""


def _normalize_input_dialect(value: Any) -> str | None:
    """Normalize input dialect names to the canonical enum."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    aliases = {
        "plain": PLAIN_TRITON_DIALECT,
        "plain_triton": PLAIN_TRITON_DIALECT,
        "triton": PLAIN_TRITON_DIALECT,
        "nv": NV_GLUON_DIALECT,
        "nv_gluon": NV_GLUON_DIALECT,
        "nvidia_gluon": NV_GLUON_DIALECT,
        "triton+nv_gluon": NV_GLUON_DIALECT,
        "amd": AMD_GLUON_DIALECT,
        "amd_gluon": AMD_GLUON_DIALECT,
        "triton+amd_gluon": AMD_GLUON_DIALECT,
    }
    return aliases.get(text, text if text in ALL_INPUT_DIALECTS else None)


def _normalize_gluon_feature_mode(
    value: Any,
    *,
    input_dialect: str,
    kernel_type: str = "triton",
) -> str:
    """Normalize the Gluon feature gate.

    Triton-family kernels default to ``auto`` so AMD Gluon is explored as a
    candidate output. Non-Triton kernels stay ``off`` because Gluon guidance
    only applies on the Triton route.
    """
    default_mode = (
        GLUON_FEATURE_MODE_AUTO
        if str(kernel_type).strip().lower() == "triton"
        else GLUON_FEATURE_MODE_OFF
    )
    text = str(value or "").strip().lower()
    if not text:
        return default_mode
    aliases = {
        "on": GLUON_FEATURE_MODE_AUTO,
        "enabled": GLUON_FEATURE_MODE_AUTO,
        "true": GLUON_FEATURE_MODE_AUTO,
        "gluon-on": GLUON_FEATURE_MODE_AUTO,
        "gluon_on": GLUON_FEATURE_MODE_AUTO,
        "gluon-enabled": GLUON_FEATURE_MODE_AUTO,
        "disabled": GLUON_FEATURE_MODE_OFF,
        "false": GLUON_FEATURE_MODE_OFF,
        "gluon-off": GLUON_FEATURE_MODE_OFF,
        "gluon_off": GLUON_FEATURE_MODE_OFF,
    }
    text = aliases.get(text, text)
    return text if text in ALL_GLUON_FEATURE_MODES else default_mode


def _normalize_gluon_baseline_profile(value: Any) -> str:
    """Normalize the Gluon baseline profile."""
    text = str(value or "").strip().lower()
    if not text:
        return GLUON_BASELINE_PROFILE_RAW
    aliases = {
        "mi300": GLUON_BASELINE_PROFILE_MI3XX,
        "mi300x": GLUON_BASELINE_PROFILE_MI3XX,
        "mi325x": GLUON_BASELINE_PROFILE_MI3XX,
    }
    text = aliases.get(text, text)
    return text if text in ALL_GLUON_BASELINE_PROFILES else GLUON_BASELINE_PROFILE_RAW


def _normalize_target_backend(value: Any) -> str | None:
    """Normalize target backend strings such as ``gfx942`` or ``hip/gfx942``."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    match = _GFX_ARCH_RE.search(text)
    if match:
        return f"hip/{match.group(0)}"
    return text


@lru_cache(maxsize=1)
def detect_rocm_target_backend() -> str | None:
    """Best-effort early ROCm target detection using ``rocminfo``.

    This intentionally does not use ``sudo``. In restricted environments users
    should set ``GEAK_TARGET_BACKEND`` explicitly.
    """
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None

    matches = _GFX_ARCH_RE.findall(result.stdout or "")
    if not matches:
        return None
    counts = Counter(matches)
    arch = counts.most_common(1)[0][0]
    return f"hip/{arch.lower()}"


def resolve_target_backend(value: Any = None) -> str:
    """Resolve target backend for Triton-family planning.

    Priority: explicit value, ``GEAK_TARGET_BACKEND``, early ``rocminfo``
    detection, then the repository default.
    """
    return (
        _normalize_target_backend(value)
        or _normalize_target_backend(os.getenv("GEAK_TARGET_BACKEND"))
        or detect_rocm_target_backend()
        or DEFAULT_TARGET_BACKEND
    )


def _normalize_allowed_output_dialects(value: Any) -> list[str]:
    """Normalize a sequence or comma-delimited string of output dialects."""
    if not value:
        return list(DEFAULT_ALLOWED_OUTPUT_DIALECTS)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    else:
        raw_items = [str(item).strip() for item in value]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _normalize_input_dialect(item)
        if text is None or text == NV_GLUON_DIALECT or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized or list(DEFAULT_ALLOWED_OUTPUT_DIALECTS)


def _normalize_preferred_output_dialects(
    value: Any,
    *,
    allowed_output_dialects: list[str],
) -> list[str]:
    """Normalize an ordered output-dialect preference list.

    The resulting order is always a subset of ``allowed_output_dialects`` and
    preserves any missing allowed dialects at the end so downstream consumers
    still see the full search space.
    """
    allowed = _normalize_allowed_output_dialects(allowed_output_dialects)
    if not value:
        return list(allowed)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    else:
        raw_items = [str(item).strip() for item in value]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _normalize_input_dialect(item)
        if text is None or text not in allowed or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    for dialect in allowed:
        if dialect not in seen:
            normalized.append(dialect)
    return normalized or list(allowed)


def _normalize_output_dialect_search_policy(value: Any) -> str | None:
    """Normalize output-dialect planning policy names."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    aliases = {
        "plain_only": PLAIN_TRITON_ONLY_SEARCH_POLICY,
        "plain_triton_only": PLAIN_TRITON_ONLY_SEARCH_POLICY,
        "compare_allowed_outputs": COMPARE_ALLOWED_OUTPUTS_WITHOUT_BIAS_POLICY,
        "compare": COMPARE_ALLOWED_OUTPUTS_WITHOUT_BIAS_POLICY,
        "prefer_gluon": PREFER_AMD_GLUON_IF_VIABLE_POLICY,
        "prefer_amd_gluon": PREFER_AMD_GLUON_IF_VIABLE_POLICY,
        "prefer_amd_gluon_if_viable": PREFER_AMD_GLUON_IF_VIABLE_POLICY,
        "prefer_amd_gluon_then_fallback_plain_triton": PREFER_AMD_GLUON_IF_VIABLE_POLICY,
        "require_gluon": REQUIRE_AMD_GLUON_POLICY,
        "require_amd_gluon": REQUIRE_AMD_GLUON_POLICY,
    }
    text = aliases.get(text, text)
    return text if text in ALL_OUTPUT_DIALECT_SEARCH_POLICIES else None


def infer_input_dialect(
    kernel_path: Path,
    kernel_type: str = "unknown",
    *,
    content: str | None = None,
) -> str:
    """Infer the Triton-family input dialect for a kernel file."""
    normalized_kernel_type = str(kernel_type).strip().lower()
    if normalized_kernel_type in {"hip", "ck", "asm", "cuda", "other"}:
        return PLAIN_TRITON_DIALECT

    text = _read_kernel_text(kernel_path, content).lower()
    stem = kernel_path.stem.lower()
    is_gluon = any(marker in text for marker in _GLUON_CORE_MARKERS)
    if not is_gluon:
        return PLAIN_TRITON_DIALECT

    if any(marker in text for marker in _AMD_GLUON_MARKERS) or stem.endswith("_amd"):
        return AMD_GLUON_DIALECT
    if any(marker in text for marker in _NV_GLUON_MARKERS) or stem.endswith("_nv"):
        return NV_GLUON_DIALECT
    # Prefer AMD when a Gluon file is otherwise ambiguous because the supported
    # optimized Gluon output dialect is AMD-only in this product path.
    return AMD_GLUON_DIALECT


def derive_allowed_output_dialects(input_dialect: str, gluon_feature_mode: str) -> list[str]:
    """Derive the output search space from the input dialect and feature gate."""
    if gluon_feature_mode == GLUON_FEATURE_MODE_FORCE:
        return [AMD_GLUON_DIALECT]
    if gluon_feature_mode != GLUON_FEATURE_MODE_OFF:
        return [PLAIN_TRITON_DIALECT, AMD_GLUON_DIALECT]
    return [PLAIN_TRITON_DIALECT]


def derive_preferred_output_dialects(
    input_dialect: str,
    gluon_feature_mode: str,
    allowed_output_dialects: list[str],
) -> list[str]:
    """Derive the ordered output-dialect preference list for planning."""
    outputs = _normalize_allowed_output_dialects(allowed_output_dialects)
    if gluon_feature_mode == GLUON_FEATURE_MODE_FORCE:
        return [AMD_GLUON_DIALECT]
    if (
        AMD_GLUON_DIALECT in outputs
        and gluon_feature_mode != GLUON_FEATURE_MODE_OFF
    ):
        return [AMD_GLUON_DIALECT] + [dialect for dialect in outputs if dialect != AMD_GLUON_DIALECT]
    return outputs


def derive_output_dialect_search_policy(
    input_dialect: str,
    gluon_feature_mode: str,
    preferred_output_dialects: list[str],
    allowed_output_dialects: list[str],
) -> str:
    """Derive the planner's output-dialect search policy."""
    preferred = _normalize_preferred_output_dialects(
        preferred_output_dialects,
        allowed_output_dialects=allowed_output_dialects,
    )
    allowed = _normalize_allowed_output_dialects(allowed_output_dialects)
    if gluon_feature_mode == GLUON_FEATURE_MODE_FORCE:
        return REQUIRE_AMD_GLUON_POLICY
    if preferred == [PLAIN_TRITON_DIALECT] and AMD_GLUON_DIALECT not in allowed:
        return PLAIN_TRITON_ONLY_SEARCH_POLICY
    if preferred and preferred[0] == AMD_GLUON_DIALECT and AMD_GLUON_DIALECT in allowed:
        if allowed == [AMD_GLUON_DIALECT]:
            return REQUIRE_AMD_GLUON_POLICY
        return PREFER_AMD_GLUON_IF_VIABLE_POLICY
    return COMPARE_ALLOWED_OUTPUTS_WITHOUT_BIAS_POLICY


def _normalize_shape_coverage_profile(value: Any) -> str | None:
    """Normalize a shape-coverage profile string."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    aliases = {
        "none": SHAPE_COVERAGE_UNKNOWN,
        "n/a": SHAPE_COVERAGE_UNKNOWN,
        "1": SHAPE_COVERAGE_SINGLE,
        "single_shape": SHAPE_COVERAGE_SINGLE,
        "single-shape": SHAPE_COVERAGE_SINGLE,
        "multi_shape": SHAPE_COVERAGE_MULTI,
        "multi-shape": SHAPE_COVERAGE_MULTI,
        "shape_bucketed": SHAPE_COVERAGE_BUCKETED,
        "bucket": SHAPE_COVERAGE_BUCKETED,
        "buckets": SHAPE_COVERAGE_BUCKETED,
    }
    text = aliases.get(text, text)
    return text if text in ALL_SHAPE_COVERAGE_PROFILES else None


def _normalize_benchmark_test_cases(value: Any) -> list[dict[str, Any]]:
    """Normalize ``benchmark_test_cases`` payloads into a list of dicts.

    Each entry preserves ``case_id`` and a ``params`` dict, plus optional
    ``baseline_ms`` / ``opt_ms`` / ``speedup`` fields when present.
    """
    if not value:
        return []
    if isinstance(value, dict):
        items = value.get("test_cases") or value.get("cases") or []
    else:
        items = value
    if not isinstance(items, (list, tuple)):
        return []
    def _first_present(raw_case: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in raw_case:
                return raw_case.get(key)
        return None

    def _positive_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _optional_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    cases: list[dict[str, Any]] = []
    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        case_id = str(
            raw.get("case_id")
            or raw.get("test_case_id")
            or raw.get("id")
            or f"case_{idx}"
        )
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        # AgentKernelArena repository tasks often put semantic dimensions in
        # ``metadata`` and only expose a positional ``shape`` list separately.
        if not params and isinstance(raw.get("metadata"), dict):
            params = dict(raw.get("metadata") or {})
        # Tolerate flat shape lists used by some torch2hip-style runners.
        if not params:
            for key in ("shape", "shapes", "input_shape", "input_shapes"):
                if key in raw:
                    params = {key: raw[key]}
                    break
        baseline_ms = _positive_float(_first_present(raw, ("baseline_ms", "ori_time")))
        opt_ms = _positive_float(_first_present(raw, ("opt_ms", "execution_time_ms", "opt_time")))
        cases.append(
            {
                "case_id": case_id,
                "params": dict(params or {}),
                "baseline_ms": baseline_ms,
                "opt_ms": opt_ms,
                "speedup": _optional_float(raw.get("speedup")),
            }
        )
    return cases


def _shape_signature_text(test_cases: list[dict[str, Any]]) -> str:
    """Serialize ``benchmark_test_cases`` params into a search signature."""
    chunks: list[str] = []
    for c in test_cases:
        params = c.get("params") or {}
        for key, val in params.items():
            chunks.append(f"{key}={val}")
    return " ".join(chunks).lower()


def _looks_bucketed(test_cases: list[dict[str, Any]]) -> bool:
    """Return True if cases span more than one order of magnitude on any axis."""
    if len(test_cases) < _SHAPE_BUCKET_MIN_CASES:
        return False
    axis_values: dict[str, list[float]] = {}
    for c in test_cases:
        params = c.get("params") or {}
        for key, val in params.items():
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)) and val > 0:
                axis_values.setdefault(str(key), []).append(float(val))
            elif isinstance(val, (list, tuple)):
                for idx, item in enumerate(val):
                    if isinstance(item, (int, float)) and not isinstance(item, bool) and item > 0:
                        axis_values.setdefault(f"{key}_dim_{idx}", []).append(float(item))
    for values in axis_values.values():
        if len(values) >= _SHAPE_BUCKET_MIN_CASES and max(values) / min(values) >= _SHAPE_BUCKET_MIN_RATIO:
            return True
    return False


def derive_shape_coverage_profile(
    *,
    benchmark_shape_count: Any = None,
    benchmark_test_cases: Any = None,
) -> str:
    """Map shape count and case parameters to a planner-facing profile.

    The classification is deliberately conservative:

    - ``unknown`` when no signal is available.
    - ``single`` for one observed case.
    - ``multi`` for >=2 cases without obvious bucket spread.
    - ``bucketed`` when at least four cases span more than one order of
      magnitude on a numeric dimension (e.g. M/N/K, seq_len, hidden_size).
    """
    cases = _normalize_benchmark_test_cases(benchmark_test_cases)
    try:
        count = int(benchmark_shape_count) if benchmark_shape_count is not None else 0
    except (TypeError, ValueError):
        count = 0
    if not count and cases:
        count = len(cases)
    if count <= 0:
        return SHAPE_COVERAGE_UNKNOWN
    if count == 1:
        return SHAPE_COVERAGE_SINGLE
    if cases and _looks_bucketed(cases):
        return SHAPE_COVERAGE_BUCKETED
    return SHAPE_COVERAGE_MULTI


def feature_uses_gluon_guidance(
    kernel_type: Any,
    *,
    input_dialect: Any = None,
    gluon_feature_mode: Any = None,
    allowed_output_dialects: Any = None,
) -> bool:
    """Return whether a task should receive Gluon-specific guidance.

    This is a framework-level, metadata-driven decision and must stay generic:
    it should not depend on any particular example kernel or task label.

    Prefer :func:`feature_uses_gluon_guidance_from_meta` at call sites that
    already hold a normalized ``feature_meta`` dict; that wrapper guarantees
    ``kernel_type`` is read from the same source as the rest of the dict
    and avoids the historical ``feature_uses_gluon_guidance("triton", ...)``
    hard-coded literals.
    """
    if str(kernel_type or "").strip().lower() != "triton":
        return False

    normalized_input_dialect = _normalize_input_dialect(input_dialect) or PLAIN_TRITON_DIALECT
    normalized_feature_mode = _normalize_gluon_feature_mode(
        gluon_feature_mode,
        input_dialect=normalized_input_dialect,
        kernel_type=kernel_type,
    )
    if normalized_feature_mode == GLUON_FEATURE_MODE_OFF:
        return False
    normalized_outputs = (
        _normalize_allowed_output_dialects(allowed_output_dialects)
        if allowed_output_dialects
        else derive_allowed_output_dialects(normalized_input_dialect, normalized_feature_mode)
    )

    return (
        normalized_feature_mode != GLUON_FEATURE_MODE_OFF
        or AMD_GLUON_DIALECT in normalized_outputs
    )


def feature_uses_gluon_guidance_from_meta(
    feature_meta: dict[str, Any] | None,
) -> bool:
    """Convenience wrapper around :func:`feature_uses_gluon_guidance`.

    Reads ``kernel_type`` / ``input_dialect`` / ``gluon_feature_mode`` /
    ``allowed_output_dialects`` from a normalized ``feature_meta`` dict
    (as produced by :func:`build_gluon_feature_metadata`). This is the
    preferred entry point in code paths that already hold the dict so
    they don't have to hard-code ``"triton"`` at the call site.
    """
    if not feature_meta:
        return False
    if not feature_meta.get("kernel_type"):
        return False
    return feature_uses_gluon_guidance(
        feature_meta.get("kernel_type"),
        input_dialect=feature_meta.get("input_dialect"),
        gluon_feature_mode=feature_meta.get("gluon_feature_mode"),
        allowed_output_dialects=feature_meta.get("allowed_output_dialects"),
    )


def build_gluon_feature_metadata(
    kernel_path: Path,
    kernel_type: str = "unknown",
    *,
    input_dialect: Any = None,
    gluon_feature_mode: Any = None,
    gluon_baseline_profile: Any = None,
    allowed_output_dialects: Any = None,
    preferred_output_dialects: Any = None,
    output_dialect_search_policy: Any = None,
    target_backend: Any = None,
    benchmark_shape_count: Any = None,
    benchmark_test_cases: Any = None,
    shape_coverage_profile: Any = None,
    content: str | None = None,
) -> dict[str, Any]:
    """Build the shared Gluon feature metadata contract."""
    normalized_input_dialect = _normalize_input_dialect(input_dialect) or infer_input_dialect(
        kernel_path,
        kernel_type,
        content=content,
    )
    normalized_feature_mode = _normalize_gluon_feature_mode(
        gluon_feature_mode,
        input_dialect=normalized_input_dialect,
        kernel_type=kernel_type,
    )
    normalized_profile = _normalize_gluon_baseline_profile(gluon_baseline_profile)
    if normalized_feature_mode == GLUON_FEATURE_MODE_OFF:
        normalized_profile = GLUON_BASELINE_PROFILE_RAW
    normalized_target_backend = resolve_target_backend(target_backend)
    if normalized_feature_mode == GLUON_FEATURE_MODE_FORCE:
        normalized_outputs = [AMD_GLUON_DIALECT]
    elif normalized_feature_mode == GLUON_FEATURE_MODE_OFF:
        normalized_outputs = [PLAIN_TRITON_DIALECT]
    else:
        normalized_outputs = (
            _normalize_allowed_output_dialects(allowed_output_dialects)
            if allowed_output_dialects
            else derive_allowed_output_dialects(normalized_input_dialect, normalized_feature_mode)
        )
    normalized_preferred_outputs = (
        _normalize_preferred_output_dialects(
            preferred_output_dialects,
            allowed_output_dialects=normalized_outputs,
        )
        if preferred_output_dialects
        else derive_preferred_output_dialects(
            normalized_input_dialect,
            normalized_feature_mode,
            normalized_outputs,
        )
    )
    normalized_search_policy = _normalize_output_dialect_search_policy(output_dialect_search_policy) or (
        derive_output_dialect_search_policy(
            normalized_input_dialect,
            normalized_feature_mode,
            normalized_preferred_outputs,
            normalized_outputs,
        )
    )
    normalized_test_cases = _normalize_benchmark_test_cases(benchmark_test_cases)
    try:
        shape_count = (
            int(benchmark_shape_count)
            if benchmark_shape_count is not None
            else (len(normalized_test_cases) or None)
        )
    except (TypeError, ValueError):
        shape_count = len(normalized_test_cases) or None
    normalized_shape_profile = (
        _normalize_shape_coverage_profile(shape_coverage_profile)
        or derive_shape_coverage_profile(
            benchmark_shape_count=shape_count,
            benchmark_test_cases=normalized_test_cases,
        )
    )

    return {
        "kernel_type": str(kernel_type or "unknown").strip().lower() or "unknown",
        "input_dialect": normalized_input_dialect,
        "gluon_feature_mode": normalized_feature_mode,
        "gluon_baseline_profile": normalized_profile,
        "allowed_output_dialects": normalized_outputs,
        "preferred_output_dialects": normalized_preferred_outputs,
        "output_dialect_search_policy": normalized_search_policy,
        "target_backend": normalized_target_backend,
        "benchmark_shape_count": shape_count,
        "benchmark_test_cases": normalized_test_cases,
        "shape_coverage_profile": normalized_shape_profile,
    }


def allowed_skill_tiers_for_feature(
    kernel_type: str,
    *,
    gluon_feature_mode: Any = None,
    gluon_baseline_profile: Any = None,
) -> list[str]:
    """Return the visible skill tiers for a task."""
    if str(kernel_type).strip().lower() != "triton":
        return []
    return [GENERAL_SKILL_TIER]


def build_gluon_feature_prompt_block(feature_meta: dict[str, Any] | None, *, heading: str) -> str:
    """Render the Gluon feature metadata as a Markdown block."""
    if not feature_meta:
        return ""
    input_dialect = str(feature_meta.get("input_dialect") or PLAIN_TRITON_DIALECT)
    gluon_feature_mode = str(feature_meta.get("gluon_feature_mode") or GLUON_FEATURE_MODE_OFF)
    gluon_baseline_profile = str(feature_meta.get("gluon_baseline_profile") or GLUON_BASELINE_PROFILE_RAW)
    outputs = feature_meta.get("allowed_output_dialects") or list(DEFAULT_ALLOWED_OUTPUT_DIALECTS)
    preferred_outputs = feature_meta.get("preferred_output_dialects") or list(outputs)
    search_policy = str(
        feature_meta.get("output_dialect_search_policy") or COMPARE_ALLOWED_OUTPUTS_WITHOUT_BIAS_POLICY
    )
    target_backend = str(feature_meta.get("target_backend") or DEFAULT_TARGET_BACKEND)
    shape_profile = str(
        feature_meta.get("shape_coverage_profile") or DEFAULT_SHAPE_COVERAGE_PROFILE
    )
    shape_count_val = feature_meta.get("benchmark_shape_count")

    lines = [
        heading,
        f"- Input dialect: {input_dialect}",
        f"- Gluon feature mode: {gluon_feature_mode}",
        f"- Gluon baseline profile: {gluon_baseline_profile}",
        f"- Allowed output dialects: {', '.join(str(item) for item in outputs)}",
        f"- Preferred output dialect order: {', '.join(str(item) for item in preferred_outputs)}",
        f"- Output-dialect search policy: {search_policy}",
        f"- Target backend: {target_backend}",
        f"- Shape coverage profile: {shape_profile}"
        + (f" ({int(shape_count_val)} cases)" if isinstance(shape_count_val, int) and shape_count_val > 0 else ""),
    ]

    if search_policy == PREFER_AMD_GLUON_IF_VIABLE_POLICY:
        lines.append("- Prefer AMD Gluon only as a same-direction implementation overlay when it looks structurally promising for this Triton-family input.")
        if input_dialect == NV_GLUON_DIALECT:
            lines.append("- If the input is NVIDIA-oriented Gluon, translate vendor-specific APIs, layouts, or memory paths into AMD-facing Gluon semantics before tuning.")
        elif input_dialect == AMD_GLUON_DIALECT:
            lines.append("- The input is already AMD Gluon; keep the optimized path in amd_gluon space unless the allowed outputs explicitly require a comparison fallback.")
        else:
            lines.append("- Generate an early AMD Gluon candidate only when it names the optimization direction and concrete Gluon overlay reason; otherwise spend width on plain Triton directions.")
        if PLAIN_TRITON_DIALECT in outputs:
            lines.append("- Keep a same-direction plain Triton competitor alive unless the policy explicitly requires AMD Gluon.")
    elif search_policy == REQUIRE_AMD_GLUON_POLICY:
        lines.append("- This run requires an AMD Gluon output path.")
    elif search_policy == PLAIN_TRITON_ONLY_SEARCH_POLICY:
        lines.append("- This run is restricted to plain Triton output only.")

    if gluon_feature_mode == GLUON_FEATURE_MODE_OFF:
        lines.append("- Gluon-specific prompt and skill guidance is disabled.")
        if input_dialect in {NV_GLUON_DIALECT, AMD_GLUON_DIALECT}:
            lines.append("- Existing Gluon input is still valid input; preserve semantics even with the feature gate off.")
    elif gluon_baseline_profile == GLUON_BASELINE_PROFILE_RAW:
        lines.append("- The Gluon feature gate is enabled; rely on the single shared Gluon guide and skill rather than extra profile-specific references.")
        lines.append("- Compare plain Triton and AMD Gluon output paths only when the allowed outputs permit them.")
    else:
        lines.append("- Benchmark-safe MI3xx Gluon guidance is enabled for this run.")
        lines.append("- Allowed guidance is limited to generic AMD Gluon constraints such as explicit layouts, wave64-valid decomposition, AMD CDNA3 memory ops, and MFMA-friendly dataflow.")
        lines.append("- Do NOT rely on example-specific manifests, summaries, translation playbooks, or authoring-only truth data.")

    return "\n".join(lines)


@dataclass
class BuildInfo:
    """How to compile/build a kernel."""

    compiler: str | None = None  # "triton" (JIT), "hipcc", "nvcc", "cmake", None (precompiled)
    build_system: str | None = None  # "setup.py", "CMakeLists.txt", "Makefile", None
    build_dir: Path | None = None  # Where compiled artifacts go
    pybind_module: str | None = None  # e.g., "aiter._C" or "torch.ops.aiter"


@dataclass
class KernelMeta:
    """Cross-module contract for discovered kernel metadata."""

    kernel_path: str = ""
    kernel_name: str = ""
    kernel_type: str = "unknown"  # triton, hip, ck, asm, unknown
    kernel_language: str = "python"  # python, cpp, asm
    function_names: list[str] = field(default_factory=list)
    workspace_path: str = ""
    input_dialect: str = PLAIN_TRITON_DIALECT
    gluon_feature_mode: str = GLUON_FEATURE_MODE_OFF
    gluon_baseline_profile: str = GLUON_BASELINE_PROFILE_RAW
    allowed_output_dialects: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_OUTPUT_DIALECTS))
    preferred_output_dialects: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_OUTPUT_DIALECTS))
    output_dialect_search_policy: str = PLAIN_TRITON_ONLY_SEARCH_POLICY
    target_backend: str = DEFAULT_TARGET_BACKEND


@dataclass
class KernelInfo(KernelMeta):
    """Richer internal kernel record that still satisfies ``KernelMeta``."""

    file_path: Path | None = None
    has_jit_decorator: bool = False
    has_autotune: bool = False
    inner_kernel_path: Path | None = None
    inner_kernel_language: str | None = None
    build_info: BuildInfo | None = None
    fusion_opportunities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.file_path is None and self.kernel_path:
            self.file_path = Path(self.kernel_path)
        if self.file_path is not None:
            self.file_path = self.file_path.resolve()
            if not self.kernel_path:
                self.kernel_path = str(self.file_path)

    def to_meta(self) -> KernelMeta:
        """Return the explicit contract view of this kernel."""
        return KernelMeta(
            kernel_path=self.kernel_path,
            kernel_name=self.kernel_name,
            kernel_type=self.kernel_type,
            kernel_language=self.kernel_language,
            function_names=list(self.function_names),
            workspace_path=self.workspace_path,
            input_dialect=self.input_dialect,
            gluon_feature_mode=self.gluon_feature_mode,
            gluon_baseline_profile=self.gluon_baseline_profile,
            allowed_output_dialects=list(self.allowed_output_dialects),
            preferred_output_dialects=list(self.preferred_output_dialects),
            output_dialect_search_policy=self.output_dialect_search_policy,
            target_backend=self.target_backend,
        )


@dataclass
class KernelNode:
    """A single function/kernel in the dependency graph."""

    name: str
    file_path: Path
    language: str  # "python", "triton", "hip", "ck", "asm"
    node_type: str  # "wrapper", "jit_kernel", "device_func", "asm_module", "torch_op"
    line_range: tuple[int, int] | None = None


@dataclass
class FusionOpportunity:
    """A detected opportunity to fuse operations."""

    description: str
    involved_nodes: list[str] = field(default_factory=list)
    languages: set[str] = field(default_factory=set)
    fusion_type: str = ""  # "sequential_launch", "absorb_wrapper_op", "cross_language"
    estimated_benefit: str = "medium"  # "high", "medium", "low"


@dataclass
class KernelDependencyGraph:
    """Cross-language dependency graph for a kernel and its sub-kernels."""

    root_name: str
    nodes: dict[str, KernelNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    sequential_launches: list[list[str]] = field(default_factory=list)
    wrapper_ops: list[str] = field(default_factory=list)
    language_boundaries: list[tuple[str, str, str]] = field(default_factory=list)
    fusion_opportunities: list[FusionOpportunity] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for inclusion in agent task prompts."""
        lines = [f"Dependency graph for {self.root_name}:"]
        lines.append(f"  Nodes ({len(self.nodes)}):")
        for name, node in self.nodes.items():
            lines.append(f"    - {name} [{node.language}/{node.node_type}] in {node.file_path.name}")
        if self.edges:
            lines.append(f"  Call edges ({len(self.edges)}):")
            for caller, callee in self.edges:
                lines.append(f"    {caller} -> {callee}")
        if self.sequential_launches:
            lines.append("  Sequential kernel launches (potential fusion targets):")
            for group in self.sequential_launches:
                lines.append(f"    [{' -> '.join(group)}]")
        if self.wrapper_ops:
            lines.append("  Wrapper operations between launches:")
            for op in self.wrapper_ops:
                lines.append(f"    - {op}")
        if self.language_boundaries:
            lines.append("  Language boundaries:")
            for caller, callee, boundary in self.language_boundaries:
                lines.append(f"    {caller} -> {callee} ({boundary})")
        if self.fusion_opportunities:
            lines.append(f"  Fusion opportunities ({len(self.fusion_opportunities)}):")
            for opp in self.fusion_opportunities:
                lines.append(f"    - [{opp.estimated_benefit}] {opp.description}")
        return "\n".join(lines)


@dataclass
class TestPatterns:
    """Legacy dataclass kept for backward compatibility.

    The UTA now reads test/benchmark files directly rather than relying
    on regex-extracted patterns. These fields may still be populated by
    older discovery dicts but are not used by the current pipeline.
    """

    tolerances: list[str] = field(default_factory=list)
    input_shapes: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    reference_impls: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)
    shape_variables: list[str] = field(default_factory=list)
    global_variables: list = field(default_factory=list)
    line_count: int = 0


def _parse_shape_size(shape_str: str) -> int | None:
    """Parse a shape string like '(1024, 1024)' and return the product of its dimensions."""
    nums = re.findall(r"\d+", shape_str)
    if not nums:
        return None
    size = 1
    for n in nums:
        size *= int(n)
    return size


def select_shapes_uniform(shapes: list[str], count: int) -> list[str]:
    """Select *count* shapes uniformly spread from smallest to largest.

    Deduplicates, sorts by total element count (product of dimensions),
    then picks evenly-spaced indices so the result spans the full
    small-to-large range.  Returns up to *count* shapes (fewer if the
    input list is shorter).
    """
    seen: set[str] = set()
    sized: list[tuple[int, str]] = []
    for s in shapes:
        if s in seen:
            continue
        seen.add(s)
        sz = _parse_shape_size(s)
        if sz is not None:
            sized.append((sz, s))

    if not sized:
        return []

    sized.sort(key=lambda t: t[0])

    if count <= 0:
        return []
    if count == 1:
        return [sized[len(sized) // 2][1]]

    n = len(sized)
    if n <= count:
        return [s for _, s in sized]

    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    seen_idx: set[int] = set()
    unique_indices: list[int] = []
    for idx in indices:
        if idx not in seen_idx:
            seen_idx.add(idx)
            unique_indices.append(idx)

    return [sized[i][1] for i in unique_indices]


@dataclass
class TestInfo:
    """Information about a discovered test."""

    file_path: Path
    test_type: str  # pytest, script, makefile
    command: str
    confidence: float  # 0-1
    patterns: TestPatterns | None = None


@dataclass
class BenchmarkInfo:
    """Information about a discovered benchmark."""

    file_path: Path
    bench_type: str  # pytest, script, custom
    command: str
    confidence: float
    patterns: TestPatterns | None = None


def _patterns_from_dict(d: dict) -> TestPatterns:
    """Build a ``TestPatterns`` from a raw discovery patterns dict."""
    return TestPatterns(
        import_patterns=d.get("import_patterns", []),
        global_variables=d.get("global_variables", []),
        line_count=d.get("line_count", 0),
        # Legacy fields (older discovery dicts)
        tolerances=d.get("tolerances", []),
        input_shapes=d.get("input_shapes", []),
        dtypes=d.get("dtypes", []),
        reference_impls=d.get("reference_impls", []),
        shape_variables=d.get("shape_variables", []),
    )


_DISCOVERY_TRITON_FAMILY_MARKERS = (
    "@triton",
    "tl.",
    "@gluon.jit",
    "triton.experimental.gluon",
    "from triton.experimental import gluon",
)


def _check_imported_triton_family(content: str, file_path: Path, _depth: int = 0) -> bool:
    """Follow nearby imports to detect Triton-family wrapper modules."""
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

    for match in import_re.finditer(content):
        module_path = match.group(1).replace(".", "/")
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
            if any(marker in imported for marker in _DISCOVERY_TRITON_FAMILY_MARKERS) or "@triton.autotune" in imported:
                return True
            if _depth < 2 and ("import triton" in imported or "triton.experimental.gluon" in imported):
                if _check_imported_triton_family(imported, candidate, _depth + 1):
                    return True
            break
    return False


def _infer_kernel_type(kernel_path: Path) -> str:
    """Infer kernel type when discovery metadata is blank or ``unknown``."""
    ext = kernel_path.suffix.lower()
    if ext == ".py":
        try:
            text = kernel_path.read_text(errors="ignore")
        except OSError:
            return "unknown"
        if any(marker in text for marker in _DISCOVERY_TRITON_FAMILY_MARKERS):
            return "triton"
        if "import triton" in text or "triton.experimental.gluon" in text:
            if _check_imported_triton_family(text, kernel_path):
                return "triton"
            return "triton"
        return "unknown"
    if ext in {".cu", ".hip", ".hpp", ".cpp"}:
        path_lower = str(kernel_path).lower()
        if "composable_kernel" in path_lower or "/ck_" in path_lower or "/ck/" in path_lower:
            return "ck"
        return "hip"
    if ext in {".s", ".asm"}:
        return "asm"
    return "unknown"


def _infer_kernel_language(kernel_path: Path, kernel_type: str) -> str:
    """Derive ``kernel_language`` from file extension and discovery ``type``."""
    if kernel_type == "asm":
        return "asm"
    if kernel_path.suffix == ".py":
        return "python"
    return "cpp"


@dataclass
class DiscoveryResult:
    """Result of the discovery pipeline."""

    kernels: list[KernelInfo] = field(default_factory=list)
    tests: list[TestInfo] = field(default_factory=list)
    benchmarks: list[BenchmarkInfo] = field(default_factory=list)
    dependency_graphs: dict[str, KernelDependencyGraph] = field(default_factory=dict)
    workspace_path: Path | None = None
    needs_user_confirmation: bool = True
    user_provided_test: str | None = None
    user_provided_bench: str | None = None

    @classmethod
    def from_dict(cls, disc_dict: dict, kernel_path: str | Path) -> DiscoveryResult:
        """Build a ``DiscoveryResult`` from a raw discovery JSON dict.

        This is the single canonical conversion path -- all CLI entry
        points should use this instead of inline dict unpacking.
        """
        kp = Path(kernel_path)
        workspace = Path(disc_dict.get("workspace", kp.parent)).resolve()
        kernel_info = disc_dict.get("kernel") or {}
        kernels: list[KernelInfo] = []
        if kernel_info.get("file"):
            raw_type = str(kernel_info.get("type") or "").strip().lower()
            ktype = _infer_kernel_type(kp) if raw_type in {"", "unknown"} else raw_type
            klang = _infer_kernel_language(kp, ktype)
            resolved_kernel_path = Path(kernel_info["file"]).resolve()
            feature_meta = build_gluon_feature_metadata(
                resolved_kernel_path,
                ktype,
                input_dialect=kernel_info.get("input_dialect"),
                gluon_feature_mode=kernel_info.get("gluon_feature_mode"),
                gluon_baseline_profile=kernel_info.get("gluon_baseline_profile"),
                allowed_output_dialects=kernel_info.get("allowed_output_dialects"),
                preferred_output_dialects=kernel_info.get("preferred_output_dialects"),
                output_dialect_search_policy=kernel_info.get("output_dialect_search_policy"),
                target_backend=kernel_info.get("target_backend"),
                benchmark_shape_count=kernel_info.get("benchmark_shape_count"),
                benchmark_test_cases=kernel_info.get("benchmark_test_cases"),
                shape_coverage_profile=kernel_info.get("shape_coverage_profile"),
            )

            _build_info: BuildInfo | None = None
            if klang == "cpp":
                _repo = kp.parent
                while _repo != _repo.parent:
                    if (_repo / "setup.py").exists():
                        _build_info = BuildInfo(compiler="hipcc", build_system="setup.py", build_dir=_repo)
                        break
                    if (_repo / "CMakeLists.txt").exists():
                        _build_info = BuildInfo(compiler="hipcc", build_system="CMakeLists.txt", build_dir=_repo)
                        break
                    if (_repo / "Makefile").exists():
                        _build_info = BuildInfo(compiler="hipcc", build_system="Makefile", build_dir=_repo)
                        break
                    _repo = _repo.parent

            kernels.append(
                KernelInfo(
                    kernel_path=str(resolved_kernel_path),
                    kernel_name=kernel_info.get("name", kp.stem),
                    kernel_type=ktype,
                    kernel_language=klang,
                    function_names=kernel_info.get("functions", []),
                    workspace_path=str(workspace),
                    file_path=resolved_kernel_path,
                    build_info=_build_info,
                    input_dialect=feature_meta["input_dialect"],
                    gluon_feature_mode=feature_meta["gluon_feature_mode"],
                    gluon_baseline_profile=feature_meta["gluon_baseline_profile"],
                    allowed_output_dialects=list(feature_meta["allowed_output_dialects"]),
                    preferred_output_dialects=list(feature_meta["preferred_output_dialects"]),
                    output_dialect_search_policy=feature_meta["output_dialect_search_policy"],
                    target_backend=feature_meta["target_backend"],
                )
            )
        tests = [
            TestInfo(
                file_path=Path(t["file"]),
                test_type=t.get("type", "script"),
                command=t.get("command", ""),
                confidence=t.get("confidence", 0.5),
                patterns=_patterns_from_dict(t.get("patterns")) if t.get("patterns") else None,
            )
            for t in (disc_dict.get("tests") or [])
        ]
        benchmarks = [
            BenchmarkInfo(
                file_path=Path(b["file"]),
                bench_type=b.get("type", "script"),
                command=b.get("command", ""),
                confidence=b.get("confidence", 0.5),
                patterns=_patterns_from_dict(b.get("patterns")) if b.get("patterns") else None,
            )
            for b in (disc_dict.get("benchmarks") or [])
        ]
        return cls(
            kernels=kernels,
            tests=tests,
            benchmarks=benchmarks,
            workspace_path=workspace,
        )
