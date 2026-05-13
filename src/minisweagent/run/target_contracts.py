"""Shared helpers for task target-scope contracts."""

from __future__ import annotations

import re

_TARGET_GROUP_RE = re.compile(
    r"(?:expressions?|components?|symbols?|variables?|paths?|loads?|stores?|tensors?)\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_STAGE_SCOPE_RE = re.compile(
    r"\b(stage[\s_-]*\d+)\b(?=\s+(?:inner\s+)?(?:loop|kernel|path|subpath|helper|loads?|stores?|layout|overlay|component)\b)",
    re.IGNORECASE,
)
_DO_NOT_RE = re.compile(r"^\s*(?:do\s+not|don't)\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_TARGET_LINE_MARKERS = (
    "allowed change",
    "concrete changes needed",
    "apply the same",
    "apply this",
    "also apply",
    "in `",
    "replace",
    "rewrite",
    "the wrapper function",
    "wrapper function",
    "find `",
    "focus on",
    "specifically the",
)
_REFERENCE_TARGET_LINE_MARKERS = (
    "look at",
    "reference pattern",
    "for the reference",
    "study the existing",
    "docs",
    "read ",
)
_BARE_ACTION_SYMBOL_RE = re.compile(
    r"\b(_?[A-Za-z][A-Za-z0-9_]*(?:kernel|wrapper|dispatch|gluon|decode)[A-Za-z0-9_]*)\b"
)
_ABSTRACT_ATOMIC_COMPONENT_SYMBOLS = frozenset(
    {
        "wrapper_shape_dispatch",
        "layout_parent_slice",
        "load_store_buffer",
        "matrix_operand_mfma",
        "reduction_accumulator",
        "state_update_softmax",
        "epilogue_output_store",
        "scheduler_launch_runtime",
        "source_contract_integration",
        "index_map",
        "mask_boundary",
        "load_store",
        "layout_broadcast",
        "matrix_operand",
        "scale_dtype",
        "selection_update",
        "state_update",
        "epilogue_fusion",
        "shape_dispatch",
        "wrapper_integration",
        "scheduler_launch",
    }
)


def _normalize_stage_symbol(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value.strip().lower())


def target_symbols_from_scoped_text(value: str | None) -> list[str]:
    """Infer explicit target symbols from scoped task text.

    This intentionally handles generic task-contract forms only:
    backticked local names, named target lists such as ``expressions (a, b)``,
    and stage-scoped phrases such as ``stage1 inner loop``. It does not infer
    arbitrary identifiers from prose.
    """
    text = str(value or "")
    symbols: list[str] = []
    symbols.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))
    for group_match in _TARGET_GROUP_RE.finditer(text):
        group_symbols = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", group_match.group(1))
        if len(group_symbols) > 1:
            symbols.extend(group_symbols)
    symbols.extend(_normalize_stage_symbol(match.group(1)) for match in _STAGE_SCOPE_RE.finditer(text))
    stripped = text.strip().strip("`")
    leading_symbol = re.match(
        r"^(_[A-Za-z0-9_]+|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+|stage[\s_-]*\d+)\b",
        stripped,
        re.IGNORECASE,
    )
    if leading_symbol:
        symbol = leading_symbol.group(1)
        if re.match(r"^stage[\s_-]*\d+$", symbol, re.IGNORECASE):
            symbols.append(_normalize_stage_symbol(symbol))
        else:
            symbols.append(symbol.strip().lower())
    return symbols


def target_symbols_from_actionable_prose(value: str | None) -> list[str]:
    """Infer concrete target symbols from action-oriented task prose.

    This is narrower than scanning every backticked identifier in a task body:
    it only accepts lines that describe what to edit and skips lines that name
    reference implementations or docs to read.
    """
    symbols: list[str] = []
    for line in str(value or "").splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in _REFERENCE_TARGET_LINE_MARKERS):
            continue
        if not any(marker in lowered for marker in _ACTION_TARGET_LINE_MARKERS):
            continue
        for symbol in target_symbols_from_scoped_text(line):
            normalized = symbol.strip()
            if normalized and normalized not in symbols:
                symbols.append(normalized)
        for symbol in _BARE_ACTION_SYMBOL_RE.findall(line):
            normalized = symbol.strip()
            if normalized and normalized.lower() not in _ABSTRACT_ATOMIC_COMPONENT_SYMBOLS and normalized not in symbols:
                symbols.append(normalized)
    return symbols


def filter_abstract_target_symbols(symbols: list[str]) -> list[str]:
    """Drop abstract component labels from concrete patch target symbols."""
    filtered: list[str] = []
    for symbol in symbols:
        normalized = str(symbol or "").strip()
        if not normalized:
            continue
        if normalized.lower() in _ABSTRACT_ATOMIC_COMPONENT_SYMBOLS:
            continue
        if normalized not in filtered:
            filtered.append(normalized)
    return filtered


def split_patch_and_route_symbols(
    symbols: list[str],
    *,
    target_component: str | None = None,
) -> tuple[list[str], list[str]]:
    """Split concrete patch targets from execution-route proof symbols.

    Wrapper/shape-dispatch tasks often mention both the wrapper that should be
    edited and the Gluon kernel that proves the measured route. In that case the
    wrapper-like symbol is the patch target and the kernel-like symbol is route
    evidence, not a requirement to edit the kernel body.
    """
    unique = filter_abstract_target_symbols(symbols)
    component = str(target_component or "").strip().lower()
    if "wrapper" not in component and "shape_dispatch" not in component and "dispatch" not in component:
        return unique, []

    patch_targets = [
        symbol
        for symbol in unique
        if any(marker in symbol.lower() for marker in ("wrapper", "dispatch"))
    ]
    if not patch_targets:
        return unique, []
    route_symbols = [symbol for symbol in unique if symbol not in patch_targets]
    return patch_targets, route_symbols


def forbidden_symbols_from_scoped_text(value: str | None) -> list[str]:
    """Infer coarse forbidden-scope categories from task contract prose."""
    text = str(value or "").lower()
    symbols: list[str] = []
    if re.search(r"\b(?:whole|full|entire)\s+(?:kernel|stage|helper)\b", text):
        symbols.append("whole_kernel_rewrite")
    if re.search(r"\bhelper[-\s]*only\b|\b(?:add|define|create)\s+(?:an?\s+)?(?:unused\s+)?(?:gluon\s+)?helper\b", text):
        symbols.append("new_gluon_helper")
    if re.search(r"\b(?:main\s+)?dot\s+loop\b|\btl\.dot\b|\bmatrix\s+subpath\b", text):
        symbols.append("dot_loop")
    if re.search(r"\bmfma\b|\bdotoperandlayout\b|\bgl\.amd\.[a-z0-9_]+\.mfma\b", text):
        symbols.append("mfma_path")
    if re.search(r"\ba\s*/\s*b\s+loads?\b|\ba\s+and\s+b\s+loads?\b|\binput\s+loads?\b", text):
        symbols.append("ab_input_loads")
    if re.search(r"\bbuffer_load\b|\bbuffer_store\b", text):
        symbols.append("buffer_ops")
    if re.search(r"\bbias\s+load\b", text):
        symbols.append("bias_load")

    unique: list[str] = []
    for symbol in symbols:
        if symbol not in unique:
            unique.append(symbol)
    return unique


def do_not_clauses(value: str | None) -> list[str]:
    """Return standalone Do NOT clauses from a task body."""
    return [match.group(1).strip() for match in _DO_NOT_RE.finditer(str(value or ""))]


def _added_lines(patch_text: str) -> list[str]:
    return [line[1:].strip() for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++")]


def _patch_adds_gluon_jit(patch_text: str) -> bool:
    return bool(re.search(r"(?m)^\+@gluon\.jit\b", patch_text.lower()))


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


def _patch_rewrites_whole_kernel_or_helper(patch_text: str) -> bool:
    raw = patch_text.lower()
    if "-@triton.jit" in raw and "+@gluon.jit" in raw:
        return True
    for name in _extract_added_gluon_jit_defs(patch_text):
        lowered = name.lower()
        if "gluon" not in lowered and not lowered.endswith(("_g", "_gl")):
            return True
    return False


def patch_touches_forbidden_target_symbol(patch_text: str, forbidden_symbols: list[str]) -> str | None:
    """Return the first coarse forbidden target scope touched by a patch."""
    if not forbidden_symbols:
        return None
    added = "\n".join(_added_lines(patch_text)).lower()
    checks = {
        # Legacy metadata emitted before the split. Keep broad behavior only when
        # a task explicitly carries the legacy symbol.
        "whole_kernel_or_helper": lambda: _patch_adds_gluon_jit(patch_text),
        "whole_kernel_rewrite": lambda: _patch_rewrites_whole_kernel_or_helper(patch_text),
        "new_gluon_helper": lambda: _patch_adds_gluon_jit(patch_text),
        "dot_loop": lambda: bool(re.search(r"\b(?:gl\.dot|gl\.dot_fma|mfma)\s*\(", added)),
        "mfma_path": lambda: bool(re.search(r"\b(?:dotoperandlayout|amdmfmalayout|amdwmmalayout|mfma)\b", added)),
        "ab_input_loads": lambda: bool(re.search(r"\b[ab]\s*=\s*gl\.load\s*\(\s*[ab]_ptrs\b", added)),
        "buffer_ops": lambda: "buffer_load" in added or "buffer_store" in added,
        "bias_load": lambda: bool(re.search(r"\bbias\s*=\s*gl\.load\b|\bbias_ptr", added)),
    }
    for symbol in forbidden_symbols:
        check = checks.get(str(symbol).strip().lower())
        if check and check():
            return symbol
    return None
