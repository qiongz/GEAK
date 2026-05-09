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
    return symbols


def forbidden_symbols_from_scoped_text(value: str | None) -> list[str]:
    """Infer coarse forbidden-scope categories from task contract prose."""
    text = str(value or "").lower()
    symbols: list[str] = []
    if re.search(r"\b(?:whole|full|entire)\s+(?:kernel|stage|helper)\b", text):
        symbols.append("whole_kernel_or_helper")
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
