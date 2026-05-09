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
