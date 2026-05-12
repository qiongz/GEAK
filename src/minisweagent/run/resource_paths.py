"""Helpers for locating GEAK project resources across source and installs."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path

from minisweagent import get_repo_root, package_dir


def _candidate_roots(workspace: Path | None = None, extra_roots: Iterable[Path | str] = ()) -> list[Path]:
    roots: list[Path] = []

    def add(root: Path | str | None) -> None:
        if root is None:
            return
        try:
            path = Path(root).expanduser().resolve()
        except (OSError, RuntimeError):
            return
        if path not in roots:
            roots.append(path)

    for root in extra_roots:
        add(root)
    if workspace is not None:
        try:
            current = Path(workspace).expanduser().resolve()
            for root in (current, current.parent, current.parent.parent):
                add(root)
        except (OSError, RuntimeError):
            pass
    add(os.environ.get("GEAK_ROOT"))
    add(Path("/workspace"))
    add(get_repo_root())
    add(package_dir.parent.parent)
    add(Path(sys.prefix) / "share" / "geak")
    add(Path(sys.prefix) / "local" / "share" / "geak")
    return roots


def resolve_project_resource(
    relative_path: str,
    *,
    workspace: Path | str | None = None,
    extra_roots: Iterable[Path | str] = (),
) -> Path | None:
    """Resolve a repository resource from source trees or installed data files."""
    rel = Path(str(relative_path).strip())
    if not str(rel):
        return None
    if rel.is_absolute():
        return rel.resolve() if rel.exists() else None
    workspace_path = Path(workspace).expanduser() if workspace is not None else None
    for root in _candidate_roots(workspace_path, extra_roots):
        candidate = root / rel
        if candidate.exists():
            return candidate.resolve()
    return None

