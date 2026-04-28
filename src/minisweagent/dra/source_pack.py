"""Build a local source pack for DRA.

The pack is the local, repo-grounded context DRA should understand before it
asks the web anything. It is intentionally not a generic codebase summary: it
contains the kernel entrypoint, local imports, task runner, extension loader,
and native source files that are likely to affect the benchmark.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import deque
from pathlib import Path


_TEXT_EXTS = {
    ".py",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".cu",
    ".cuh",
    ".hip",
    ".h",
    ".hpp",
    ".hh",
    ".cmake",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
}
_MAX_FILE_CHARS = 40_000
_MAX_TOTAL_CHARS = 240_000
_MAX_IMPORT_FILES = 20


def build_source_pack(
    kernel_path: Path,
    repo_root: Path,
    output_path: Path,
    *,
    profile_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
    benchmark_baseline_path: Path | None = None,
    commandment_path: Path | None = None,
    task_context: str | None = None,
) -> Path:
    """Write a Markdown source pack and return its path."""
    kernel_path = kernel_path.resolve()
    repo_root = repo_root.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, Path]] = []
    _add(files, "Kernel entrypoint", kernel_path)

    for p in _resolve_local_imports(kernel_path, repo_root):
        _add(files, "Local Python dependency", p)

    for rel in (
        "scripts/task_runner.py",
        "kernel_loader.py",
        "setup.py",
        "pyproject.toml",
        "CMakeLists.txt",
    ):
        _add(files, "Benchmark/build file", repo_root / rel)

    src_dir = repo_root / "src"
    if src_dir.exists():
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _TEXT_EXTS:
                _add(files, "Native/source file", p)

    lines = [
        "# DRA Source Pack",
        "",
        f"- Repo root: `{repo_root}`",
        f"- Kernel entrypoint: `{kernel_path}`",
        "",
        "This is the local source context. Web research questions must not ask the internet for facts available here.",
        "",
    ]
    task_context = (task_context or "").strip()
    if task_context:
        lines.extend(
            [
                "## User Task / Hardware Context",
                "",
                "Use this section as the authority for target hardware, workload, and user constraints.",
                "",
                "```text",
                task_context,
                "```",
                "",
            ]
        )
    hardware_context = _render_hardware_context()
    if hardware_context:
        lines.append(hardware_context)
        lines.append("")
    perf_context = _render_performance_context(
        profile_path=profile_path,
        baseline_metrics_path=baseline_metrics_path,
        benchmark_baseline_path=benchmark_baseline_path,
        commandment_path=commandment_path,
    )
    if perf_context:
        lines.append(perf_context)
        lines.append("")
    total = len("\n".join(lines))
    seen: set[Path] = set()
    for label, path in files:
        path = path.resolve()
        if path in seen or not _is_within(path, repo_root):
            continue
        seen.add(path)
        text = _read_text(path)
        if not text:
            continue
        rel = path.relative_to(repo_root)
        block = _render_file(label, rel, text)
        if total + len(block) > _MAX_TOTAL_CHARS:
            lines.append("\n## Truncation\n")
            lines.append(f"Stopped before `{rel}` to keep source pack under {_MAX_TOTAL_CHARS} chars.")
            break
        lines.append(block)
        total += len(block)

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _render_hardware_context() -> str:
    """Collect target hardware facts from AMD/ROCm CLI tools.

    This is intentionally dynamic run context, not DRA config. The goal is to
    tell DRA what hardware it is actually optimizing for before it writes web
    queries.
    """
    rocm_smi = _run_cmd(["rocm-smi", "--showproductname"])
    rocminfo = _run_cmd(["rocminfo"])
    amdgpu_arch = _run_cmd(["amdgpu-arch"])
    hipcc_version = _run_cmd(["hipcc", "--version"])

    gfx_versions = sorted(set(re.findall(r"GFX Version:\s*(gfx[0-9a-zA-Z]+)", rocm_smi)))
    if not gfx_versions:
        gfx_versions = sorted(set(re.findall(r"\b(gfx[0-9]{3,}[a-zA-Z0-9]*)\b", rocminfo + "\n" + amdgpu_arch)))

    card_series = sorted(set(_field_values(rocm_smi, "Card Series")))
    card_models = sorted(set(_field_values(rocm_smi, "Card Model")))
    marketing = sorted(set(_field_values(rocminfo, "Marketing Name")))
    marketing = [m for m in marketing if m and not m.startswith("AMD EPYC")]

    inferred: list[str] = []
    if any(m.lower() == "0x75a3" for m in card_models):
        inferred.append("PCI model 0x75a3 maps to AMD Instinct MI355X on this host family")
    if "gfx950" in gfx_versions:
        inferred.append("gfx950 is the ROCm target ISA; treat this as CDNA4-class hardware")

    if not any([gfx_versions, card_series, card_models, marketing, inferred, amdgpu_arch.strip()]):
        return ""

    lines = ["## Detected AMD/ROCm Hardware Context", ""]
    if card_series:
        lines.append(f"- Card series from `rocm-smi`: `{', '.join(card_series)}`")
    if marketing:
        lines.append(f"- Marketing name from `rocminfo`: `{', '.join(marketing)}`")
    if card_models:
        lines.append(f"- Card model IDs from `rocm-smi`: `{', '.join(card_models)}`")
    if gfx_versions:
        lines.append(f"- GFX targets: `{', '.join(gfx_versions)}`")
    if amdgpu_arch.strip():
        arch_lines = ", ".join(sorted(set(x.strip() for x in amdgpu_arch.splitlines() if x.strip())))
        lines.append(f"- `amdgpu-arch`: `{arch_lines}`")
    if inferred:
        lines.append("- Inferred target facts:")
        for item in inferred:
            lines.append(f"  - {item}")
    if hipcc_version.strip():
        version_line = next((line.strip() for line in hipcc_version.splitlines() if line.strip()), "")
        if version_line:
            lines.append(f"- `hipcc --version`: `{version_line[:180]}`")
    lines.append("")
    lines.append(
        "Hardware-specific web queries should use these detected target facts "
        "(for example gfx950/CDNA4/MI355X when present), not generic older CDNA targets."
    )
    return "\n".join(lines)


def _run_cmd(cmd: list[str], timeout_s: float = 8.0, max_chars: int = 20_000) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return text[:max_chars]


def _field_values(text: str, field: str) -> list[str]:
    vals: list[str] = []
    pattern = re.compile(rf"{re.escape(field)}:\s*(.+)")
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        val = match.group(1).strip()
        if val and val != "N/A" and "Error when calling" not in val:
            vals.append(val)
    return vals


def _render_performance_context(
    *,
    profile_path: Path | None,
    baseline_metrics_path: Path | None,
    benchmark_baseline_path: Path | None,
    commandment_path: Path | None,
) -> str:
    lines = ["## Local Performance Context", ""]
    added = False

    baseline = _read_json(baseline_metrics_path)
    if isinstance(baseline, dict):
        added = True
        lines.append("### Baseline metrics")
        lines.append(f"- Bottleneck: `{baseline.get('bottleneck', 'unknown')}`")
        if baseline.get("duration_us") is not None:
            lines.append(f"- Duration: `{baseline.get('duration_us')}` us")
        if baseline.get("benchmark_duration_us") is not None:
            lines.append(f"- Benchmark duration: `{baseline.get('benchmark_duration_us')}` us")
        top = baseline.get("top_kernels") or []
        if top:
            lines.append("- Top kernels:")
            for k in top[:8]:
                lines.append(
                    "  - "
                    f"`{k.get('name', '?')}`: {k.get('duration_us', '?')} us, "
                    f"{k.get('pct_of_total', '?')}%, bottleneck={k.get('bottleneck', '?')}"
                )
        metrics = baseline.get("metrics") or {}
        if metrics:
            interesting = {
                k: metrics.get(k)
                for k in (
                    "memory.hbm_bandwidth_utilization",
                    "memory.l2_hit_rate",
                    "duration_us_min",
                    "duration_us_max",
                )
                if k in metrics
            }
            if interesting:
                lines.append(f"- Key metrics: `{json.dumps(interesting, ensure_ascii=False)}`")
        lines.append("")

    profile = _read_json(profile_path)
    if isinstance(profile, dict):
        added = True
        kernels = []
        for result in profile.get("results") or []:
            kernels.extend(result.get("kernels") or [])
        if kernels:
            lines.append("### Profile kernels")
            for k in sorted(kernels, key=lambda x: float(x.get("duration_us") or 0), reverse=True)[:10]:
                metrics = k.get("metrics") or {}
                lines.append(
                    "- "
                    f"`{k.get('name', '?')}`: {k.get('duration_us', '?')} us, "
                    f"bottleneck={k.get('bottleneck', '?')}, "
                    f"l2_hit={metrics.get('memory.l2_hit_rate', '?')}, "
                    f"hbm_util={metrics.get('memory.hbm_bandwidth_utilization', '?')}"
                )
            lines.append("")

    benchmark = _read_text_if_exists(benchmark_baseline_path, max_chars=12_000)
    if benchmark:
        added = True
        lines.append("### Benchmark baseline output")
        lines.append("```text")
        lines.append(benchmark.rstrip())
        lines.append("```")
        lines.append("")

    commandment = _read_text_if_exists(commandment_path, max_chars=12_000)
    if commandment:
        added = True
        lines.append("### Correctness / evaluation contract")
        lines.append(commandment.rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() if added else ""


def _add(files: list[tuple[str, Path]], label: str, path: Path) -> None:
    if path.exists() and path.is_file():
        files.append((label, path))


def _read_json(path: Path | None) -> object | None:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text_if_exists(path: Path | None, max_chars: int) -> str:
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... [truncated, original {len(text)} chars] ..."
    return text


def _resolve_local_imports(kernel_path: Path, repo_root: Path) -> list[Path]:
    out: list[Path] = []
    queue: deque[Path] = deque([kernel_path])
    seen: set[Path] = set()
    while queue and len(out) < _MAX_IMPORT_FILES:
        path = queue.popleft().resolve()
        if path in seen or not _is_within(path, repo_root) or path.suffix != ".py":
            continue
        seen.add(path)
        for imported in _imports_from_python(path, repo_root):
            if imported not in seen and imported not in out:
                out.append(imported)
                queue.append(imported)
                if len(out) >= _MAX_IMPORT_FILES:
                    break
    return out


def _imports_from_python(path: Path, repo_root: Path) -> list[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    out: list[Path] = []
    for module in modules:
        p = _module_to_path(module, repo_root)
        if p:
            out.append(p)
    return out


def _module_to_path(module: str, repo_root: Path) -> Path | None:
    parts = module.split(".")
    candidates = [
        repo_root.joinpath(*parts).with_suffix(".py"),
        repo_root.joinpath(*parts, "__init__.py"),
    ]
    # Also support flat task-local imports like `import kernel_loader`.
    if len(parts) == 1:
        candidates.append(repo_root / f"{parts[0]}.py")
    for c in candidates:
        if c.exists() and c.is_file() and _is_within(c.resolve(), repo_root):
            return c.resolve()
    return None


def _read_text(path: Path) -> str:
    if path.suffix.lower() not in _TEXT_EXTS:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > _MAX_FILE_CHARS:
        return text[:_MAX_FILE_CHARS] + f"\n\n... [truncated, original {len(text)} chars] ..."
    return text


def _render_file(label: str, rel_path: Path, text: str) -> str:
    lang = _lang_for_suffix(rel_path.suffix)
    return f"## {label}: `{rel_path}`\n\n```{lang}\n{text.rstrip()}\n```\n"


def _lang_for_suffix(suffix: str) -> str:
    return {
        ".py": "python",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".cu": "cpp",
        ".cuh": "cpp",
        ".hip": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }.get(suffix.lower(), "")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
