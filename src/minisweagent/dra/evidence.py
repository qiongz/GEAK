"""Evidence layer for DRA.

This is the *only* module in `dra/` that knows about the on-disk shape of
preprocess artifacts (profile.json, baseline_metrics.json, ...). The current
DRA runtime uses it as an artifact reader only; open-search evidence is handled
in `iterative_search.py` / `web_search.py`.

Design intent:
  - Reading run artifacts is centralized so the runner cannot accidentally
    couple itself to filesystem layout.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs container
# ---------------------------------------------------------------------------


@dataclass
class DRAInputs:
    """Filesystem inputs to the DRA pipeline.

    Only `kernel_path` is strictly required; every other path is optional.
    Missing paths are silently skipped at read time.
    """

    kernel_path: Path
    output_dir: Path

    profile_path: Path | None = None
    baseline_metrics_path: Path | None = None
    discovery_path: Path | None = None
    source_pack_path: Path | None = None
    codebase_context_path: Path | None = None
    commandment_path: Path | None = None
    knowledge_base_path: Path | None = None

    previous_results_dir: Path | None = None
    previous_tasks_dir: Path | None = None
    round_evaluations: list[dict[str, Any]] = field(default_factory=list)
    current_round: int = 1

    def to_dict_for_artifact(self) -> dict[str, Any]:
        """A small dict suitable for embedding in the final artifact's `inputs`."""
        return {
            "kernel_path": str(self.kernel_path),
            "profile_path": str(self.profile_path) if self.profile_path else None,
            "baseline_metrics_path": (
                str(self.baseline_metrics_path) if self.baseline_metrics_path else None
            ),
            "discovery_path": str(self.discovery_path) if self.discovery_path else None,
            "source_pack_path": str(self.source_pack_path) if self.source_pack_path else None,
            "commandment_path": str(self.commandment_path) if self.commandment_path else None,
            "knowledge_base_path": (
                str(self.knowledge_base_path) if self.knowledge_base_path else None
            ),
            "current_round": self.current_round,
            "has_previous_results": bool(
                self.previous_results_dir and Path(self.previous_results_dir).is_dir()
            ),
            "has_previous_tasks": bool(
                self.previous_tasks_dir and Path(self.previous_tasks_dir).is_dir()
            ),
            "num_round_evaluations": len(self.round_evaluations),
        }


# ---------------------------------------------------------------------------
# Evidence source
# ---------------------------------------------------------------------------


# Cap each text payload so we never ship multi-MB blobs into a prompt context.
_MAX_TEXT_PAYLOAD = 200_000


def _read_text_capped(path: Path, max_chars: int = _MAX_TEXT_PAYLOAD) -> str:
    """Read a text file, truncating to `max_chars` with a sentinel marker."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + f"\n\n... [truncated, original {len(raw)} chars] ..."


def _read_json_safely(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse JSON at %s: %s", path, exc)
        return None


@dataclass
class SearchHit:
    """A normalized search result from `EvidenceSource.search`."""

    title: str
    content: str
    score: float
    source: str  # "embedding" | "bm25" | "embedding+bm25"
    layer: str = "unknown"
    category: str = "unknown"
    origin: str = "kb"  # "kb" | "prior_run" | "repo" (extensible)

    def to_evidence_ref(self) -> str:
        """Compact citation string suitable for the `evidence` field of an Answer."""
        return f"{self.origin}://{self.title} (score={self.score:.3f})"


class EvidenceSource:
    """Centralized read-side adapter for DRA.

    Responsibilities:
      - Read preprocess artifacts (kernel, profile, baseline, discovery, ...).
      - Keep legacy local search helpers available for old callers, but the
        active DRA flow does not call them.
      - Optionally surface prior-run experiences from cross_session memory.
    """

    def __init__(
        self,
        inputs: DRAInputs,
        retrieval_top_k: int = 8,
        index_path: str | None = None,
        use_prior_runs: bool = False,
    ):
        self.inputs = inputs
        self.retrieval_top_k = retrieval_top_k
        self.index_path = index_path
        self.use_prior_runs = use_prior_runs
        self._retriever = None  # lazy

    # ------------------------------------------------------------------
    # Artifact readers
    # ------------------------------------------------------------------

    def read_kernel(self) -> str:
        """The kernel source file, capped."""
        p = self.inputs.kernel_path
        if not p or not p.exists():
            logger.warning("Kernel file not found at %s", p)
            return ""
        return _read_text_capped(p)

    def read_profile(self) -> dict[str, Any] | None:
        p = self.inputs.profile_path
        if not p or not p.exists():
            return None
        data = _read_json_safely(p)
        return data if isinstance(data, dict) else None

    def read_baseline_metrics(self) -> dict[str, Any] | None:
        p = self.inputs.baseline_metrics_path
        if not p or not p.exists():
            return None
        data = _read_json_safely(p)
        return data if isinstance(data, dict) else None

    def read_discovery(self) -> dict[str, Any] | None:
        p = self.inputs.discovery_path
        if not p or not p.exists():
            return None
        data = _read_json_safely(p)
        return data if isinstance(data, dict) else None

    def read_codebase_context(self) -> str:
        p = self.inputs.codebase_context_path
        if not p or not p.exists():
            return ""
        return _read_text_capped(p)

    def read_source_pack(self) -> str:
        p = self.inputs.source_pack_path
        if not p or not p.exists():
            return self.read_kernel()
        return _read_text_capped(p)

    def read_commandment(self) -> str:
        p = self.inputs.commandment_path
        if not p or not p.exists():
            return ""
        return _read_text_capped(p)

    def summarize_previous_results(self) -> str:
        """Best-effort summary of prior round outputs, if available."""
        d = self.inputs.previous_results_dir
        if not d or not Path(d).is_dir():
            return ""
        lines = [f"## Previous results in {d}"]
        for child in sorted(Path(d).iterdir()):
            if child.is_file():
                lines.append(f"- {child.name} ({child.stat().st_size} bytes)")
        return "\n".join(lines)

    def round_evaluations_summary(self) -> str:
        """Compact view of the orchestrator's per-round evaluations."""
        if not self.inputs.round_evaluations:
            return ""
        chunks: list[str] = []
        for rev in self.inputs.round_evaluations:
            r = rev.get("round", "?")
            best_task = rev.get("best_task", "N/A")
            fb = rev.get("full_benchmark") or {}
            speedup = (
                fb.get("verified_speedup", "N/A")
                if isinstance(fb, dict) and fb
                else rev.get("benchmark_speedup", "N/A")
            )
            chunks.append(f"Round {r}: best_task={best_task}, speedup={speedup}x")
        return "\n".join(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @property
    def retriever(self):
        """Lazy-load the HybridRetriever, in-process (no MCP)."""
        if self._retriever is None:
            try:
                from rag_mcp.retrieval import DEFAULT_INDEX_PATH, HybridRetriever
            except ImportError as exc:
                raise RuntimeError(
                    "rag-mcp package not installed. "
                    "Install with: pip install -e mcp_tools/rag-mcp"
                ) from exc
            index = Path(self.index_path) if self.index_path else DEFAULT_INDEX_PATH
            if not index.exists():
                raise RuntimeError(
                    f"RAG index not found at {index}. "
                    "Build it with: python scripts/build_index.py --force"
                )
            self._retriever = HybridRetriever(index_path=index)
        return self._retriever

    def search(self, query: str, k: int | None = None) -> list[SearchHit]:
        """Hybrid search over the local knowledge base.

        Returns a list of normalized `SearchHit`s rather than the retriever's
        raw tuple format so callers in `runner.py` stay decoupled.
        """
        top_k = k if k is not None else self.retrieval_top_k
        try:
            raw = self.retriever.search(query, k=top_k)
        except Exception as exc:  # retrieval failures should not crash DRA
            logger.warning("RAG search failed for %r: %s", query, exc)
            return []

        hits: list[SearchHit] = []
        for doc, score, source, _orig_score in raw:
            md = getattr(doc, "metadata", {}) or {}
            title = md.get("section") or md.get("title") or md.get("source", "Unknown")
            hits.append(
                SearchHit(
                    title=str(title)[:120],
                    content=getattr(doc, "page_content", ""),
                    score=float(score),
                    source=source,
                    layer=str(md.get("layer", "unknown")),
                    category=str(md.get("category", "unknown")),
                    origin="kb",
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Optional: cross-session prior runs
    # ------------------------------------------------------------------

    def prior_run_context(self) -> str:
        """Pull a small block of prior-run context if cross_session is enabled."""
        if not self.use_prior_runs:
            return ""
        try:
            from minisweagent.memory.integration import assemble_memory_context  # type: ignore
        except ImportError:
            return ""

        try:
            bm = self.read_baseline_metrics() or {}
            ctx = assemble_memory_context(
                kernel_path=str(self.inputs.kernel_path),
                bottleneck_type=bm.get("bottleneck"),
                profiling_metrics=bm,
            )
        except Exception as exc:
            logger.warning("prior_run_context failed: %s", exc)
            return ""
        return ctx or ""


# ---------------------------------------------------------------------------
# Source-pack-aware local code selector
#
# The synthesizer kept admitting "I cannot see the actual source code, so I'm
# guessing from PointNet++ folklore" even though the source pack already
# contained ``three_nn_cuda.hip`` verbatim. The fix is to slice the source
# pack into per-file blocks and let the synthesizer re-receive only the
# blocks that look relevant to the question being answered.
#
# This is intentionally not a real RAG retriever -- no embeddings, no index,
# no extra LLM calls. Just keyword/identifier overlap, computed in-process.
# ---------------------------------------------------------------------------


@dataclass
class SourcePackChunk:
    """One file's block from the source pack, ready to score and ship."""

    label: str  # e.g. "Kernel entrypoint", "Native/source file"
    rel_path: str  # e.g. "src/three_nn_cuda.hip"
    body: str  # the fenced code body (without the ``` fences)
    lang: str  # e.g. "cpp", "python"

    @property
    def header(self) -> str:
        return f"## {self.label}: `{self.rel_path}`"

    def render(self, max_chars: int) -> str:
        body = self.body
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n... [truncated, original {len(self.body)} chars] ..."
        return f"{self.header}\n\n```{self.lang}\n{body}\n```"


# Matches the file blocks emitted by source_pack._render_file:
#   ## <Label>: `<rel_path>`
#
#   ```<lang>
#   ...body...
#   ```
_FILE_BLOCK_RE = re.compile(
    r"^## (?P<label>[^:\n]+): `(?P<path>[^`]+)`\s*\n+```(?P<lang>[a-zA-Z0-9_+-]*)\n"
    r"(?P<body>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


def parse_source_pack_chunks(source_pack_text: str) -> list[SourcePackChunk]:
    """Slice the source pack Markdown into one chunk per file block.

    Tolerant: if the regex matches nothing (custom source pack format), the
    caller falls back to passing the whole pack truncated.
    """
    out: list[SourcePackChunk] = []
    for m in _FILE_BLOCK_RE.finditer(source_pack_text or ""):
        out.append(
            SourcePackChunk(
                label=m.group("label").strip(),
                rel_path=m.group("path").strip(),
                body=m.group("body"),
                lang=m.group("lang") or "",
            )
        )
    return out


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# Identifier-shaped tokens (snake_case, camelCase, or with leading
# underscores) are much more discriminating than plain English. We boost
# matches on these because in this domain a token like ``three_nn_kernel``
# or ``__shfl_xor`` uniquely identifies the right chunk in a way that
# ``loop`` or ``block`` never could.
_IDENTIFIER_RE = re.compile(r"^(?:_+\w+|[A-Za-z]+_[\w_]+|[a-z][a-z0-9]+[A-Z]\w*)$")

# Stopwords that are too generic to anchor on; dropping them prevents every
# question with the word "kernel" from matching every chunk identically.
# These are deliberately aggressive: the goal is to leave only the words
# that actually distinguish one source-pack chunk from another.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "from", "are", "what",
        "how", "why", "when", "which", "does", "can", "any", "into", "out",
        "kernel", "function", "code", "source", "file", "files", "data",
        "value", "values", "type", "types", "use", "used", "uses", "using",
        "would", "should", "could", "have", "has", "had", "is", "be", "we",
        "it", "its", "their", "there", "than", "then", "between", "across",
        "given", "based", "without", "much", "many", "some", "all", "one",
        "two", "three", "first", "second", "next", "rather", "instead",
        "more", "less", "best", "good", "common", "general", "specific",
        "exact", "actual", "current", "default", "different", "same", "small",
        "large", "high", "low", "main", "key", "primary", "single", "multi",
        "GPU", "CPU", "AMD", "ROCm", "HIP", "CUDA",
        # Common English noise that leaks from question docstrings into the
        # wrapper file's docstring -- the same words appear in both, so
        # they're useless for distinguishing chunks.
        "find", "found", "set", "sets", "point", "points", "neighbor",
        "neighbors", "shape", "shapes", "input", "output", "outputs",
        "tensor", "tensors", "args", "returns", "return", "top", "near",
        "nearest", "distance", "distances", "where", "must", "args",
    }
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text or "") if t not in _STOPWORDS}


def _identifier_tokens(token_set: set[str]) -> set[str]:
    return {t for t in token_set if _IDENTIFIER_RE.match(t)}


# Filter out comment / docstring lines before scoring chunk bodies. A match
# inside ``// ... three nearest neighbors ...`` is much weaker evidence
# that this is the right chunk than a match on the actual code line
# ``__global__ void three_nn_kernel(...)``.
_COMMENT_LINE_RE = re.compile(
    r"""
    ^\s*(?:
        \#.* |              # python / shell comments
        //.* |              # cpp single-line
        /\*.*?\*/ |         # cpp inline block (rare)
        \*.* |              # cpp continuation of a /* */ block
        \"\"\".* |          # python triple-double opening
        '''.*               # python triple-single opening
    )$
    """,
    re.MULTILINE | re.VERBOSE,
)


def _strip_comments(body: str) -> str:
    """Best-effort strip of comment / docstring lines for scoring.

    Not meant to be a real lexer -- just enough to stop docstrings in the
    Python wrapper from outscoring real code in the .hip kernel file.
    """
    return _COMMENT_LINE_RE.sub("", body)


def select_local_code_chunks(
    source_pack_text: str,
    question: str,
    *,
    affected: list[str] | None = None,
    extra_terms: list[str] | None = None,
    max_chunks: int = 3,
    max_chars_per_chunk: int = 8000,
    total_char_budget: int = 16000,
) -> str:
    """Return Markdown-rendered local code chunks most relevant to ``question``.

    Scoring is keyword/identifier overlap between the question (plus any
    ``affected`` / ``extra_terms`` hints) and each chunk's path + code body
    (comments / docstrings stripped). Identifier-shaped tokens (snake_case,
    camelCase, dunder) are weighted heavily because they uniquely identify
    code in this domain. Ties break by ``label`` priority (kernel
    entrypoint > native source > Python deps > build files) and then by
    chunk size (smaller first). Returns "" if nothing parses or scores.
    """
    chunks = parse_source_pack_chunks(source_pack_text)
    if not chunks:
        return ""

    query_tokens = _tokens(question)
    for hint in affected or []:
        query_tokens |= _tokens(hint)
    for hint in extra_terms or []:
        query_tokens |= _tokens(hint)
    if not query_tokens:
        return ""
    query_idents = _identifier_tokens(query_tokens)

    label_priority = {
        "Kernel entrypoint": 3,
        "Native/source file": 2,
        "Local Python dependency": 1,
        "Benchmark/build file": 0,
    }

    scored: list[tuple[float, int, SourcePackChunk]] = []
    for ch in chunks:
        code_only = _strip_comments(ch.body)
        body_tokens = _tokens(code_only)
        body_idents = _identifier_tokens(body_tokens)
        path_tokens = _tokens(ch.rel_path)

        # Identifier overlap is the single strongest signal; weight it 3x.
        # Path overlap also matters because a question naming a function
        # almost always wants the file containing it. Plain word overlap
        # gets a small contribution as a tiebreaker.
        ident_overlap = len(query_idents & body_idents)
        path_overlap = len(query_tokens & path_tokens)
        word_overlap = len(query_tokens & body_tokens)
        score = 3.0 * ident_overlap + 2.0 * path_overlap + 0.25 * word_overlap

        if score <= 0:
            continue
        prio = label_priority.get(ch.label, 0)
        # Sort key: -score, then -priority, then chunk length (smaller wins on tie).
        scored.append((-(score + 0.1 * prio), len(ch.body), ch))

    if not scored:
        return ""

    scored.sort()
    rendered: list[str] = []
    used_chars = 0
    for _, _, ch in scored[:max_chunks]:
        block = ch.render(max_chars=max_chars_per_chunk)
        if used_chars + len(block) > total_char_budget:
            break
        rendered.append(block)
        used_chars += len(block)

    return "\n\n".join(rendered)
