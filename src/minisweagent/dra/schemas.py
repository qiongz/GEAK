"""Data schemas for DRA (Deep Research Artifact) generation.

Two internal artifact families feed Markdown rendering:
  - DeepSearchArtifact: convergent, source-pack-grounded research output
  - ExperimentalDirectionsArtifact: orthogonal, exploratory probes

Intermediate types (Question, Answer, BlindSpot) are passed between stages
of the runner and are not persisted as standalone artifacts. The task generator
consumes Markdown; JSONL traces are kept only for audit/debugging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Literal


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Recommended status that the task generator can act on directly.
# These are the only signals the downstream prompt needs to interpret.
AnswerStatus = Literal["prefer", "deprioritize", "reject", "open"]


# ---------------------------------------------------------------------------
# Stage 0 output: structured facts extracted from inputs
# ---------------------------------------------------------------------------


@dataclass
class Facts:
    """Compact structured view of the run context (Stage 0 output)."""

    kernel_language: str = "unknown"
    kernel_backend: str = "unknown"
    bottleneck_type: str = "unknown"
    hot_kernels: list[dict[str, Any]] = field(default_factory=list)
    benchmark_contract: str = ""
    correctness_constraints: list[str] = field(default_factory=list)
    prior_successes: list[str] = field(default_factory=list)
    prior_failures: list[str] = field(default_factory=list)
    likely_targets: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage 1/2 outputs: questions
# ---------------------------------------------------------------------------


@dataclass
class Question:
    """A candidate research question (Stage 1) with optional ranking (Stage 2).

    ``needs_web`` is the routing signal between two synthesis paths:
      - ``True``  -> Stage 3+4 runs full open-search + read + synth. Used for
                    questions that genuinely require external knowledge (papers,
                    other implementations, hardware specs).
      - ``False`` -> Stage 3+4 skips the web entirely and runs a fast local-only
                    synthesis using just the source pack + facts. Used for
                    introspection questions whose answers are already in the
                    repository (e.g. "what data layout does the wrapper use?").

    Defaults to ``True`` for back-compat with old payloads.
    """

    question: str
    search_queries: list[str] = field(default_factory=list)
    rationale: str = ""
    decision_impact: int = 0  # 0-10
    actionability: int = 0  # 0-10
    kernel_relevance: int = 0  # 0-10
    rank_score: float = 0.0  # composite ranking score (computed in Stage 2)
    needs_web: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage 4 output: per-question structured answer
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCite:
    """A single cited source backing an answer claim.

    Replaces the old free-form ``evidence: list[str]`` so that downstream
    consumers (task generator, RAG-write step, future de-dup) can reason about
    where each claim actually came from.
    """

    source_type: str  # "kb" | "web_search" | "web_arxiv" | "web_github" | "web_rocm_docs" | "web_hn" | "facts" | "prior_run"
    title: str = ""
    url: str = ""  # populated for web_* origins; empty for KB chunks
    chunk_id: str = ""  # populated for KB chunks; empty for web_* origins
    snippet: str = ""  # short excerpt (<= ~400 chars)
    score: float = 0.0  # retriever / search score (origin-dependent scale)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """Structured per-question synthesis (Stage 4 / Stage 6 output)."""

    question: str
    answer: str
    # Structured citations replacing the old free-form list[str]. Old runs
    # produced strings like "kb://Section (score=0.5)" -- new runs produce
    # EvidenceCite objects with source_type, url/chunk_id, snippet, score.
    evidence: list[EvidenceCite] = field(default_factory=list)
    affected: list[str] = field(default_factory=list)
    taskgen_implications: str = ""
    status: AnswerStatus = "open"
    source_stage: str = "first_pass"  # "first_pass" | "second_pass" | "second_pass_round_N"
    refinement_history: list[str] = field(default_factory=list)  # query refinement chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": [e.to_dict() for e in self.evidence],
            "affected": self.affected,
            "taskgen_implications": self.taskgen_implications,
            "status": self.status,
            "source_stage": self.source_stage,
            "refinement_history": self.refinement_history,
        }


# ---------------------------------------------------------------------------
# Stage 5 output: blindspot critique
# ---------------------------------------------------------------------------


@dataclass
class BlindSpot:
    """A weakly-supported assumption or unexplored area surfaced by the critique."""

    description: str
    why_it_matters: str = ""
    follow_up_question: str = ""
    round: int = 1  # which blindspot round produced this (1, 2, 3, ...)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage 7 final artifacts
# ---------------------------------------------------------------------------


@dataclass
class TaskgenGuidance:
    """The structured guidance that the task generator consumes."""

    prefer_first: list[str] = field(default_factory=list)
    deprioritize: list[str] = field(default_factory=list)
    reject: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeepSearchArtifact:
    """Final convergent research artifact rendered to deep_search.md."""

    timestamp: str = field(default_factory=_now_iso)
    inputs: dict[str, Any] = field(default_factory=dict)
    facts: Facts = field(default_factory=Facts)
    questions: list[Question] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)
    blindspots: list[BlindSpot] = field(default_factory=list)
    ranked_hypotheses: list[str] = field(default_factory=list)
    taskgen_guidance: TaskgenGuidance = field(default_factory=TaskgenGuidance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "facts": self.facts.to_dict(),
            "questions": [q.to_dict() for q in self.questions],
            "answers": [a.to_dict() for a in self.answers],
            "blindspots": [b.to_dict() for b in self.blindspots],
            "ranked_hypotheses": self.ranked_hypotheses,
            "taskgen_guidance": self.taskgen_guidance.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


@dataclass
class ExperimentalDirection:
    """One orthogonal probe rendered into experimental_directions.md."""

    direction_id: str
    thesis: str
    why_orthogonal: str = ""
    assumption_challenged: str = ""
    strategy_family: str = ""
    target_files_or_functions: list[str] = field(default_factory=list)
    expected_upside: str = ""
    implementation_cost: str = ""
    kill_criteria: str = ""
    notes_for_taskgen: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentalDirection:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ExperimentalDirectionsArtifact:
    """Final orthogonal artifact rendered to experimental_directions.md."""

    timestamp: str = field(default_factory=_now_iso)
    inputs: dict[str, Any] = field(default_factory=dict)
    directions: list[ExperimentalDirection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "directions": [d.to_dict() for d in self.directions],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)
