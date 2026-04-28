"""DRA runner: the Stage 0-7 loop with iterative web evidence.

Public entrypoint is ``run_dra(...)``, which:

  0. Stage 0: extracts structured facts from inputs
  1+2. Stages 1+2: generates and ranks research questions (one LLM call)
  3+4. Stages 3+4: per question runs iterative open-search -> refine search
       (parallelised across questions) and synthesizes one optimization judgment
       per question
  5+6. Stages 5+6: multi-round blindspot loop. Each round critiques all
       answers gathered so far, surfaces new (deduped) blindspots, and runs
       a second search+synth pass on each. Early-stops when a round yields
       zero new blindspots.
  7. Stage 7: composes deep_search.{md,json}
  (optional) Experimental: produces experimental_directions.{md,json}

All web research reads now flow through ``iterative_search`` so the synthesizer
gets open-search result metadata plus selected page bodies for technical context.

A global ``call_budget`` counts every LLM call AND every web call against
``cfg.max_total_calls``. When exceeded, the runner aborts the current stage
gracefully and proceeds to Stage 7 with whatever it has.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minisweagent.dra import prompts
from minisweagent.dra.config import DRAConfig
from minisweagent.dra.evidence import DRAInputs, EvidenceSource, select_local_code_chunks
from minisweagent.dra.iterative_search import SearchResult, UnifiedHit, iterative_search
from minisweagent.dra.mcp_fetch import McpFetchClient
from minisweagent.dra.mcp_fetch import aclose_all as _close_mcp
from minisweagent.dra.schemas import (
    Answer,
    BlindSpot,
    DeepSearchArtifact,
    EvidenceCite,
    ExperimentalDirection,
    ExperimentalDirectionsArtifact,
    Facts,
    Question,
    TaskgenGuidance,
)
from minisweagent.dra.web_search import WebSearchSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost / budget bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class CallBudget:
    """Global counter shared across all stages of a single DRA invocation."""

    max_total: int
    llm_calls: int = 0
    web_calls: int = 0  # search + refinement combined
    aborted: bool = False
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.llm_calls + self.web_calls

    def remaining(self) -> int:
        return max(0, self.max_total - self.total)

    def record(self, kind: str) -> None:
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        if kind == "llm":
            self.llm_calls += 1
        else:
            self.web_calls += 1
        if self.total >= self.max_total:
            self.aborted = True

    def web_sink(self):
        """Returns a callback compatible with ``iterative_search.on_call``."""

        def _sink(kind: str) -> None:
            # Search and query-refinement calls count toward web budget.
            self.record("web")

        return _sink

    def summary(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "web_calls": self.web_calls,
            "total": self.total,
            "max_total": self.max_total,
            "aborted_on_budget": self.aborted,
            **{f"kind:{k}": v for k, v in self.by_kind.items()},
        }


@dataclass
class RunStats:
    """Per-run telemetry surfaced in the preprocess success log."""

    questions_asked: int = 0
    blindspot_rounds_run: int = 0
    blindspots_total: int = 0
    answers_total: int = 0
    web_fetches_total: int = 0
    refinements_total: int = 0
    aborted_on_budget: bool = False

    def merge_search_result(self, sr: SearchResult) -> None:
        self.web_fetches_total += sr.web_fetch_count
        self.refinements_total += sr.refinement_count


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_dra(
    inputs: DRAInputs,
    config: DRAConfig | None = None,
    model: Any | None = None,
) -> dict[str, Path]:
    """Run the full DRA pipeline and write artifacts to disk.

    Args:
        inputs:  Filesystem inputs (kernel, profile, baseline, ...).
        config:  DRAConfig; defaults to ``DRAConfig.from_env()``.
        model:   An object with ``.query(messages) -> {"content": str}``. If
                 omitted, an ``AmdLlmModel`` is constructed from the config.

    Returns dict of paths to the written artifacts. Always includes
    ``deep_search_md``. Includes the experimental Markdown artifact if
    ``config.run_experimental``. Query/result traces are written as JSONL for
    audit/debugging.

    Sync entrypoint: internally drives an asyncio loop because the evidence
    layer fans out across multiple open-search calls per question.
    """
    cfg = config or DRAConfig.from_env()
    if not cfg.enabled:
        raise RuntimeError("DRA is disabled (GEAK_DRA_DISABLE=1)")

    inputs.output_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        from minisweagent.models.amd_llm import AmdLlmModel

        model = AmdLlmModel(model_name=cfg.model_name, api_key=cfg.api_key)
        if hasattr(model, "_impl") and hasattr(model._impl, "tools"):
            model._impl.tools = []

    return asyncio.run(_run_dra_async(inputs, cfg, model))


async def _run_dra_async(
    inputs: DRAInputs,
    cfg: DRAConfig,
    model: Any,
) -> dict[str, Path]:
    evidence = EvidenceSource(
        inputs=inputs,
        retrieval_top_k=cfg.retrieval_top_k,
        index_path=cfg.index_path,
        use_prior_runs=cfg.use_prior_runs,
    )

    web = (
        WebSearchSource(
            github_token=cfg.github_token,
            open_websearch_url=cfg.open_websearch_url,
            enable_open_websearch=cfg.open_websearch_enabled,
            enable_arxiv=cfg.supplemental_web_enabled,
            enable_github=cfg.supplemental_web_enabled,
            enable_hn=cfg.supplemental_web_enabled,
            enable_rocm_docs=cfg.supplemental_web_enabled,
        )
        if cfg.web_search_enabled
        else None
    )
    fetch_client = (
        McpFetchClient.get_or_create(cfg.mcp_fetch_url) if cfg.web_search_enabled else None
    )

    budget = CallBudget(max_total=cfg.max_total_calls)
    stats = RunStats()
    search_records: list[dict[str, Any]] = []
    read_records: list[dict[str, Any]] = []

    try:
        logger.info("[DRA] Stage 0: extracting facts")
        facts = await asyncio.to_thread(_stage0_extract_facts, model, evidence, budget)
        source_pack_text = evidence.read_source_pack()

        logger.info(
            "[DRA] Stages 1+2: generating and ranking questions (max=%d, deep_mode=%s)",
            cfg.max_questions,
            cfg.deep_mode,
        )
        questions = await asyncio.to_thread(
            _stage12_generate_and_rank_questions, model, facts, source_pack_text, cfg.max_questions, budget
        )
        stats.questions_asked = len(questions)
        logger.info("[DRA] Selected %d ranked questions", len(questions))

        logger.info(
            "[DRA] Stages 3+4: per-question open-search research + synthesis "
            "(concurrency=%d result_budget=%d)",
            cfg.web_concurrency,
            cfg.per_question_result_budget,
        )
        first_pass_answers, fp_search_results = await _stage34_first_evidence_pass(
            model=model,
            evidence=evidence,
            web=web,
            fetch_client=fetch_client,
            facts=facts,
            questions=questions,
            cfg=cfg,
            budget=budget,
        )
        for sr in fp_search_results:
            stats.merge_search_result(sr)
        for q, sr in zip(questions, fp_search_results):
            search_records.append(_search_record(q.question, "first_pass", sr))
            read_records.extend(_read_records_from_search_result(q.question, "first_pass", sr))

        # ---- Stage 5+6: multi-round blindspot loop ----
        all_answers: list[Answer] = list(first_pass_answers)
        all_blindspots: list[BlindSpot] = []
        for round_idx in range(1, cfg.max_blindspot_rounds + 1):
            if budget.aborted:
                logger.warning("[DRA] Budget exceeded; skipping remaining blindspot rounds")
                break
            logger.info(
                "[DRA] Stage 5 round %d/%d: blindspot critique",
                round_idx,
                cfg.max_blindspot_rounds,
            )
            new_blindspots = await asyncio.to_thread(
                _stage5_blindspot,
                model,
                facts,
                all_answers,
                all_blindspots,
                cfg.max_blindspots,
                round_idx,
                cfg.max_blindspot_rounds,
                budget,
            )
            new_blindspots = _dedup_blindspots(new_blindspots, all_blindspots)
            for b in new_blindspots:
                b.round = round_idx
            logger.info("[DRA] Round %d: %d new (deduped) blindspots", round_idx, len(new_blindspots))
            stats.blindspot_rounds_run = round_idx
            stats.blindspots_total += len(new_blindspots)
            if not new_blindspots:
                logger.info("[DRA] Round %d produced no new blindspots; early-stop", round_idx)
                break
            all_blindspots.extend(new_blindspots)

            logger.info("[DRA] Stage 6 round %d: targeted second-pass research", round_idx)
            second_pass, sp_search_results = await _stage6_second_pass(
                model=model,
                evidence=evidence,
                web=web,
                fetch_client=fetch_client,
                facts=facts,
                blindspots=new_blindspots,
                cfg=cfg,
                budget=budget,
                round_idx=round_idx,
            )
            for sr in sp_search_results:
                stats.merge_search_result(sr)
            for ans, sr in zip(second_pass, sp_search_results):
                search_records.append(_search_record(ans.question, f"second_pass_round_{round_idx}", sr))
                read_records.extend(
                    _read_records_from_search_result(ans.question, f"second_pass_round_{round_idx}", sr)
                )
            all_answers.extend(second_pass)

            if budget.aborted:
                logger.warning("[DRA] Budget exceeded after Stage 6 round %d", round_idx)
                break

        stats.answers_total = len(all_answers)
        stats.aborted_on_budget = budget.aborted

        logger.info("[DRA] Stage 7: final synthesis")
        artifact = await asyncio.to_thread(
            _stage7_final_synthesis,
            model=model,
            inputs=inputs,
            facts=facts,
            questions=questions,
            answers=all_answers,
            blindspots=all_blindspots,
            budget=budget,
        )

        written = _write_deep_search(artifact, inputs.output_dir)
        synth_path = _write_synthesis_record(artifact, inputs.output_dir, stats, budget)
        search_path = _write_search_records(search_records, inputs.output_dir, artifact.timestamp)
        read_path = _write_read_records(read_records, inputs.output_dir, artifact.timestamp)
        written.update({"synth_records": synth_path, "search_records": search_path, "read_records": read_path})
        logger.info("[DRA] Wrote %s", written["deep_search_md"])
        logger.info("[DRA] Wrote query/result trace %s", search_path)
        logger.info("[DRA] Wrote read trace %s", read_path)

        if cfg.run_experimental and not budget.aborted:
            logger.info("[DRA] Running experimental_directions pass")
            ed_artifact = await asyncio.to_thread(
                _experimental_pass, model, inputs, facts, artifact, budget
            )
            ed_written = _write_experimental(ed_artifact, inputs.output_dir)
            written.update(ed_written)
            logger.info(
                "[DRA] Wrote %s",
                ed_written["experimental_md"],
            )

        logger.info(
            "[DRA] Run complete. questions=%d blindspot_rounds=%d answers=%d "
            "page_reads=%d refinements=%d llm_calls=%d total_calls=%d/%d aborted=%s",
            stats.questions_asked,
            stats.blindspot_rounds_run,
            stats.answers_total,
            stats.web_fetches_total,
            stats.refinements_total,
            budget.llm_calls,
            budget.total,
            budget.max_total,
            stats.aborted_on_budget,
        )
        return written
    finally:
        with contextlib.suppress(Exception):
            await _close_mcp()


# ---------------------------------------------------------------------------
# Stage 0: Fact extraction
# ---------------------------------------------------------------------------


def _stage0_extract_facts(model: Any, evidence: EvidenceSource, budget: CallBudget) -> Facts:
    prompt = prompts.FACTS_PROMPT.format(
        source_pack=evidence.read_source_pack(),
        previous_results=evidence.summarize_previous_results() or "(empty)",
        round_evaluations=evidence.round_evaluations_summary() or "(empty)",
    )
    payload = _llm_json(model, prompt, fallback={}, budget=budget)
    return _facts_from_payload(payload)


# ---------------------------------------------------------------------------
# Stages 1+2: Question generation and ranking
# ---------------------------------------------------------------------------


def _stage12_generate_and_rank_questions(
    model: Any, facts: Facts, source_pack: str, max_questions: int, budget: CallBudget
) -> list[Question]:
    prompt = prompts.QUESTIONS_PROMPT.format(
        facts_json=_json(facts.to_dict()),
        source_pack=source_pack,
        max_questions=max_questions,
    )
    payload = _llm_json(model, prompt, fallback={"questions": []}, budget=budget)

    raw = payload.get("questions") or []
    questions: list[Question] = []
    for q in raw:
        if not isinstance(q, dict) or "question" not in q:
            continue
        questions.append(
            Question(
                question=str(q.get("question", "")).strip(),
                search_queries=_str_list(q.get("search_queries"))[:4],
                rationale=str(q.get("rationale", "")).strip(),
                decision_impact=int(q.get("decision_impact", 0) or 0),
                actionability=int(q.get("actionability", 0) or 0),
                kernel_relevance=int(q.get("kernel_relevance", 0) or 0),
                rank_score=float(q.get("rank_score", 0.0) or 0.0),
            )
        )
    questions = [q for q in questions if q.question]
    questions.sort(key=lambda q: q.rank_score, reverse=True)
    return questions[:max_questions]


# ---------------------------------------------------------------------------
# Stages 3+4: First-pass web research + per-question synthesis (parallelised)
# ---------------------------------------------------------------------------


async def _stage34_first_evidence_pass(
    *,
    model: Any,
    evidence: EvidenceSource,
    web: WebSearchSource | None,
    fetch_client: McpFetchClient | None,
    facts: Facts,
    questions: list[Question],
    cfg: DRAConfig,
    budget: CallBudget,
) -> tuple[list[Answer], list[SearchResult]]:
    if not questions:
        return [], []

    prior_run_ctx = await asyncio.to_thread(evidence.prior_run_context)
    source_pack_text = await asyncio.to_thread(evidence.read_source_pack)
    sem = asyncio.Semaphore(max(1, cfg.web_concurrency))

    async def _do(q: Question, idx: int) -> tuple[Answer, SearchResult]:
        async with sem:
            if budget.aborted:
                return _placeholder_answer(q.question, "first_pass"), SearchResult(hits=[], tried_queries=[])
            sr = await iterative_search(
                question=q.question,
                initial_queries=q.search_queries,
                evidence=evidence,
                web=web,
                fetch_client=fetch_client,
                top_k=cfg.retrieval_top_k,
                min_quality=cfg.min_search_results_per_question,
                max_refinements=cfg.web_max_refinements,
                search_depth=cfg.search_depth,
                per_question_result_budget=cfg.per_question_result_budget,
                read_top_k=cfg.read_top_k,
                read_max_length=cfg.read_max_length,
                refine_query_fn=_make_refine_fn(model, budget),
                triage_results_fn=_make_triage_fn(model, budget),
                on_call=budget.web_sink(),
            )
            ans = await asyncio.to_thread(
                _synthesize_one,
                model=model,
                facts=facts,
                question=q.question,
                hits=sr.hits,
                prior_run_ctx=prior_run_ctx,
                source_stage="first_pass",
                refinement_history=sr.tried_queries,
                cfg=cfg,
                budget=budget,
                source_pack_text=source_pack_text,
                affected_hint=None,
            )
            return ans, sr

    pairs = await asyncio.gather(*(_do(q, i) for i, q in enumerate(questions)))
    answers = [p[0] for p in pairs]
    srs = [p[1] for p in pairs]
    return answers, srs


# ---------------------------------------------------------------------------
# Stage 5: Blindspot critique (one round)
# ---------------------------------------------------------------------------


def _stage5_blindspot(
    model: Any,
    facts: Facts,
    answers: list[Answer],
    prior_blindspots: list[BlindSpot],
    max_blindspots: int,
    round_idx: int,
    max_rounds: int,
    budget: CallBudget,
) -> list[BlindSpot]:
    if budget.aborted:
        return []
    prompt = prompts.BLINDSPOT_PROMPT.format(
        round_idx=round_idx,
        max_rounds=max_rounds,
        facts_json=_json(facts.to_dict()),
        answers_json=_json([a.to_dict() for a in answers]),
        max_blindspots=max_blindspots,
        prior_blindspots_json=_json([b.to_dict() for b in prior_blindspots]),
    )
    payload = _llm_json(model, prompt, fallback={"blindspots": []}, budget=budget)
    raw = payload.get("blindspots") or []
    out: list[BlindSpot] = []
    for b in raw[:max_blindspots]:
        if not isinstance(b, dict):
            continue
        desc = str(b.get("description", "")).strip()
        if not desc:
            continue
        out.append(
            BlindSpot(
                description=desc,
                why_it_matters=str(b.get("why_it_matters", "")).strip(),
                follow_up_question=str(b.get("follow_up_question", "")).strip(),
                round=round_idx,
            )
        )
    return out


def _dedup_blindspots(new: list[BlindSpot], existing: list[BlindSpot]) -> list[BlindSpot]:
    """Coarse semantic dedup of new blindspots against the existing pool AND
    within the new batch itself, using a containment-style token-overlap.

    Containment (intersection / min(|a|,|b|)) is friendlier than Jaccard for
    near-paraphrases like:
        "register pressure causing low occupancy"
        "register pressure causes low occupancy on CDNA"
    where one set is a near-superset of the other.
    """
    pool = [_norm_text(e.description) for e in existing]
    out: list[BlindSpot] = []
    for b in new:
        n = _norm_text(b.description)
        if not n:
            continue
        if any(_token_overlap(n, e) >= 0.6 for e in pool):
            continue
        out.append(b)
        pool.append(n)
    return out


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _token_overlap(a: str, b: str) -> float:
    """Containment overlap: |A ∩ B| / min(|A|, |B|).

    Returns 1.0 when one token set is a subset of the other (the common
    paraphrase case), and is symmetric in arguments.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ---------------------------------------------------------------------------
# Stage 6: Targeted second pass (parallelised, one batch per round)
# ---------------------------------------------------------------------------


async def _stage6_second_pass(
    *,
    model: Any,
    evidence: EvidenceSource,
    web: WebSearchSource | None,
    fetch_client: McpFetchClient | None,
    facts: Facts,
    blindspots: list[BlindSpot],
    cfg: DRAConfig,
    budget: CallBudget,
    round_idx: int,
) -> tuple[list[Answer], list[SearchResult]]:
    follow_ups = [b for b in blindspots if b.follow_up_question]
    if not follow_ups:
        return [], []

    prior_run_ctx = await asyncio.to_thread(evidence.prior_run_context)
    source_pack_text = await asyncio.to_thread(evidence.read_source_pack)
    sem = asyncio.Semaphore(max(1, cfg.web_concurrency))

    async def _do(b: BlindSpot) -> tuple[Answer, SearchResult]:
        async with sem:
            if budget.aborted:
                return (
                    _placeholder_answer(b.follow_up_question, f"second_pass_round_{round_idx}"),
                    SearchResult(hits=[], tried_queries=[]),
                )
            sr = await iterative_search(
                question=b.follow_up_question,
                evidence=evidence,
                web=web,
                fetch_client=fetch_client,
                top_k=cfg.retrieval_top_k,
                min_quality=cfg.min_search_results_per_question,
                max_refinements=cfg.web_max_refinements,
                search_depth=cfg.search_depth,
                per_question_result_budget=cfg.per_question_result_budget,
                read_top_k=cfg.read_top_k,
                read_max_length=cfg.read_max_length,
                refine_query_fn=_make_refine_fn(model, budget),
                triage_results_fn=_make_triage_fn(model, budget),
                on_call=budget.web_sink(),
            )
            ans = await asyncio.to_thread(
                _synthesize_one,
                model=model,
                facts=facts,
                question=b.follow_up_question,
                hits=sr.hits,
                prior_run_ctx=prior_run_ctx,
                source_stage=f"second_pass_round_{round_idx}",
                refinement_history=sr.tried_queries,
                cfg=cfg,
                budget=budget,
                source_pack_text=source_pack_text,
                # The blindspot's own description is rich context for picking
                # which file to read -- often more specific than the follow_up
                # question, which can be terse.
                affected_hint=[b.description, b.why_it_matters] if b.description else None,
            )
            return ans, sr

    pairs = await asyncio.gather(*(_do(b) for b in follow_ups))
    answers = [p[0] for p in pairs]
    srs = [p[1] for p in pairs]
    return answers, srs


# ---------------------------------------------------------------------------
# Query refinement callback (used by iterative_search)
# ---------------------------------------------------------------------------


def _make_refine_fn(model: Any, budget: CallBudget):
    async def _refine(question: str, tried: list[str], weak_hits: list[UnifiedHit]) -> str | None:
        if budget.aborted:
            return None
        weak_titles = "\n".join(f"- {h.title} ({h.origin})" for h in weak_hits[:8]) or "(none)"
        prompt = prompts.QUERY_REFINEMENT_PROMPT.format(
            question=question,
            tried_queries="\n".join(f"- {t}" for t in tried),
            weak_titles=weak_titles,
        )
        payload = await asyncio.to_thread(_llm_json, model, prompt, {"refined_query": ""}, budget)
        refined = str(payload.get("refined_query") or "").strip()
        return refined or None

    return _refine


def _make_triage_fn(model: Any, budget: CallBudget):
    async def _triage(
        question: str,
        query: str,
        round_idx: int,
        max_rounds: int,
        hits: list[UnifiedHit],
        read_top_k: int,
    ) -> dict[str, Any]:
        if budget.aborted:
            return {"selected_urls": [], "sufficient": False, "follow_up_queries": []}
        prompt = prompts.SEARCH_TRIAGE_PROMPT.format(
            question=question,
            query=query,
            round_idx=round_idx,
            max_rounds=max_rounds,
            results=_render_hits_for_triage(hits),
            read_top_k=read_top_k,
        )
        payload = await asyncio.to_thread(
            _llm_json,
            model,
            prompt,
            {"selected_urls": [], "sufficient": False, "follow_up_queries": []},
            budget,
        )
        return payload

    return _triage


def _render_hits_for_triage(hits: list[UnifiedHit], max_hits: int = 20) -> str:
    if not hits:
        return "(none)"
    lines: list[str] = []
    for i, h in enumerate(hits[:max_hits], 1):
        snippet = str(h.extra.get("snippet") or h.content or "").replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260] + "..."
        engine = h.extra.get("engine")
        engine_part = f", engine={engine}" if engine else ""
        lines.append(
            f"[{i}] title={h.title!r}, url={h.url!r}, origin={h.origin}{engine_part}, "
            f"score={h.score:.3f}, snippet={snippet!r}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 7: Final synthesis + Markdown rendering
# ---------------------------------------------------------------------------


def _stage7_final_synthesis(
    *,
    model: Any,
    inputs: DRAInputs,
    facts: Facts,
    questions: list[Question],
    answers: list[Answer],
    blindspots: list[BlindSpot],
    budget: CallBudget,
) -> DeepSearchArtifact:
    """Two-call Stage 7:

      (1) FINAL_GUIDANCE_PROMPT  -> small JSON (ranked_hypotheses + taskgen_guidance).
          Easy to parse because it explicitly forbids embedded Markdown blobs.
      (2) FINAL_EXEC_SUMMARY_PROMPT -> free-form Markdown executive summary.
          NEVER passed through a JSON parser.

    If (1) still fails, deterministic fallback guidance is constructed from
    each answer's status / taskgen_implications so the artifact is never empty.
    """
    facts_json = _json(facts.to_dict())
    answers_json = _json([a.to_dict() for a in answers])
    blindspots_json = _json([b.to_dict() for b in blindspots])

    guidance_prompt = prompts.FINAL_GUIDANCE_PROMPT.format(
        facts_json=facts_json,
        answers_json=answers_json,
        blindspots_json=blindspots_json,
    )
    guidance_payload = _llm_json(
        model,
        guidance_prompt,
        fallback={"ranked_hypotheses": [], "taskgen_guidance": {}},
        budget=budget,
        stage_tag="stage7_guidance",
        dump_dir=inputs.output_dir,
    )

    guidance_raw = guidance_payload.get("taskgen_guidance") or {}
    ranked_hypotheses = _str_list(guidance_payload.get("ranked_hypotheses"))
    guidance = TaskgenGuidance(
        prefer_first=_str_list(guidance_raw.get("prefer_first")),
        deprioritize=_str_list(guidance_raw.get("deprioritize")),
        reject=_str_list(guidance_raw.get("reject")),
        open_questions=_str_list(guidance_raw.get("open_questions")),
    )

    if not (
        guidance.prefer_first
        or guidance.deprioritize
        or guidance.reject
        or guidance.open_questions
        or ranked_hypotheses
    ):
        logger.warning("[DRA] Stage 7 guidance call returned empty; using deterministic fallback")
        guidance, ranked_hypotheses = _deterministic_guidance_from_answers(answers, blindspots)

    artifact = DeepSearchArtifact(
        inputs=inputs.to_dict_for_artifact(),
        facts=facts,
        questions=questions,
        answers=answers,
        blindspots=blindspots,
        ranked_hypotheses=ranked_hypotheses,
        taskgen_guidance=guidance,
    )

    artifact_executive_md = _stage7_executive_summary(
        model=model,
        facts_json=facts_json,
        guidance=guidance,
        ranked_hypotheses=ranked_hypotheses,
        answers_json=answers_json,
        blindspots_json=blindspots_json,
        artifact=artifact,
        budget=budget,
        dump_dir=inputs.output_dir,
    )
    artifact._executive_md = artifact_executive_md  # type: ignore[attr-defined]
    return artifact


def _stage7_executive_summary(
    *,
    model: Any,
    facts_json: str,
    guidance: TaskgenGuidance,
    ranked_hypotheses: list[str],
    answers_json: str,
    blindspots_json: str,
    artifact: DeepSearchArtifact,
    budget: CallBudget,
    dump_dir: Path,
) -> str:
    """Free-form Markdown call. No JSON parsing. Falls back to deterministic
    summary on any error or empty response.
    """
    if budget.aborted:
        return _fallback_executive_summary(artifact)
    budget.record("llm")
    guidance_json = _json(
        {
            "ranked_hypotheses": ranked_hypotheses,
            "taskgen_guidance": guidance.to_dict(),
        }
    )
    prompt = prompts.FINAL_EXEC_SUMMARY_PROMPT.format(
        facts_json=facts_json,
        guidance_json=guidance_json,
        answers_json=answers_json,
        blindspots_json=blindspots_json,
    )
    try:
        response = model.query(
            [
                {
                    "role": "system",
                    "content": "Output Markdown only. No JSON, no preamble, no closing remarks.",
                },
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.warning("[DRA] Stage 7 executive summary call failed: %s", exc)
        return _fallback_executive_summary(artifact)
    content = (response or {}).get("content", "") if isinstance(response, dict) else ""
    if not isinstance(content, str) or not content.strip():
        logger.warning("[DRA] Stage 7 executive summary returned empty")
        _dump_failed_response(dump_dir, "stage7_exec_summary", prompt, content or "")
        return _fallback_executive_summary(artifact)
    return content.strip()


def _deterministic_guidance_from_answers(
    answers: list[Answer], blindspots: list[BlindSpot]
) -> tuple[TaskgenGuidance, list[str]]:
    """Build TaskgenGuidance + ranked_hypotheses without any LLM call.

    Used when the structured Stage 7 call fails so the artifact never ships
    completely empty. The grouping is exactly the per-answer status the model
    already committed to during Stage 4 / Stage 6; we just project it.
    """
    prefer: list[str] = []
    deprio: list[str] = []
    reject: list[str] = []
    open_qs: list[str] = []
    for a in answers:
        impl = (a.taskgen_implications or "").strip()
        line = impl if impl else (a.answer or "").strip()
        if not line:
            continue
        line = _truncate_words(line, 60)
        bucket = {
            "prefer": prefer,
            "deprioritize": deprio,
            "reject": reject,
            "open": open_qs,
        }.get(a.status, open_qs)
        bucket.append(line)
    for b in blindspots:
        if b.follow_up_question:
            open_qs.append(_truncate_words(b.follow_up_question, 40))
    ranked = [
        _truncate_words(a.answer or a.question, 30)
        for a in answers
        if a.status == "prefer"
    ][:8]
    return (
        TaskgenGuidance(
            prefer_first=prefer[:8],
            deprioritize=deprio[:8],
            reject=reject[:8],
            open_questions=open_qs[:10],
        ),
        ranked,
    )


def _write_deep_search(artifact: DeepSearchArtifact, out_dir: Path) -> dict[str, Path]:
    md_path = out_dir / "deep_search.md"
    md_path.write_text(_render_deep_search_md(artifact), encoding="utf-8")
    return {"deep_search_md": md_path}


def _search_record(question: str, stage: str, sr: SearchResult) -> dict[str, Any]:
    return {
        "question": question,
        "stage": stage,
        "tried_queries": sr.tried_queries,
        "result_count": len(sr.hits),
        "raw_search_result_count": sr.search_result_count,
        "refinement_count": sr.refinement_count,
        "results": [
            {
                "title": h.title,
                "url": h.url,
                "origin": h.origin,
                "score": h.score,
                "snippet": (h.extra.get("snippet") or h.content or "")[:500],
                "engine": h.extra.get("engine"),
            }
            for h in sr.hits
        ],
    }


def _read_records_from_search_result(question: str, stage: str, sr: SearchResult) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in sr.read_records:
        content = str(rec.get("content") or "")
        out.append(
            {
                "question": question,
                "stage": stage,
                "url": rec.get("url"),
                "read_url": rec.get("read_url"),
                "chars": rec.get("chars"),
                "content": content,
            }
        )
    return out


def _write_search_records(records: list[dict[str, Any]], out_dir: Path, ts: str) -> Path:
    path = out_dir / "deep_search_search_records.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.write(
            json.dumps(
                {
                    "kind": "search_summary",
                    "records": len(records),
                    "total_results": sum(int(r.get("result_count") or 0) for r in records),
                    "ts": ts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return path


def _write_read_records(records: list[dict[str, Any]], out_dir: Path, ts: str) -> Path:
    path = out_dir / "deep_search_read_records.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.write(
            json.dumps(
                {
                    "kind": "read_summary",
                    "records": len(records),
                    "total_chars": sum(int(r.get("chars") or 0) for r in records),
                    "ts": ts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return path


def _write_synthesis_record(
    artifact: DeepSearchArtifact,
    out_dir: Path,
    stats: RunStats,
    budget: CallBudget,
) -> Path:
    """Write a JSONL record per (question, answer) plus run telemetry."""
    p = out_dir / "deep_search_synth_records.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for a in artifact.answers:
            rec = {
                "question": a.question,
                "answer": a.answer,
                "status": a.status,
                "source_stage": a.source_stage,
                "evidence": [e.to_dict() for e in a.evidence],
                "affected": a.affected,
                "kernel_path": artifact.inputs.get("kernel_path"),
                "bottleneck": artifact.facts.bottleneck_type,
                "ts": artifact.timestamp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Last record: the run-level summary so a consumer can index by run.
        meta = {
            "kind": "run_summary",
            "stats": {
                "questions": stats.questions_asked,
                "blindspot_rounds": stats.blindspot_rounds_run,
                "blindspots_total": stats.blindspots_total,
                "answers_total": stats.answers_total,
                "web_fetches_total": stats.web_fetches_total,
                "page_reads_total": stats.web_fetches_total,
                "refinements_total": stats.refinements_total,
                "aborted_on_budget": stats.aborted_on_budget,
            },
            "budget": budget.summary(),
            "ts": artifact.timestamp,
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return p


def _render_deep_search_md(artifact: DeepSearchArtifact) -> str:
    lines: list[str] = []
    exec_md = getattr(artifact, "_executive_md", "") or _fallback_executive_summary(artifact)
    lines.append("# Deep Search\n")
    lines.append(f"_Generated: {artifact.timestamp}_\n")
    lines.append("## Task-Ready Summary\n")
    lines.append(exec_md.rstrip() + "\n")

    lines.append("## Ranked Hypotheses\n")
    for i, h in enumerate(artifact.ranked_hypotheses, 1):
        lines.append(f"{i}. {h}")
    lines.append("")

    g = artifact.taskgen_guidance
    lines.append("## Task-Generator Guidance\n")
    lines.append("### Prefer First")
    lines.extend(_render_guidance_items(g.prefer_first, numbered=True))
    lines.append("\n### Deprioritize")
    lines.extend(_render_guidance_items(g.deprioritize))
    lines.append("\n### Reject")
    lines.extend(_render_guidance_items(g.reject))
    lines.append("\n### Open Measurements / Questions")
    lines.extend(_render_guidance_items(g.open_questions))
    lines.append("")

    if artifact.blindspots:
        lines.append("## Open Gaps\n")
        for b in artifact.blindspots[:10]:
            tag = f"round {b.round}" if b.round else ""
            lines.append(f"- **{b.description}** ({tag}) - {b.why_it_matters}".rstrip())
        lines.append("")

    lines.append("## Research Notes\n")
    for a in artifact.answers[:12]:
        lines.append(f"### Q ({a.source_stage}, status={a.status}): {a.question}")
        lines.append("")
        lines.append(_truncate_words(a.answer, 140))
        if a.affected:
            lines.append(f"\n**Affected:** {', '.join(a.affected)}")
        if a.evidence:
            ev_lines = []
            for ev in a.evidence[:5]:
                if ev.url:
                    ev_lines.append(f"- [{ev.source_type}] {ev.title} - {ev.url}")
                else:
                    ev_lines.append(f"- [{ev.source_type}] {ev.title} ({ev.chunk_id})")
            lines.append("\n**References consulted:**")
            lines.extend(ev_lines)
        if a.refinement_history and len(a.refinement_history) > 1:
            lines.append(f"\n**Search history:** {' -> '.join(a.refinement_history)}")
        if a.taskgen_implications:
            lines.append(f"\n**Task-gen implication:** {a.taskgen_implications}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_guidance_items(items: list[str], *, numbered: bool = False) -> list[str]:
    if not items:
        return ["- (none)"]
    out: list[str] = []
    for i, item in enumerate(items, 1):
        text = _format_guidance_item(item)
        prefix = f"{i}. " if numbered else "- "
        out.append(prefix + text)
    return out


def _format_guidance_item(item: Any) -> str:
    """Render task guidance compactly, including dict-shaped model output.

    The model sometimes returns small objects even though the schema stores
    guidance as strings. Keep the Markdown readable instead of dumping Python
    dict reprs into the main artifact. When no explicit id/title/strategy
    name is present, we omit the bold-prefixed title entirely rather than
    inventing a placeholder like "**Bet**".
    """
    if isinstance(item, dict):
        title = item.get("id") or item.get("title") or item.get("strategy")
        target = item.get("target") or item.get("target_files_or_functions")
        edit = item.get("edit_pattern") or item.get("strategy") or item.get("thesis")
        why = item.get("expected_speedup_mechanism") or item.get("expected_upside")
        kill = item.get("validation_kill_criterion") or item.get("kill_criteria")
        sources = item.get("sources") or item.get("source_pointers") or item.get("evidence")
        parts: list[str] = []
        if title:
            parts.append(f"**{title}**")
        if target:
            parts.append(f"Target: {target}")
        if edit:
            parts.append(f"Edit: {edit}")
        if why:
            parts.append(f"Why: {why}")
        if kill:
            parts.append(f"Kill: {kill}")
        if sources:
            parts.append(f"Sources: {sources}")
        if not parts:
            # Last-resort: render the bare dict so the run is still inspectable.
            return _truncate_words(json.dumps(item, ensure_ascii=False), 120)
        return " | ".join(_truncate_words(str(p), 70) for p in parts if p)
    return _truncate_words(str(item), 120)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip() + " ..."


def _fallback_executive_summary(artifact: DeepSearchArtifact) -> str:
    g = artifact.taskgen_guidance
    bits: list[str] = []
    if artifact.facts.bottleneck_type:
        bits.append(f"Bottleneck: **{artifact.facts.bottleneck_type}**.")
    if g.prefer_first:
        bits.append(f"Top recommendation: {g.prefer_first[0]}.")
    if not bits:
        bits.append("No executive summary produced.")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Experimental directions pass
# ---------------------------------------------------------------------------


def _experimental_pass(
    model: Any,
    inputs: DRAInputs,
    facts: Facts,
    deep_search: DeepSearchArtifact,
    budget: CallBudget,
) -> ExperimentalDirectionsArtifact:
    summary = {
        "ranked_hypotheses": deep_search.ranked_hypotheses,
        "taskgen_guidance": deep_search.taskgen_guidance.to_dict(),
    }
    prompt = prompts.EXPERIMENTAL_PROMPT.format(
        facts_json=_json(facts.to_dict()),
        deep_search_summary_json=_json(summary),
    )
    payload = _llm_json(model, prompt, fallback={"directions": []}, budget=budget)

    directions: list[ExperimentalDirection] = []
    for d in payload.get("directions") or []:
        if not isinstance(d, dict):
            continue
        thesis = str(d.get("thesis", "")).strip()
        if not thesis:
            continue
        directions.append(
            ExperimentalDirection(
                direction_id=str(d.get("direction_id") or _short_id()),
                thesis=thesis,
                why_orthogonal=str(d.get("why_orthogonal", "")).strip(),
                assumption_challenged=str(d.get("assumption_challenged", "")).strip(),
                strategy_family=str(d.get("strategy_family", "")).strip(),
                target_files_or_functions=_str_list(d.get("target_files_or_functions")),
                expected_upside=str(d.get("expected_upside", "")).strip(),
                implementation_cost=str(d.get("implementation_cost", "")).strip(),
                kill_criteria=str(d.get("kill_criteria", "")).strip(),
                notes_for_taskgen=str(d.get("notes_for_taskgen", "")).strip(),
            )
        )

    return ExperimentalDirectionsArtifact(
        inputs=inputs.to_dict_for_artifact(),
        directions=directions,
    )


def _write_experimental(
    artifact: ExperimentalDirectionsArtifact, out_dir: Path
) -> dict[str, Path]:
    md_path = out_dir / "experimental_directions.md"
    md_path.write_text(_render_experimental_md(artifact), encoding="utf-8")
    return {"experimental_md": md_path}


def _render_experimental_md(artifact: ExperimentalDirectionsArtifact) -> str:
    lines: list[str] = []
    lines.append("# Experimental Directions\n")
    lines.append(f"_Generated: {artifact.timestamp}_\n")
    if not artifact.directions:
        lines.append("_No directions produced._\n")
        return "\n".join(lines)

    for d in artifact.directions:
        lines.append(f"## {d.direction_id}: {d.thesis}\n")
        if d.why_orthogonal:
            lines.append(f"- **Why orthogonal:** {d.why_orthogonal}")
        if d.assumption_challenged:
            lines.append(f"- **Assumption challenged:** {d.assumption_challenged}")
        if d.strategy_family:
            lines.append(f"- **Strategy family:** {d.strategy_family}")
        if d.target_files_or_functions:
            lines.append(f"- **Targets:** {', '.join(d.target_files_or_functions)}")
        if d.expected_upside:
            lines.append(f"- **Expected upside:** {d.expected_upside}")
        if d.implementation_cost:
            lines.append(f"- **Cost:** {d.implementation_cost}")
        if d.kill_criteria:
            lines.append(f"- **Kill criteria:** {d.kill_criteria}")
        if d.notes_for_taskgen:
            lines.append(f"- **Notes for taskgen:** {d.notes_for_taskgen}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Per-question synthesis (shared between Stage 4 and Stage 6)
# ---------------------------------------------------------------------------


_FULL_SOURCE_PACK_BUDGET_CHARS = 80_000


def _synthesize_one(
    *,
    model: Any,
    facts: Facts,
    question: str,
    hits: list[UnifiedHit],
    prior_run_ctx: str,
    source_stage: str,
    refinement_history: list[str],
    cfg: DRAConfig,
    budget: CallBudget,
    source_pack_text: str = "",
    affected_hint: list[str] | None = None,
) -> Answer:
    if budget.aborted:
        return _placeholder_answer(question, source_stage)
    # Source-pack delivery: ship the WHOLE pack when it fits in a sensible
    # input-token budget. The packs we see in practice (single-kernel repos)
    # are 15-40 KB, well under the threshold, so the synthesizer always sees
    # the full code -- no chance of a routing-bias miss like the "wrapper
    # not inspected" blindspot. Only fall back to the keyword-overlap slicer
    # for unusually large packs (multi-kernel repos), where the full pack
    # would blow input-token budget across 28 per-question calls.
    local_code = ""
    if source_pack_text:
        if len(source_pack_text) <= _FULL_SOURCE_PACK_BUDGET_CHARS:
            local_code = source_pack_text
        else:
            local_code = select_local_code_chunks(
                source_pack_text,
                question,
                affected=affected_hint,
                extra_terms=facts.likely_targets,
            )
    prompt = prompts.PER_QUESTION_SYNTH_PROMPT.format(
        question=question,
        local_code=local_code or "(no local source pack available)",
        facts_json=_json(facts.to_dict()),
        kb_chunks=_render_unified_hits_for_prompt(hits) or "(no open-search results retrieved)",
        prior_run_context=prior_run_ctx or "(none)",
    )
    payload = _llm_json(model, prompt, fallback={}, budget=budget)
    status = payload.get("status", "open")
    if status not in ("prefer", "deprioritize", "reject", "open"):
        status = "open"

    evidence_list = _coerce_evidence(payload.get("evidence"), hits)

    return Answer(
        question=question,
        answer=str(payload.get("answer", "")).strip() or "(no answer produced)",
        evidence=evidence_list,
        affected=_str_list(payload.get("affected")),
        taskgen_implications=str(payload.get("taskgen_implications", "")).strip(),
        status=status,
        source_stage=source_stage,
        refinement_history=list(refinement_history),
    )


def _placeholder_answer(question: str, source_stage: str) -> Answer:
    return Answer(
        question=question,
        answer="(skipped: DRA budget exceeded)",
        evidence=[],
        affected=[],
        taskgen_implications="",
        status="open",
        source_stage=source_stage,
    )


def _coerce_evidence(raw: Any, hits: list[UnifiedHit]) -> list[EvidenceCite]:
    """Normalize an optional legacy ``evidence`` field into ``list[EvidenceCite]``.

    Current prompts no longer ask for citation lists; search/read traces are
    written separately. This helper remains for old model outputs and tests, but
    it no longer fabricates citations from retrieved hits.

    Post-processing: when a cite carries a URL or chunk_id that matches one
    of the actual hits, we OVERRIDE the model's stated ``source_type`` with
    the adapter's recorded ``origin``. This stops the LLM from mislabeling
    e.g. a ``web_search`` Google-SERP result as ``web_rocm_docs`` just because
    the URL happens to contain ``rocm.docs.amd.com``.
    """
    # Build a URL/chunk-id -> hit lookup so we can correct mislabeled origins.
    by_url: dict[str, UnifiedHit] = {}
    by_chunk: dict[str, UnifiedHit] = {}
    for h in hits:
        if h.url:
            by_url[h.url] = h
        if h.chunk_id:
            by_chunk[h.chunk_id] = h

    out: list[EvidenceCite] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                url = str(item.get("url") or "")
                chunk_id = str(item.get("chunk_id") or "")
                # Origin authority order: url match > chunk_id match > model claim
                authoritative = by_url.get(url) or by_chunk.get(chunk_id)
                source_type = (
                    authoritative.origin
                    if authoritative is not None
                    else str(item.get("source_type") or "web_search")
                )
                out.append(
                    EvidenceCite(
                        source_type=source_type,
                        title=str(item.get("title") or "")[:240],
                        url=url,
                        chunk_id=chunk_id,
                        snippet=str(item.get("snippet") or "")[:400],
                        score=float(item.get("score") or 0.0),
                    )
                )
            elif isinstance(item, str) and item.strip():
                out.append(EvidenceCite(source_type="web_search", title=item.strip()[:240]))
    if out:
        return out
    return []


def _render_unified_hits_for_prompt(hits: list[UnifiedHit], max_chunk_chars: int = 1800) -> str:
    """Render hits compactly for inclusion in a synthesis prompt.

    The synthesizer needs stable handles for inline references and search traces,
    so we lead each item with [N] {origin} {title} (url|chunk_id).
    """
    if not hits:
        return ""
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        ref = h.url or h.chunk_id or h.title
        body = h.content or ""
        if len(body) > max_chunk_chars:
            body = body[:max_chunk_chars] + " ...[truncated]"
        parts.append(
            f"### [{i}] {h.origin} | {h.title} | {ref} (score={h.score:.3f})\n{body}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _llm_json(
    model: Any,
    prompt: str,
    fallback: dict[str, Any],
    budget: CallBudget,
    *,
    stage_tag: str = "unknown",
    dump_dir: Path | None = None,
) -> dict[str, Any]:
    """Issue a single LLM call expecting exactly one JSON object back.

    Bumps the global call budget. Returns ``fallback`` on any failure or
    when budget is already exceeded.

    On parse failure, an LLM-driven repair pass is attempted. If both fail
    AND ``dump_dir`` is provided, the raw response (and the failed repair
    attempt, if any) are dumped to ``dump_dir/dra_llm_failures/<stage_tag>_<ts>.txt``
    so the silent-failure shape can be diagnosed instead of just losing
    the stage.
    """
    if budget.aborted:
        return dict(fallback)
    budget.record("llm")
    try:
        response = model.query(
            [
                {"role": "system", "content": "Return exactly one valid JSON object and nothing else."},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.warning("[DRA] LLM call failed (%s): %s", stage_tag, exc)
        return dict(fallback)

    content = (response or {}).get("content", "") if isinstance(response, dict) else ""
    if not isinstance(content, str) or not content.strip():
        logger.warning("[DRA] LLM returned empty content (%s)", stage_tag)
        return dict(fallback)

    parsed = _parse_json_object(content)
    if parsed is None:
        logger.warning(
            "[DRA] LLM did not return one exact JSON object (stage=%s len=%d); attempting repair",
            stage_tag,
            len(content),
        )
        parsed = _repair_json_object(model, content, fallback, budget)
        if parsed is None:
            _dump_failed_response(dump_dir, stage_tag, prompt, content)
            return dict(fallback)
    return parsed


def _dump_failed_response(
    dump_dir: Path | None,
    stage_tag: str,
    prompt: str,
    raw_response: str,
) -> None:
    """Persist a failed LLM response so a human can see what shape the model
    actually returned. Cheap forensic trail; never raises.
    """
    if dump_dir is None:
        return
    try:
        out = Path(dump_dir) / "dra_llm_failures"
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = out / f"{stage_tag}_{ts}.txt"
        path.write_text(
            "## stage\n"
            f"{stage_tag}\n\n"
            "## prompt (truncated to 8000 chars)\n"
            f"{prompt[:8000]}\n\n"
            "## raw response\n"
            f"{raw_response}\n",
            encoding="utf-8",
        )
        logger.warning("[DRA] dumped failed response to %s", path)
    except Exception as exc:
        logger.debug("[DRA] failed to dump LLM response: %s", exc)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from common model response shapes.

    Prefer exact JSON, but also accept fenced or balanced JSON object snippets.
    The prompts remain strict; this parser exists so a harmless Markdown wrapper
    does not erase an entire DRA stage.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    if text.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", text)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    for candidate in _balanced_json_object_candidates(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _balanced_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None
    return candidates


def _repair_json_object(
    model: Any,
    content: str,
    fallback: dict[str, Any],
    budget: CallBudget,
) -> dict[str, Any] | None:
    """Ask the model to convert its own malformed response into exact JSON.

    This keeps the main prompt strict while avoiding total stage loss when the
    model emits useful content with a preamble or Markdown wrapper.
    """
    if budget.aborted:
        return None
    budget.record("llm")
    repair_prompt = (
        "Convert the following response into exactly one valid JSON object for the "
        "same requested keys. Preserve the substantive content. Output JSON only.\n\n"
        "Response:\n"
        f"{content[:20000]}"
    )
    try:
        response = model.query(
            [
                {"role": "system", "content": "Return exactly one valid JSON object and nothing else."},
                {"role": "user", "content": repair_prompt},
            ]
        )
    except Exception as exc:
        logger.warning("[DRA] LLM JSON repair failed: %s", exc)
        return None
    repaired = (response or {}).get("content", "") if isinstance(response, dict) else ""
    parsed = _parse_json_object(repaired) if isinstance(repaired, str) else None
    if parsed is None:
        logger.warning("[DRA] LLM JSON repair still failed (len=%d)", len(repaired or ""))
        return None
    # If the stage expected a key like "questions", returning some other object
    # is worse than falling back visibly.
    if fallback and not any(k in parsed for k in fallback):
        return None
    return parsed


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _facts_from_payload(payload: dict[str, Any]) -> Facts:
    return Facts(
        kernel_language=str(payload.get("kernel_language", "unknown")),
        kernel_backend=str(payload.get("kernel_backend", "unknown")),
        bottleneck_type=str(payload.get("bottleneck_type", "unknown")),
        hot_kernels=[h for h in (payload.get("hot_kernels") or []) if isinstance(h, dict)],
        benchmark_contract=str(payload.get("benchmark_contract", "")),
        correctness_constraints=_str_list(payload.get("correctness_constraints")),
        prior_successes=_str_list(payload.get("prior_successes")),
        prior_failures=_str_list(payload.get("prior_failures")),
        likely_targets=_clean_likely_targets(payload.get("likely_targets")),
        notes=str(payload.get("notes", "")),
    )


def _str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [_format_guidance_item(v) if isinstance(v, dict) else str(v) for v in value if v is not None]
    return []


def _clean_likely_targets(value: Any) -> list[str]:
    out: list[str] = []
    for item in _str_list(value):
        s = item.strip()
        low = s.lower()
        if not s or "bet" in low or s in {"-", "*"}:
            continue
        if not any(tok in s for tok in ("/", ".", "_")) and low in {
            "kernel",
            "wrapper",
            "source",
            "task",
            "implementation",
        }:
            continue
        out.append(s)
    return out


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def _short_id() -> str:
    return f"ed-{uuid.uuid4().hex[:6]}"
