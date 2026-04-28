"""Per-question open-search loop used by DRA Stage 3-4 / Stage 6.

For one research question we want a small set of high-signal web materials to
feed to the synthesizer. The shape of the loop is:

  1. Query the open-websearch sidecar (optionally with supplemental adapters).
  2. Triage returned titles/snippets/URLs and choose which URLs to read.
  3. Read selected URLs through AMD MCP fetch (URL -> markdown/text).
  4. If the result set is still weak (< ``min_quality`` items) and we have
     refinement budget left, ask the model to rewrite the query and recurse
     once. Track ``refinement_history`` so we never repeat a query.

Returns a uniform list of ``UnifiedHit`` with search metadata plus fetched page
content where available, along with tried queries and read records.

Bounded by:
  - ``per_question_result_budget`` web results per question (default 12)
  - ``max_refinements`` LLM rewrites per question (default 2)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from minisweagent.dra.evidence import EvidenceSource
from minisweagent.dra.mcp_fetch import McpFetchClient
from minisweagent.dra.web_search import WebHit, WebSearchSource

logger = logging.getLogger(__name__)


@dataclass
class UnifiedHit:
    """A chunk of research material feeding the synthesizer, KB or web alike.

    ``origin`` is the canonical taxonomy used by ``Answer.evidence``:
        kb | web_search | web_arxiv | web_github | web_rocm_docs | web_hn | facts | prior_run
    """

    title: str
    content: str
    score: float
    origin: str
    url: str = ""
    chunk_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_web(cls, hit: WebHit, content: str) -> UnifiedHit:
        return cls(
            title=hit.title,
            content=content,
            score=hit.score,
            origin=hit.origin,
            url=hit.url,
            extra={"snippet": hit.snippet, **hit.extra},
        )

    @classmethod
    def from_web_result(cls, hit: WebHit) -> UnifiedHit:
        """Create research material directly from open-search result metadata.

        A search result's title/snippet/URL are enough to keep the loop moving;
        selected URLs may later be upgraded with fetched page content.
        """
        content = "\n".join(
            part
            for part in (
                f"Title: {hit.title}",
                f"URL: {hit.url}",
                f"Snippet: {hit.snippet}" if hit.snippet else "",
                f"Engine: {hit.extra.get('engine')}" if hit.extra.get("engine") else "",
            )
            if part
        )
        return cls(
            title=hit.title,
            content=content,
            score=hit.score,
            origin=hit.origin,
            url=hit.url,
            extra={"snippet": hit.snippet, **hit.extra},
        )


@dataclass
class SearchResult:
    """Output of ``iterative_search`` for one question."""

    hits: list[UnifiedHit]
    tried_queries: list[str]
    web_fetch_count: int = 0
    refinement_count: int = 0
    search_result_count: int = 0
    read_records: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


async def iterative_search(
    question: str,
    evidence: EvidenceSource,
    web: WebSearchSource | None,
    fetch_client: McpFetchClient | None = None,
    *,
    initial_queries: list[str] | None = None,
    top_k: int = 12,
    min_quality: int = 4,
    max_refinements: int = 2,
    search_depth: int = 1,
    per_question_result_budget: int = 12,
    read_top_k: int = 4,
    read_max_length: int = 8000,
    refine_query_fn: Callable[[str, list[str], list[UnifiedHit]], Awaitable[str | None]] | None = None,
    triage_results_fn: Callable[
        [str, str, int, int, list[UnifiedHit], int], Awaitable[dict[str, Any]]
    ]
    | None = None,
    on_call: Callable[[str], None] | None = None,
) -> SearchResult:
    """Run the open-search -> refine loop for one question.

    Args:
      question:                The research question to find evidence for.
      evidence:                EvidenceSource for artifact readers only; local RAG
                               is intentionally not used by this loop.
      web:                     WebSearchSource (None disables web evidence).
      fetch_client:            Page reader used after triage (AMD fetch MCP).
      top_k:                   Final number of merged hits to return.
      min_quality:             If open-search returns fewer than this many useful results
                               and refinements remain, trigger a query rewrite.
      max_refinements:         Cap on LLM-driven query rewrites (NOT counting the
                               original query).
      search_depth:            Max search/triage rounds before synthesis.
      per_question_result_budget: Cap on search results kept per question,
                               regardless of how many adapters returned hits.
      read_top_k:               Max selected URLs read after triage.
      read_max_length:          Max markdown chars retained per read URL.
      refine_query_fn:         Async callable that, given the question and the
                               history of tried queries plus the current weak
                               hits, returns a refined query (or None to stop).
                               Defaults to no-op if not provided.
      on_call:                 Optional sink called once per network call with
                              a tag like "web_search" / "refine". The runner uses this to tally calls
                               against the global ``max_total_calls`` ceiling.

    Returns SearchResult with merged hits and the queries actually issued.
    """
    tried: list[str] = []
    web_results = 0
    web_reads = 0
    read_records: list[dict[str, Any]] = []
    refinements = 0

    current_query = question.strip()
    accumulated: dict[str, UnifiedHit] = {}  # dedupe key: url or title

    max_rounds = max(1, search_depth)
    # Clip and dedupe initial queries. If the question generator gave us
    # several search_queries, ROUND 1 issues them all in parallel instead
    # of letting only the first one run — the remaining slots then go to
    # triage follow-ups / refinement in later rounds.
    pending_queries: list[str] = []
    seen_pending: set[str] = set()
    for raw in initial_queries or [current_query]:
        q = _clip_query(raw)
        if not q or q in seen_pending:
            continue
        pending_queries.append(q)
        seen_pending.add(q)
    if not pending_queries:
        clipped = _clip_query(current_query)
        if clipped:
            pending_queries.append(clipped)
            seen_pending.add(clipped)

    for attempt in range(max_rounds):
        if not pending_queries:
            break

        # ---- 1. Open-search evidence ----
        # ROUND 1: drain ALL pending queries and fan them out in parallel.
        # ROUND 2+: pop one (refinement / triage follow-up). Keeping the
        # one-query-per-round shape for later rounds preserves the existing
        # refine-after-triage logic without changing call accounting.
        if attempt == 0:
            batch = list(pending_queries)
            pending_queries.clear()
        else:
            batch = [pending_queries.pop(0)]

        # Skip queries we've already tried (cheap idempotency vs the LLM
        # refining its way into a duplicate of an initial query).
        batch = [q for q in batch if q.strip() not in {t.strip() for t in tried}]
        if not batch:
            continue
        for q in batch:
            tried.append(q)
        current_query = batch[-1]  # for triage prompt context

        if web is not None and web_results < per_question_result_budget:
            results_per_query = await _search_batch(web, batch, on_call)
            for query, web_hits in zip(batch, results_per_query):
                if web_results >= per_question_result_budget:
                    break
                remaining = per_question_result_budget - web_results
                for hit in (web_hits or [])[:remaining]:
                    key = hit.url or hit.title
                    if not key or key in accumulated:
                        continue
                    accumulated[key] = UnifiedHit.from_web_result(hit)
                    web_results += 1
                    if web_results >= per_question_result_budget:
                        break

        # ---- 2. Triage gate: decide what to read and whether to go deeper. ----
        triage: dict[str, Any] = {}
        if triage_results_fn is not None and accumulated:
            try:
                triage = await triage_results_fn(
                    question,
                    current_query,
                    attempt + 1,
                    max_rounds,
                    list(accumulated.values()),
                    read_top_k,
                )
            except Exception as exc:
                logger.debug("[DRA] triage_results_fn failed: %s", exc)
                triage = {}

        # If triage can already pick read targets, read them now. Deeper searches
        # can add more candidates, but early reads give the later synthesis real
        # page bodies even if refinement fails.
        selected_urls = _selected_urls_from_triage(triage, list(accumulated.values()), read_top_k)
        if fetch_client is not None:
            for url in selected_urls:
                if any(r.get("url") == url for r in read_records):
                    continue
                rec = await _read_url(fetch_client, url, read_max_length, on_call)
                if not rec:
                    continue
                read_records.append(rec)
                web_reads += 1
                hit = accumulated.get(url)
                if hit is not None:
                    hit.content = _content_from_read_record(hit, rec)
                    hit.extra["read_url"] = rec.get("read_url") or url

        sufficient = bool(triage.get("sufficient")) if triage else len(accumulated) >= min_quality
        if sufficient and len(read_records) >= min(read_top_k, max(1, min_quality)):
            break

        # ---- 3. Deeper search / refinement ----
        if attempt >= max_rounds - 1:
            break
        follow_ups = _follow_ups_from_triage(triage)
        if not follow_ups and refine_query_fn is not None and refinements < max_refinements:
            if on_call:
                on_call("refine")
            try:
                refined = await refine_query_fn(question, list(tried), list(accumulated.values()))
            except Exception as exc:
                logger.debug("[DRA] refine_query_fn failed: %s", exc)
                refined = None
            if refined:
                follow_ups = [refined]
        for refined in follow_ups:
            refined = str(refined or "").strip()
            if not refined or refined in {q.strip() for q in tried}:
                continue
            if refined not in pending_queries:
                pending_queries.append(refined)
                refinements += 1

    merged = list(accumulated.values())
    merged.sort(key=lambda h: -h.score)
    return SearchResult(
        hits=merged[:top_k],
        tried_queries=tried,
        web_fetch_count=web_reads,
        refinement_count=refinements,
        search_result_count=web_results,
        read_records=read_records,
    )


_MAX_QUERY_CHARS = 140


def _clip_query(raw: str | None) -> str:
    """Normalize and clip a query before sending to the SERP.

    Long verbose research questions (e.g. ``How does the warp-cooperative
    kernel partition work across threads ...``) tend to return SEO noise
    because search engines treat anything past ~10 keywords as bag-of-words.
    Clip to a hard cap and strip whitespace; the question generator already
    produces concise ``search_queries``, so this only fires on the
    fallback path where the verbose ``question`` itself is being used.
    """
    if not raw:
        return ""
    q = " ".join(str(raw).split())
    if len(q) <= _MAX_QUERY_CHARS:
        return q
    return q[:_MAX_QUERY_CHARS].rsplit(" ", 1)[0]


async def _search_batch(
    web: WebSearchSource,
    queries: list[str],
    on_call: Callable[[str], None] | None,
) -> list[list[WebHit]]:
    """Issue a batch of independent searches concurrently.

    Each query consumes one ``web_search`` budget tick (the ``on_call`` sink
    counts them individually so the global call budget stays accurate). Any
    per-query exception is converted to ``[]`` so one failed engine cannot
    poison the whole batch.
    """

    async def _one(q: str) -> list[WebHit]:
        if on_call:
            on_call("web_search")
        try:
            return await web.search(q)
        except Exception as exc:
            logger.debug("[DRA] web search failed for %r: %s", q, exc)
            return []

    return await asyncio.gather(*(_one(q) for q in queries))


def _selected_urls_from_triage(
    triage: dict[str, Any], hits: list[UnifiedHit], read_top_k: int
) -> list[str]:
    raw = triage.get("selected_urls") if isinstance(triage, dict) else None
    selected = [str(u).strip() for u in raw or [] if str(u).strip()] if isinstance(raw, list) else []
    seen_hit_urls = {h.url for h in hits if h.url}
    selected = [u for u in selected if u in seen_hit_urls]
    if not selected:
        selected = [h.url for h in hits if h.url][:read_top_k]
    out: list[str] = []
    for url in selected:
        if url not in out:
            out.append(url)
    return out[:read_top_k]


def _follow_ups_from_triage(triage: dict[str, Any]) -> list[str]:
    raw = triage.get("follow_up_queries") if isinstance(triage, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw[:3] if str(q).strip()]


def _arxiv_html_url(url: str) -> str:
    if "arxiv.org/abs/" not in url:
        return url
    return url.replace("arxiv.org/abs/", "arxiv.org/html/")


async def _read_url(
    fetch_client: McpFetchClient,
    url: str,
    max_length: int,
    on_call: Callable[[str], None] | None,
) -> dict[str, Any] | None:
    candidates = [_arxiv_html_url(url)]
    if candidates[0] != url:
        candidates.append(url)
    for read_url in candidates:
        if on_call:
            on_call("read")
        content = await fetch_client.fetch_markdown(read_url, max_length=max_length)
        if content:
            return {
                "url": url,
                "read_url": read_url,
                "content": content,
                "chars": len(content),
            }
    return None


def _content_from_read_record(hit: UnifiedHit, rec: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            f"Title: {hit.title}",
            f"URL: {hit.url}",
            f"Read URL: {rec.get('read_url') or hit.url}",
            rec.get("content") or "",
        )
        if part
    )
