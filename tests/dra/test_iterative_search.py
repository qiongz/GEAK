"""Tests for iterative_search: open-search evidence and query refinement."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from minisweagent.dra.iterative_search import iterative_search
from minisweagent.dra.web_search import WebHit


@dataclass
class _StubEvidence:
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []

    def search(self, q: str, k: int):
        self.calls.append(q)
        raise AssertionError("iterative_search must not call local RAG")


@dataclass
class _StubWeb:
    web_hits_by_query: dict[str, list[WebHit]]
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []

    async def search(self, q: str) -> list[WebHit]:
        self.calls.append(q)
        return self.web_hits_by_query.get(q, [])


@dataclass
class _StubFetch:
    body_by_url: dict[str, str]
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []

    async def fetch_markdown(self, url: str, max_length: int = 5000) -> str | None:
        self.calls.append(url)
        return self.body_by_url.get(url)


def test_open_search_results_are_used_directly_without_rag_or_fetch():
    ev = _StubEvidence()
    web_hits = [
        WebHit(url=f"http://x/{i}", title=f"webhit{i}", snippet=f"snippet {i}", origin="web_search")
        for i in range(4)
    ]
    web = _StubWeb({"q": web_hits})

    sr = asyncio.run(iterative_search("q", ev, web, top_k=10, min_quality=4, max_refinements=2))  # type: ignore[arg-type]

    assert ev.calls == []
    assert web.calls == ["q"]
    assert len(sr.hits) == 4
    assert all(h.origin == "web_search" for h in sr.hits)
    assert sr.hits[0].url == "http://x/0"
    assert "snippet 0" in sr.hits[0].content


def test_refinement_runs_when_first_attempt_weak():
    ev = _StubEvidence()
    # first query returns 1 hit (still < min_quality=4); refined query returns enough
    web = _StubWeb({
        "q": [WebHit(url="u1", title="weak", origin="web_arxiv")],
        "refined-q": [
            WebHit(url=f"u{i+10}", title=f"good{i}", origin="web_github") for i in range(4)
        ],
    })

    refine_calls: list[tuple[str, list[str]]] = []

    async def refine(q, tried, weak_hits):
        refine_calls.append((q, list(tried)))
        return "refined-q"

    sr = asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q",
            ev,
            web,
            top_k=10,
            min_quality=4,
            max_refinements=2,
            search_depth=3,
            refine_query_fn=refine,
        )
    )
    assert refine_calls and refine_calls[0][1] == ["q"]
    assert sr.refinement_count == 1
    assert sr.tried_queries == ["q", "refined-q"]
    assert len(sr.hits) >= 4


def test_refinement_terminates_at_max_even_if_still_weak():
    ev = _StubEvidence()
    web = _StubWeb({})

    async def always_refine(q, tried, weak):
        return f"refine-{len(tried)}"

    sr = asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q",
            ev,
            web,
            top_k=10,
            min_quality=4,
            max_refinements=2,
            search_depth=3,
            refine_query_fn=always_refine,
        )
    )
    # Initial + 2 refinements = 3 tried queries
    assert sr.tried_queries == ["q", "refine-1", "refine-2"]
    assert sr.refinement_count == 2


def test_on_call_callback_fires_for_each_network_step():
    ev = _StubEvidence()
    web = _StubWeb({"q": [WebHit(url="u1", title="t", origin="web_hn")]})
    counts: dict[str, int] = {}

    def sink(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q", ev, web, top_k=10, min_quality=4, max_refinements=0, on_call=sink
        )
    )
    assert counts.get("web_search") == 1


def test_triage_selects_urls_to_read_and_records_content():
    ev = _StubEvidence()
    web = _StubWeb(
        {
            "q": [
                WebHit(url="https://example.com/a", title="A", snippet="a", origin="web_search"),
                WebHit(url="https://example.com/b", title="B", snippet="b", origin="web_search"),
            ]
        }
    )
    fetch = _StubFetch({"https://example.com/b": "full body b"})

    async def triage(question, query, round_idx, max_rounds, hits, read_top_k):
        return {
            "selected_urls": ["https://example.com/b"],
            "sufficient": True,
            "follow_up_queries": [],
        }

    sr = asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q",
            ev,
            web,
            fetch_client=fetch,
            top_k=10,
            min_quality=1,
            search_depth=1,
            read_top_k=1,
            triage_results_fn=triage,
        )
    )

    assert fetch.calls == ["https://example.com/b"]
    assert sr.web_fetch_count == 1
    assert sr.read_records[0]["content"] == "full body b"
    assert any("full body b" in h.content for h in sr.hits if h.url == "https://example.com/b")


def test_no_web_returns_no_hits():
    ev = _StubEvidence()
    sr = asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q", ev, web=None, fetch_client=None, top_k=10, min_quality=4, max_refinements=2
        )
    )
    assert sr.hits == []
    assert sr.web_fetch_count == 0
