"""Real-source open-search adapters used by DRA.

The primary adapter is the local open-websearch sidecar. It returns ``WebHit``
objects containing URL + title + snippet; DRA uses those fields directly as
evidence and does not fetch page markdown through AMD MCP.

  1. open-websearch local daemon (multi-engine: DuckDuckGo + Bing + Brave +
     Exa + Startpage; primary general-purpose channel, no API keys)
  2. arxiv (academic papers, opt-in supplemental)
  3. GitHub Code Search (real implementations, opt-in supplemental)
  4. Hacker News Algolia (engineer commentary, opt-in supplemental)
  5. rocm.docs.amd.com sitemap (canonical AMD reference, opt-in supplemental)

The AMD ``commons_remote`` ``web_search`` MCP tool is intentionally NOT used:
verified empirically to be flaky (returns ``[]`` for the same query that
returned 7 hits a few minutes earlier; observed multiple times across runs).
The ``conduct_research`` tool is also avoided -- it returns hallucinated
citations (its own References section admits "no external sources were
available").

open-websearch (https://github.com/aas-ee/open-websearch, Apache-2.0) replaces
both: a self-hostable HTTP daemon scraping 7+ search engines without any API
keys, with multi-engine fallback that hedges against single-engine rate
limiting. Start it once on the host or as a sidecar:

    npx open-websearch@latest          # default: HTTP+STDIO mode on :3000

Then DRA's adapter calls the MCP ``search`` tool at
``{GEAK_DRA_OPEN_WEBSEARCH_URL}`` (default ``http://localhost:3000/mcp``). If
the daemon is not reachable the adapter degrades to ``[]``.

All adapters are async and degrade to ``[]`` on any error. The orchestrator
runs them in parallel with ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class WebHit:
    """One discovered URL with metadata used directly as evidence."""

    url: str
    title: str
    snippet: str = ""
    score: float = 0.0
    origin: str = "web"  # "web_arxiv" | "web_github" | "web_rocm_docs" | "web_hn"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_HTTP_TIMEOUT_S = 10.0


def _requests_get_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    """Sync HTTP GET returning parsed JSON or None. Used inside asyncio.to_thread."""
    import requests  # local import keeps the module importable without it for unit tests

    try:
        r = requests.get(url, headers=headers or {}, timeout=_DEFAULT_HTTP_TIMEOUT_S)
        if r.status_code != 200:
            logger.debug("HTTP %s for %s", r.status_code, url)
            return None
        return r.json()
    except Exception as exc:
        logger.debug("HTTP GET failed: %s -> %s", url, exc)
        return None


def _requests_get_text(url: str, *, headers: dict[str, str] | None = None) -> str | None:
    import requests

    try:
        r = requests.get(url, headers=headers or {}, timeout=_DEFAULT_HTTP_TIMEOUT_S)
        if r.status_code != 200:
            return None
        return r.text
    except Exception as exc:
        logger.debug("HTTP GET text failed: %s -> %s", url, exc)
        return None


def _norm_query(q: str, max_chars: int = 240) -> str:
    """Strip excessive punctuation / clip length so APIs do not 400."""
    q = re.sub(r"[\"'`]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:max_chars]


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "\u2026"


# ---------------------------------------------------------------------------
# open-websearch sidecar (primary adapter, MCP-over-HTTP)
# ---------------------------------------------------------------------------
#
# Sidecar daemon (https://github.com/aas-ee/open-websearch) that scrapes
# DuckDuckGo + Brave + Startpage + Bing + Exa + ... with NO API KEYS.
#
# The daemon speaks MCP-over-HTTP at ``/mcp`` (Streamable HTTP transport),
# NOT a REST API. We talk to it through the same persistent ``fastmcp.Client``
# pattern we already use for the AMD ``fetch`` tool, just at a different URL.
#
# Operationally:
#   1. Start once on the host: ``open-websearch`` (default: HTTP MCP on :3000).
#      In our environment npx wrappers occasionally die; running the bin
#      directly is more reliable:
#        node $(npm root -g)/open-websearch/build/index.js
#   2. DRA calls the ``search`` MCP tool at the URL pointed to by
#      ``GEAK_DRA_OPEN_WEBSEARCH_URL`` (default ``http://localhost:3000/mcp``).
#   3. If the daemon is unreachable the adapter returns ``[]`` and the four
#      supplemental adapters (arxiv/github/HN/ROCm-docs) carry the load.
#
# Per-engine notes (verified empirically):
#   - duckduckgo: works reliably, default
#   - brave:      works reliably
#   - startpage:  works reliably
#   - bing:       returns anti-bot/captcha pages frequently; excluded by default
#   - exa:        unverified; opt-in
# Engines are queried in parallel by the daemon and any per-engine failure
# surfaces in the response's ``partialFailures`` (which we log at debug).

_DEFAULT_OPEN_WEBSEARCH_URL = os.environ.get(
    "GEAK_DRA_OPEN_WEBSEARCH_URL", "http://localhost:3000/mcp"
).rstrip("/")
_DEFAULT_OPEN_WEBSEARCH_ENGINES = ("duckduckgo", "brave", "startpage")


def _parse_open_websearch_payload(
    raw_text: str, max_results: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse the daemon's JSON envelope into (results, partialFailures).

    Envelope shape (verified empirically):
        {"query": ..., "engines": [...], "totalResults": N,
         "results": [{title, url, description, source, engine}, ...],
         "partialFailures": [{engine, code, message}, ...]}
    """
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return [], []
    if not isinstance(parsed, dict):
        return [], []
    results = parsed.get("results") or []
    failures = parsed.get("partialFailures") or []
    if not isinstance(results, list):
        results = []
    if not isinstance(failures, list):
        failures = []
    return results[:max_results], failures


async def search_open_websearch(
    query: str,
    max_results: int = 7,
    *,
    daemon_url: str = _DEFAULT_OPEN_WEBSEARCH_URL,
    engines: tuple[str, ...] | list[str] = _DEFAULT_OPEN_WEBSEARCH_ENGINES,
) -> list[WebHit]:
    """Call the open-websearch daemon's ``search`` MCP tool.

    Returns up to ``max_results`` ``WebHit``s with origin ``"web_search"``,
    each tagged with ``extra["engine"]`` so callers can see which backend
    produced the hit.
    """
    q = _norm_query(query)
    if not q:
        return []

    # Reuse the same persistent fastmcp.Client cache as ``mcp_fetch`` --
    # different URL key, so this is a separate connection from the AMD MCP.
    from minisweagent.dra.mcp_fetch import McpFetchClient

    client = McpFetchClient.get_or_create(daemon_url)
    try:
        result = await client.call_tool(
            "search",
            {"query": q, "limit": int(max_results), "engines": list(engines)},
        )
    except Exception as exc:
        logger.debug("search_open_websearch call failed for %r: %s", q, exc)
        return []

    # The daemon returns the JSON envelope as a TextContent block under
    # ``result.content`` (NOT structured_content / data); pull the first
    # text block and parse.
    raw_text = ""
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        raw_text = getattr(first, "text", None) or ""
    if not raw_text:
        return []

    items, failures = _parse_open_websearch_payload(raw_text, max_results)
    if failures:
        logger.debug(
            "open-websearch partial failures for %r: %s",
            q,
            ", ".join(f"{f.get('engine')}:{f.get('code')}" for f in failures),
        )

    out: list[WebHit] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("href") or ""
        if not url:
            continue
        title = item.get("title") or "(no title)"
        snippet = item.get("description") or item.get("snippet") or ""
        engine = item.get("engine") or item.get("source") or "?"
        out.append(
            WebHit(
                url=url,
                title=_truncate(title, 200),
                snippet=_truncate(snippet, 400),
                # Position-based score; the daemon does not return a native
                # cross-engine ranking value, so we use rank as a proxy.
                score=float(max_results - i),
                origin="web_search",
                extra={"engine": engine},
            )
        )
    return out


# ---------------------------------------------------------------------------
# arxiv
# ---------------------------------------------------------------------------

# arxiv's ``export.arxiv.org/api/query`` returns Atom XML. We do NOT depend on
# `feedparser` -- there is one well-defined element layout we need.
_ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"


def _parse_arxiv_atom(xml_text: str, max_results: int) -> list[WebHit]:
    if not xml_text:
        return []
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)
    except Exception as exc:
        logger.debug("arxiv XML parse failed: %s", exc)
        return []

    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: list[WebHit] = []
    for entry in root.findall("a:entry", ns)[:max_results]:
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        link_el = entry.find("a:id", ns)
        url = (link_el.text or "").strip() if link_el is not None else ""
        if not url:
            continue
        out.append(
            WebHit(
                url=url,
                title=_truncate(title, 200),
                snippet=_truncate(summary, 400),
                score=0.0,
                origin="web_arxiv",
            )
        )
    return out


async def search_arxiv(query: str, max_results: int = 5) -> list[WebHit]:
    q = _norm_query(query)
    if not q:
        return []
    params = {
        "search_query": f"all:{q}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{_ARXIV_ENDPOINT}?{urllib.parse.urlencode(params)}"
    text = await asyncio.to_thread(_requests_get_text, url)
    return _parse_arxiv_atom(text or "", max_results)


# ---------------------------------------------------------------------------
# GitHub Code Search
# ---------------------------------------------------------------------------

_GITHUB_CODE_SEARCH = "https://api.github.com/search/code"


async def search_github(
    query: str,
    max_results: int = 5,
    token: str | None = None,
) -> list[WebHit]:
    """GitHub Code Search returns matching files. ``token`` raises rate
    limit from 10/min unauth to 30/min auth."""
    q = _norm_query(query)
    if not q:
        return []
    headers = {"Accept": "application/vnd.github+json"}
    tok = token or os.environ.get("GEAK_DRA_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    else:
        logger.debug("GitHub Code Search: no token, falling back to unauth (10/min cliff)")
    url = (
        f"{_GITHUB_CODE_SEARCH}?q={urllib.parse.quote(q)}"
        f"&per_page={max_results}"
    )
    data = await asyncio.to_thread(_requests_get_json, url, headers=headers)
    if not isinstance(data, dict):
        return []
    items = data.get("items") or []
    out: list[WebHit] = []
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
        html_url = item.get("html_url") or ""
        if not html_url:
            continue
        repo = (item.get("repository") or {}).get("full_name", "")
        path = item.get("path", "")
        title = f"{repo}/{path}" if repo else path or html_url
        # Code search does not return a snippet by default; fall back to path.
        snippet = path or ""
        out.append(
            WebHit(
                url=html_url,
                title=_truncate(title, 200),
                snippet=_truncate(snippet, 400),
                score=float(item.get("score") or 0.0),
                origin="web_github",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Hacker News (Algolia)
# ---------------------------------------------------------------------------

_HN_ENDPOINT = "https://hn.algolia.com/api/v1/search"


async def search_hn(query: str, max_results: int = 5) -> list[WebHit]:
    q = _norm_query(query)
    if not q:
        return []
    url = (
        f"{_HN_ENDPOINT}?query={urllib.parse.quote(q)}"
        f"&tags=story&hitsPerPage={max_results}"
    )
    data = await asyncio.to_thread(_requests_get_json, url)
    if not isinstance(data, dict):
        return []
    out: list[WebHit] = []
    for hit in data.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        url_field = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        title = hit.get("title") or hit.get("story_title") or "(no title)"
        snippet_parts: list[str] = []
        if hit.get("author"):
            snippet_parts.append(f"by {hit['author']}")
        if hit.get("points") is not None:
            snippet_parts.append(f"{hit['points']} points")
        if hit.get("num_comments") is not None:
            snippet_parts.append(f"{hit['num_comments']} comments")
        snippet = " | ".join(snippet_parts)
        out.append(
            WebHit(
                url=url_field,
                title=_truncate(title, 200),
                snippet=_truncate(snippet, 400),
                score=float(hit.get("points") or 0.0),
                origin="web_hn",
            )
        )
    return out


# ---------------------------------------------------------------------------
# ROCm docs (sitemap-based scoring)
# ---------------------------------------------------------------------------
#
# rocm.docs.amd.com publishes a sitemap.xml. We do NOT call the sitemap on
# every search -- we cache it for the lifetime of the process. Each entry is
# a doc URL; we score by how many query terms appear in the URL path.
#
# This is intentionally crude: it is a guidance-quality channel for canonical
# AMD reference material, not a full text index. DRA uses the URL/path snippet
# directly unless supplemental adapters are explicitly enabled.

_ROCM_SITEMAP_URL = "https://rocm.docs.amd.com/sitemap.xml"
_ROCM_DOC_HOST = "rocm.docs.amd.com"
_ROCM_SITEMAP_CACHE: list[str] | None = None
_ROCM_SITEMAP_LOCK = asyncio.Lock()


async def _load_rocm_sitemap() -> list[str]:
    global _ROCM_SITEMAP_CACHE
    if _ROCM_SITEMAP_CACHE is not None:
        return _ROCM_SITEMAP_CACHE
    async with _ROCM_SITEMAP_LOCK:
        if _ROCM_SITEMAP_CACHE is not None:
            return _ROCM_SITEMAP_CACHE
        text = await asyncio.to_thread(_requests_get_text, _ROCM_SITEMAP_URL)
        if not text:
            _ROCM_SITEMAP_CACHE = []
            return _ROCM_SITEMAP_CACHE
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(text)
            ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            urls = [el.text.strip() for el in root.findall(".//s:loc", ns) if el.text]
            urls = [u for u in urls if _ROCM_DOC_HOST in u]
            logger.debug("Loaded %d ROCm doc URLs", len(urls))
            _ROCM_SITEMAP_CACHE = urls
            return urls
        except Exception as exc:
            logger.debug("ROCm sitemap parse failed: %s", exc)
            _ROCM_SITEMAP_CACHE = []
            return _ROCM_SITEMAP_CACHE


def _score_rocm_url(url: str, terms: list[str]) -> float:
    """Crude path-overlap score: count term hits in the URL path slug.

    Returns 0 if no query terms appear in the path -- callers filter on this
    so unrelated URLs do NOT leak through via the depth bias alone.
    """
    if not terms:
        return 0.0
    path = urllib.parse.urlparse(url).path.lower()
    hits = sum(1 for t in terms if t and t in path)
    if hits == 0:
        return 0.0
    depth = max(1, path.count("/"))
    return hits + (1.0 / depth)


async def search_rocm_docs(query: str, max_results: int = 5) -> list[WebHit]:
    q = _norm_query(query).lower()
    if not q:
        return []
    sitemap = await _load_rocm_sitemap()
    if not sitemap:
        return []
    # Tokenize query into salient lowercase terms (drop short stopwords).
    terms = [t for t in re.split(r"[^a-z0-9_]+", q) if len(t) >= 3]
    scored = sorted(
        ((u, _score_rocm_url(u, terms)) for u in sitemap),
        key=lambda x: x[1],
        reverse=True,
    )
    out: list[WebHit] = []
    for u, s in scored[:max_results]:
        if s <= 0:
            break
        title = urllib.parse.urlparse(u).path.rstrip("/").split("/")[-1] or "ROCm doc"
        title = title.replace("-", " ").replace("_", " ")
        out.append(
            WebHit(
                url=u,
                title=_truncate(title, 200),
                snippet=urllib.parse.urlparse(u).path,
                score=s,
                origin="web_rocm_docs",
            )
        )
    return out


# ---------------------------------------------------------------------------
# WebSearchSource: orchestrates the four adapters concurrently.
# ---------------------------------------------------------------------------


class WebSearchSource:
    """Fans a single query out to all configured adapters in parallel.

    Order of precedence on duplicate URLs across adapters: open-websearch
    wins (multi-engine general SERP, the highest-recall channel). Hits from
    a later adapter that share a URL with an earlier one are skipped.
    """

    def __init__(
        self,
        *,
        github_token: str | None = None,
        per_source_top_k: int = 5,
        open_websearch_top_k: int = 7,
        open_websearch_url: str = _DEFAULT_OPEN_WEBSEARCH_URL,
        open_websearch_engines: tuple[str, ...] | list[str] = _DEFAULT_OPEN_WEBSEARCH_ENGINES,
        enable_open_websearch: bool = True,
        enable_arxiv: bool = True,
        enable_github: bool = True,
        enable_hn: bool = True,
        enable_rocm_docs: bool = True,
    ):
        self.github_token = github_token
        self.per_source_top_k = per_source_top_k
        self.open_websearch_top_k = open_websearch_top_k
        self.open_websearch_url = open_websearch_url
        self.open_websearch_engines = open_websearch_engines
        self.enable_open_websearch = enable_open_websearch
        self.enable_arxiv = enable_arxiv
        self.enable_github = enable_github
        self.enable_hn = enable_hn
        self.enable_rocm_docs = enable_rocm_docs

    async def search(self, query: str) -> list[WebHit]:
        """Run all enabled adapters in parallel; return merged hits.

        Adapters are listed in priority order so that earlier ones win
        any URL collision in the dedup loop below. open-websearch goes
        first because it is the only general-purpose SERP channel.
        """
        tasks: list[tuple[str, asyncio.Task[list[WebHit]]]] = []
        if self.enable_open_websearch:
            tasks.append(
                ("open_websearch", asyncio.create_task(
                    search_open_websearch(
                        query,
                        self.open_websearch_top_k,
                        daemon_url=self.open_websearch_url,
                        engines=self.open_websearch_engines,
                    )
                ))
            )
        if self.enable_arxiv:
            tasks.append(("arxiv", asyncio.create_task(search_arxiv(query, self.per_source_top_k))))
        if self.enable_github:
            tasks.append(
                ("github", asyncio.create_task(
                    search_github(query, self.per_source_top_k, token=self.github_token)
                ))
            )
        if self.enable_hn:
            tasks.append(("hn", asyncio.create_task(search_hn(query, self.per_source_top_k))))
        if self.enable_rocm_docs:
            tasks.append(("rocm", asyncio.create_task(search_rocm_docs(query, self.per_source_top_k))))
        if not tasks:
            return []

        results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
        merged: list[WebHit] = []
        seen_urls: set[str] = set()
        for (name, _), r in zip(tasks, results):
            if isinstance(r, Exception):
                logger.debug("WebSearchSource adapter %s raised: %s", name, r)
                continue
            for hit in r or []:
                if hit.url in seen_urls:
                    continue
                seen_urls.add(hit.url)
                merged.append(hit)
        return merged
