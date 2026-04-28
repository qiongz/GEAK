"""Configuration for DRA (Deep Research Artifact) generation.

DRA is opt-in. The defaults are tuned for source-pack-grounded, web-heavy kernel
research:

  - 50 ranked questions through the first synthesis pass
  - 3 blindspot rounds with up to 10 follow-up questions each (early-stop on dedup)
  - Per-question open-search / triage / read / query-refinement loop
  - Web search via the open-websearch sidecar daemon (multi-engine, no API
    keys). Top results are triaged, then selected URLs are read through AMD
    MCP `fetch` as a page reader. DRA does NOT use local RAG or AMD web_search.
  - Supplemental adapters (arxiv API, GitHub Code Search, ROCm docs sitemap,
    HN Algolia) are opt-in.
  - Per-question search continues until enough useful web material is available
    or the configured depth/refinement budget is exhausted.
  - A global ceiling of 800 LLM+web calls (1200 in deep mode) keeps a single
    DRA run bounded. Empirically an A5-style deep run on a typical kernel
    consumes ~330-600 calls under the parallel-initial-query loop, so this
    leaves ~2x headroom before the runner aborts.

To enable the open-websearch primary channel, start the sidecar once per
host. It speaks MCP-over-HTTP (NOT REST) at ``/mcp`` on port 3000:

    npm install -g open-websearch
    node $(npm root -g)/open-websearch/build/index.js   # MODE=http default

(Running via ``npx`` directly is unreliable in some environments because the
npx wrapper sometimes gets reaped immediately after spawning. Invoking the
``build/index.js`` entrypoint with ``node`` is the most stable path.)

If the daemon is not reachable the open-websearch adapter degrades to ``[]``.

Env vars:
  GEAK_DRA_DISABLE=1                   -> turn DRA off entirely
  GEAK_DRA_MODEL=claude-opus-4.6       -> model used for all DRA LLM calls
  GEAK_DRA_API_KEY=...                 -> optional override
  GEAK_DRA_DEEP_MODE=1                 -> deeper search/read profile
  GEAK_DRA_MAX_QUESTIONS=50            -> max questions through full research (Stage 4)
  GEAK_DRA_MAX_BLINDSPOTS=10           -> max follow-up doubts PER blindspot round
  GEAK_DRA_MAX_BLINDSPOT_ROUNDS=3      -> how many rounds of Stage 5+6 to run
  GEAK_DRA_RETRIEVAL_TOP_K=12          -> final web hits retained per question
  GEAK_DRA_MIN_SEARCH_RESULTS=4        -> enough search/read material before stopping
  GEAK_DRA_MAX_TOTAL_CALLS=800         -> hard ceiling on LLM+web calls combined (1200 deep)
  GEAK_DRA_WEB_DISABLE=1               -> skip open-search research
  GEAK_DRA_WEB_MAX_REFINEMENTS=2       -> per-question query refinement retries
  GEAK_DRA_WEB_CONCURRENCY=8           -> async parallelism for question batch
  GEAK_DRA_PER_QUESTION_RESULT_BUDGET=12 -> open-search results kept per researched question
  GEAK_DRA_SEARCH_DEPTH=2              -> search/triage rounds before reading
  GEAK_DRA_READ_TOP_K=4                -> selected URLs read per researched question
  GEAK_DRA_READ_MAX_LENGTH=8000        -> max markdown chars retained per read URL
  GEAK_DRA_SUPPLEMENTAL_WEB_ENABLE=1     -> also query arxiv/GitHub/HN/ROCm docs
  GEAK_DRA_OPEN_WEBSEARCH_DISABLE=1    -> disable the open-websearch adapter
  GEAK_DRA_OPEN_WEBSEARCH_URL=...      -> override MCP endpoint (default http://localhost:3000/mcp)
  GEAK_DRA_RUN_EXPERIMENTAL=1          -> also produce experimental_directions.md
  GEAK_DRA_INDEX_PATH=...              -> override path to RAG index (otherwise default)
  GEAK_DRA_USE_PRIOR_RUNS=1            -> include cross_session prior-run context
  GEAK_DRA_GITHUB_TOKEN=...            -> auth GitHub Code Search (10/min unauth -> 30/min auth)
  GEAK_DRA_MCP_FETCH_URL=...           -> AMD MCP fetch endpoint used as page reader
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_DEFAULT_MCP_FETCH_URL = "https://mcp-platform.amd.com/mcp/commons_remote/"
# open-websearch speaks MCP-over-HTTP at /mcp (Streamable HTTP transport),
# not REST. The default points at the MCP endpoint of the local sidecar.
_DEFAULT_OPEN_WEBSEARCH_URL = "http://localhost:3000/mcp"


@dataclass
class DRAConfig:
    enabled: bool = True
    model_name: str = "claude-opus-4.6"
    api_key: str | None = None

    # Question / blindspot budgets
    deep_mode: bool = False
    max_questions: int = 50
    max_blindspots: int = 10  # per round
    max_blindspot_rounds: int = 3
    retrieval_top_k: int = 12

    # Search adequacy: not a citation/source-count rule, just a signal for when
    # iterative_search has gathered enough material to synthesize.
    min_search_results_per_question: int = 4

    # Hard safety ceiling on combined LLM + web calls per DRA invocation.
    # Sized for the parallel-initial-query loop (each ranked question now
    # fans 3-5 queries into the SERP in round 1 instead of one), so the web
    # budget is roughly 2-3x what the original sequential loop consumed.
    # Empirically a deep-mode run on a typical kernel uses ~330-600 calls;
    # 800 / 1200 (deep) leaves ~2x headroom before the runner aborts.
    max_total_calls: int = 800

    # Web research
    web_search_enabled: bool = True
    web_max_refinements: int = 2
    web_concurrency: int = 8
    per_question_result_budget: int = 12
    search_depth: int = 1
    read_top_k: int = 4
    read_max_length: int = 8000
    supplemental_web_enabled: bool = False
    github_token: str | None = None
    mcp_fetch_url: str = _DEFAULT_MCP_FETCH_URL
    # open-websearch sidecar daemon: primary general-purpose SERP channel.
    # Adapter degrades gracefully if the daemon is unreachable.
    open_websearch_enabled: bool = True
    open_websearch_url: str = _DEFAULT_OPEN_WEBSEARCH_URL

    # Optional second artifact.
    run_experimental: bool = True

    # Optional context sources.
    use_prior_runs: bool = False
    index_path: str | None = None

    @classmethod
    def from_env(cls) -> DRAConfig:
        if _env_bool("GEAK_DRA_DISABLE", False):
            return cls(enabled=False)

        deep_mode = _env_bool("GEAK_DRA_DEEP_MODE", False)

        return cls(
            enabled=True,
            deep_mode=deep_mode,
            model_name=os.environ.get("GEAK_DRA_MODEL", "claude-opus-4.6").strip(),
            api_key=os.environ.get("GEAK_DRA_API_KEY", "").strip() or None,
            max_questions=_env_int("GEAK_DRA_MAX_QUESTIONS", 16 if deep_mode else 50),
            max_blindspots=_env_int("GEAK_DRA_MAX_BLINDSPOTS", 6 if deep_mode else 10),
            max_blindspot_rounds=_env_int("GEAK_DRA_MAX_BLINDSPOT_ROUNDS", 2 if deep_mode else 3),
            retrieval_top_k=_env_int("GEAK_DRA_RETRIEVAL_TOP_K", 16 if deep_mode else 12),
            min_search_results_per_question=_env_int("GEAK_DRA_MIN_SEARCH_RESULTS", 4),
            max_total_calls=_env_int("GEAK_DRA_MAX_TOTAL_CALLS", 1200 if deep_mode else 800),
            web_search_enabled=not _env_bool("GEAK_DRA_WEB_DISABLE", False),
            web_max_refinements=_env_int("GEAK_DRA_WEB_MAX_REFINEMENTS", 3 if deep_mode else 2),
            web_concurrency=_env_int("GEAK_DRA_WEB_CONCURRENCY", 6 if deep_mode else 8),
            per_question_result_budget=_env_int(
                "GEAK_DRA_PER_QUESTION_RESULT_BUDGET", 16 if deep_mode else 12
            ),
            search_depth=_env_int("GEAK_DRA_SEARCH_DEPTH", 2 if deep_mode else 1),
            read_top_k=_env_int("GEAK_DRA_READ_TOP_K", 5 if deep_mode else 4),
            read_max_length=_env_int("GEAK_DRA_READ_MAX_LENGTH", 12000 if deep_mode else 8000),
            supplemental_web_enabled=_env_bool("GEAK_DRA_SUPPLEMENTAL_WEB_ENABLE", False),
            github_token=os.environ.get("GEAK_DRA_GITHUB_TOKEN", "").strip() or None,
            mcp_fetch_url=os.environ.get("GEAK_DRA_MCP_FETCH_URL", _DEFAULT_MCP_FETCH_URL).strip(),
            open_websearch_enabled=not _env_bool("GEAK_DRA_OPEN_WEBSEARCH_DISABLE", False),
            open_websearch_url=os.environ.get(
                "GEAK_DRA_OPEN_WEBSEARCH_URL", _DEFAULT_OPEN_WEBSEARCH_URL
            ).rstrip("/"),
            run_experimental=_env_bool("GEAK_DRA_RUN_EXPERIMENTAL", True),
            use_prior_runs=_env_bool("GEAK_DRA_USE_PRIOR_RUNS", False),
            index_path=os.environ.get("GEAK_DRA_INDEX_PATH", "").strip() or None,
        )
