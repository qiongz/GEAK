"""Shared MCP client helpers for DRA.

The active DRA evidence path uses open-websearch MCP `search` for URL discovery
and AMD MCP `fetch` as a page reader for selected URLs. It intentionally avoids
AMD MCP `web_search` and `conduct_research`.

Why this exists separately from plain ``requests.get``:
  - The AMD ``commons_remote`` MCP server exposes a `fetch` tool that returns
    a URL's content already converted to readable markdown (drops nav, ads,
    boilerplate).
  - We verified empirically that the ``web_search`` MCP tool returns ``[]``
    intermittently and that ``conduct_research`` hallucinates citations.
    `fetch` is the only tool on that server that reliably returned grounded
    content, so DRA uses it only for URL -> markdown reads.

Operationally:
  - One persistent ``fastmcp.Client`` per process; lazily constructed on first
    call. Re-uses the same TCP connection across many `fetch_markdown` calls.
  - Bounded concurrency is handled at call-sites.
  - Failure modes (timeout, 4xx, paywall, empty body) all surface as ``None``
    so the caller can decide whether to retry, refine, or skip.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_FETCH_URL = "https://mcp-platform.amd.com/mcp/commons_remote/"
_DEFAULT_TIMEOUT_S = 15.0


class McpFetchClient:
    """Thin async wrapper around a persistent MCP client.

    Use ``McpFetchClient.get_or_create()`` to share a TCP connection across MCP
    calls. Current DRA uses ``call_tool("search", ...)`` for open-websearch URL
    discovery and ``fetch_markdown`` for selected URL reads.
    """

    _shared: dict[str, McpFetchClient] = {}

    def __init__(self, server_url: str = _DEFAULT_FETCH_URL, timeout_s: float = _DEFAULT_TIMEOUT_S):
        self.server_url = server_url
        self.timeout_s = timeout_s
        self._client: Any | None = None
        self._client_cm: Any | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def get_or_create(cls, server_url: str = _DEFAULT_FETCH_URL) -> McpFetchClient:
        if server_url not in cls._shared:
            cls._shared[server_url] = cls(server_url=server_url)
        return cls._shared[server_url]

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            try:
                from fastmcp import Client
            except ImportError as exc:
                raise RuntimeError(
                    "fastmcp not installed; required for DRA web fetch. "
                    "Install with `pip install fastmcp`."
                ) from exc
            self._client_cm = Client(self.server_url, timeout=self.timeout_s)
            self._client = await self._client_cm.__aenter__()
            logger.debug("[DRA] Opened MCP fetch client for %s", self.server_url)
            return self._client

    async def aclose(self) -> None:
        if self._client_cm is not None:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("[DRA] MCP client close failed: %s", exc)
            self._client = None
            self._client_cm = None

    async def fetch_markdown(self, url: str, max_length: int = 5000) -> str | None:
        """Fetch ``url`` and return its content as markdown, or None on failure.

        ``max_length`` caps the returned chars; the AMD `fetch` tool itself
        also takes a `max_length` arg, so we forward it to avoid pulling and
        then discarding huge bodies.
        """
        if not url:
            return None
        try:
            client = await self._ensure_client()
            result = await client.call_tool(
                "fetch",
                {"url": url, "max_length": int(max_length)},
            )
        except asyncio.TimeoutError:
            logger.debug("[DRA] fetch timeout: %s", url)
            return None
        except Exception as exc:
            logger.debug("[DRA] fetch failed for %s: %s", url, exc)
            return None

        # The MCP `fetch` returns a CallToolResult whose `content` carries
        # text blocks. Across MCP versions this surfaces in different fields;
        # try the structured paths first, then fall back to str(result).
        text = _extract_text_from_call_result(result)
        if not text or not text.strip():
            return None
        return text

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Generic tool invocation; returns the raw fastmcp ``CallToolResult``.

        Used by adapters that need a different MCP tool than ``fetch`` from
        the same persistent client (e.g. ``search_open_websearch`` calls
        the open-websearch daemon's ``search`` tool through this).

        Raises any ``fastmcp.Client`` exception unchanged so the caller can
        decide how to handle (the search adapter wraps with try/except).
        """
        client = await self._ensure_client()
        return await client.call_tool(tool_name, arguments)


def _extract_text_from_call_result(result: Any) -> str:
    """Pull the text payload out of a fastmcp CallToolResult, version-tolerant."""
    if result is None:
        return ""
    # New-style: structured_content / data
    sc = getattr(result, "structured_content", None)
    if isinstance(sc, dict):
        # commons_remote `fetch` puts the markdown right under "result" or top-level
        for key in ("result", "content", "text", "markdown"):
            v = sc.get(key)
            if isinstance(v, str) and v.strip():
                return v
    data = getattr(result, "data", None)
    if isinstance(data, str) and data.strip():
        return data
    if isinstance(data, list):
        # list of TextContent blocks
        chunks = []
        for item in data:
            t = getattr(item, "text", None)
            if isinstance(t, str):
                chunks.append(t)
        if chunks:
            return "\n".join(chunks)
    # Older MCP: .content is a list of dicts with "text"
    content = getattr(result, "content", None)
    if isinstance(content, list):
        chunks = []
        for item in content:
            t = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
            if isinstance(t, str):
                chunks.append(t)
        if chunks:
            return "\n".join(chunks)
    # Last resort: stringify
    return str(result) if result else ""


# ---------------------------------------------------------------------------
# Module-level convenience for callers that don't want to manage the client.
# Uses a single shared instance keyed by server_url.
# ---------------------------------------------------------------------------


async def fetch_markdown(
    url: str,
    *,
    server_url: str = _DEFAULT_FETCH_URL,
    max_length: int = 5000,
) -> str | None:
    """Module-level fetch helper. Reuses a single shared client per server_url."""
    client = McpFetchClient.get_or_create(server_url)
    return await client.fetch_markdown(url, max_length=max_length)


async def aclose_all() -> None:
    """Close every shared client. Call once at the end of a DRA run."""
    for client in list(McpFetchClient._shared.values()):
        await client.aclose()
    McpFetchClient._shared.clear()
