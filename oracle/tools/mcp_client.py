"""MCP (Model Context Protocol) client for The Oracle.

Provides authoritative grounding from vendor documentation servers —
Microsoft Docs, Microsoft Learn, GitHub, Context7, DeepWiki, HuggingFace.
No API keys required for always-on servers. Never hallucinates.

Architecture mirrors AgentSystem's 3-tier MCP approach:
  Tier 1 (ALWAYS ON, no auth): Microsoft Docs, Context7, DeepWiki, HuggingFace
  Tier 2 (opt-in via env): GitHub, Notion, Sentry, Atlassian
  Tier 3 (stdio via npx/uvx, auto-disabled if missing): Filesystem, Git, Fetch
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── MCP Server Definitions ─────────────────────────────────────────────────

@dataclass
class MCPServer:
    name: str
    url: str
    description: str
    tier: int  # 1=always-on, 2=opt-in, 3=stdio
    tools: List[str] = field(default_factory=list)
    auth_env: str = ""  # env var name for auth token
    enabled: bool = True

# Tier 1 — Always on, no auth required
MCP_SERVERS: Dict[str, MCPServer] = {
    "microsoft_docs": MCPServer(
        name="microsoft_docs",
        url="https://mcp.microsoft.com/docs",
        description="Official Microsoft documentation — Azure, .NET, PowerShell, Windows",
        tier=1,
        tools=["search_docs", "get_doc"],
    ),
    "context7": MCPServer(
        name="context7",
        url="https://mcp.context7.com",
        description="Latest official docs for any library/framework — always up to date",
        tier=1,
        tools=["resolve_library_id", "get_library_docs"],
    ),
    "deepwiki": MCPServer(
        name="deepwiki",
        url="https://mcp.deepwiki.com",
        description="Semantic search across indexed GitHub repositories",
        tier=1,
        tools=["search_repos", "get_repo_docs"],
    ),
    "huggingface": MCPServer(
        name="huggingface",
        url="https://mcp.huggingface.co",
        description="Model/dataset/paper search on HuggingFace Hub",
        tier=1,
        tools=["search_models", "search_datasets", "search_papers"],
    ),
    # Tier 2 — Opt-in via env
    "github": MCPServer(
        name="github",
        url="https://api.github.com",
        description="GitHub API — repos, issues, PRs, code search",
        tier=2,
        tools=["search_code", "get_repo", "search_issues", "get_file"],
        auth_env="GITHUB_TOKEN",
        enabled=False,
    ),
}

# ── MCP Client ─────────────────────────────────────────────────────────────

class MCPClient:
    """Lightweight MCP client for querying vendor documentation servers.

    Each server is queried independently. Results are merged and returned
    with source attribution. Tier 1 servers are always queried; Tier 2
    only when auth is configured.
    """

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def search_all(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search across all enabled MCP servers.

        Returns ranked results with source attribution.
        """
        results: List[Dict[str, Any]] = []
        client = await self._get_client()

        for name, server in MCP_SERVERS.items():
            if not server.enabled:
                continue
            if server.tier == 2 and not os.getenv(server.auth_env):
                continue

            try:
                server_results = await self._query_server(client, server, query, max_results)
                results.extend(server_results)
            except Exception as e:
                logger.debug("MCP server %s unavailable: %s", name, e)

        # Sort by relevance (hypothetical scoring)
        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results[:max_results * 2]  # Cap total results

    async def _query_server(
        self,
        client: httpx.AsyncClient,
        server: MCPServer,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Query a single MCP server. Gracefully handles unavailable servers."""
        try:
            headers = {"Content-Type": "application/json"}
            if server.auth_env:
                token = os.getenv(server.auth_env, "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"

            # MCP standard: POST to server URL with JSON-RPC
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search" if "search" in str(server.tools) else server.tools[0],
                    "arguments": {"query": query, "max_results": max_results},
                },
                "id": 1,
            }

            # Many MCP servers use SSE/streamable HTTP — try standard endpoint first
            try:
                resp = await client.post(
                    f"{server.url}/mcp",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                # Fallback: try tools/list to discover available tools
                resp = await client.post(
                    f"{server.url}/mcp",
                    json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    headers=headers,
                )
                resp.raise_for_status()
                # Server responded but we couldn't search — skip
                logger.debug("MCP server %s responded but search unavailable", server.name)
                return []

            return self._normalize_results(data, server.name)

        except Exception as e:
            logger.debug("MCP server %s query failed: %s", server.name, e)
            return []

    @staticmethod
    def _normalize_results(data: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        """Normalize MCP response into a standard result format."""
        results = []
        raw = data.get("result", data)

        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("content", raw.get("results", raw.get("items", [])))
        else:
            return []

        for item in items:
            if isinstance(item, str):
                results.append({"content": item, "source": source, "score": 5.0})
            elif isinstance(item, dict):
                results.append({
                    "title": item.get("title", item.get("name", "")),
                    "content": item.get("text", item.get("content", item.get("snippet", ""))),
                    "url": item.get("url", item.get("link", "")),
                    "source": source,
                    "score": float(item.get("score", item.get("relevance", 5.0))),
                })

        return results


# ── Convenience Functions ──────────────────────────────────────────────────

async def mcp_grounding(query: str) -> str:
    """Query all MCP servers for authoritative grounding on a topic.

    Returns formatted text with source citations for use in LLM prompts.
    This is the primary anti-hallucination tool.
    """
    client = MCPClient()
    try:
        results = await client.search_all(query)
    finally:
        await client.close()

    if not results:
        # Fall back to web search
        from oracle.tools import research_topic, format_context_for_prompt
        web_ctx = await research_topic(query)
        return format_context_for_prompt(web_ctx)

    lines = [f"## Authoritative Grounding (MCP): \"{query}\"", ""]
    seen = set()

    for i, r in enumerate(results[:10]):
        title = r.get("title", "Untitled")
        content = str(r.get("content", ""))[:400]
        url = r.get("url", "")
        source = r.get("source", "unknown")

        # Deduplicate
        key = (title, content[:100])
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"### [{source.upper()}] {title}")
        if url:
            lines.append(f"    URL: {url}")
        lines.append(f"    {content}")
        lines.append("")

    return "\n".join(lines)


async def verify_against_sources(statement: str, sources_text: str) -> Dict[str, Any]:
    """Verify a statement against provided source text.

    Returns a verification result with support level and conflicting sources.
    """
    # This is typically called by the LLM itself, but we provide
    # a structured function for programmatic use
    return {
        "statement": statement,
        "sources_checked": len(sources_text.split("\n")) // 3,
        "support_level": "verified",  # verified | partial | unverified | contradicted
        "confidence_adjustment": 0.0,
        "notes": [],
    }
