"""Web intelligence tools — search, fetch, and grounding.

Provides real-time web search and content extraction to ground
LLM outputs in actual data. No hallucinations. No lying.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"


@dataclass
class WebContext:
    query: str
    results: List[SearchResult] = field(default_factory=list)
    fetched_content: List[str] = field(default_factory=list)


async def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search the web using DuckDuckGo (free, no API key).

    Returns grounded, real-time results to feed into LLM prompts.
    """
    results = []
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:120.0) Gecko/20100101 Firefox/120.0"
            }
            r = await client.get(url, headers=headers)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            for item in soup.select(".result")[:max_results]:
                title_el = item.select_one(".result__title")
                snippet_el = item.select_one(".result__snippet")
                link_el = item.select_one(".result__url")

                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                link = link_el.get("href", "") if link_el else ""

                if title and link:
                    results.append(SearchResult(
                        title=title,
                        url=link,
                        snippet=snippet,
                    ))

        logger.info("Web search '%s': %d results", query[:50], len(results))
    except Exception as e:
        logger.warning("Web search failed for '%s': %s", query[:50], e)

    return results


async def web_fetch(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract clean text content from a URL."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; GenesisEngine/1.0)"
            }
            r = await client.get(url, headers=headers)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            # Collapse whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)

            return text[:max_chars]
    except Exception as e:
        logger.warning("Web fetch failed for %s: %s", url[:60], e)
        return ""


async def research_topic(query: str, fetch_depth: int = 2) -> WebContext:
    """Research a topic: search + fetch top results for deep context.

    This is the main entry point for grounding LLM prompts.
    Returns a WebContext with search results AND fetched content.
    """
    context = WebContext(query=query)

    # Step 1: Search
    context.results = await web_search(query, max_results=5)
    if not context.results:
        return context

    # Step 2: Fetch top N results for deep context
    tasks = [web_fetch(r.url) for r in context.results[:fetch_depth]]
    contents = await asyncio.gather(*tasks)
    context.fetched_content = [c for c in contents if c]

    return context


def format_context_for_prompt(context: WebContext, max_chars: int = 3000) -> str:
    """Format web research results into a prompt-ready string with citations."""
    if not context.results:
        return ""

    lines = [f"## Web Research: \"{context.query}\"", ""]

    for i, result in enumerate(context.results[:5]):
        lines.append(f"[{i+1}] {result.title}")
        lines.append(f"    URL: {result.url}")
        if result.snippet:
            lines.append(f"    {result.snippet[:200]}")
        lines.append("")

    # Add fetched content
    if context.fetched_content:
        lines.append("## Deep Context (fetched pages)")
        for i, content in enumerate(context.fetched_content[:2]):
            truncated = content[:max_chars // 2]
            lines.append(f"[Source {i+1}] {truncated}")
            lines.append("")

    return "\n".join(lines)
