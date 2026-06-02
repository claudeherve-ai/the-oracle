"""Ingestion sources for The Oracle.

Delegates to IngestionPipeline for full automated ingestion from
news, financial, social media, GitHub, and arXiv.

Also retains the legacy direct-source ingestors for backward compatibility.

CITATION: Boil the Ocean upgrade — orchestrates IngestionPipeline.
Session: Hermes Agent, 2026-06-01.
BACK-LINK: /home/tedch/the-oracle/oracle/ingestion/pipeline.py
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import List, Optional

from oracle.models.prediction import Signal

logger = logging.getLogger(__name__)


def _make_id(source: str, content: str) -> str:
    return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()[:16]


# ── Legacy direct ingestors (kept for backward compat) ──────────


async def ingest_hackernews(limit: int = 30) -> List[Signal]:
    """Fetch top stories from Hacker News API."""
    import httpx
    signals = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
            r.raise_for_status()
            ids = r.json()[:limit]

            async def fetch_story(sid):
                try:
                    sr = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
                    return sr.json()
                except Exception:
                    return None

            tasks = [fetch_story(sid) for sid in ids]
            results = await asyncio.gather(*tasks)

            for story in results:
                if not story or "title" not in story:
                    continue
                title = story.get("title", "")
                url = story.get("url", "")
                content = f"{title} - {url}" if url else title
                signals.append(Signal(
                    id=_make_id("hackernews", content),
                    source="hackernews",
                    content=content,
                    entities=_extract_entities(title),
                    sentiment=0.0,
                    relevance=0.7 if story.get("score", 0) > 50 else 0.5,
                    metadata={"score": story.get("score", 0), "url": url},
                ))
        logger.info("HN: %d signals", len(signals))
    except Exception as e:
        logger.warning("HN ingestion failed: %s", e)
    return signals


async def ingest_reddit() -> List[Signal]:
    """Fetch from r/technology and r/programming via RSS."""
    import feedparser
    import httpx
    signals = []
    subreddits = [
        ("https://www.reddit.com/r/technology/.rss", "reddit_technology"),
        ("https://www.reddit.com/r/programming/.rss", "reddit_programming"),
    ]
    for url, source_name in subreddits:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                signals.append(Signal(
                    id=_make_id(source_name, title),
                    source=source_name,
                    content=title,
                    entities=_extract_entities(title),
                    sentiment=0.0,
                    relevance=0.5,
                    metadata={"link": entry.get("link", "")},
                ))
        except Exception as e:
            logger.warning("%s ingestion failed: %s", source_name, e)
    logger.info("Reddit: %d signals", len(signals))
    return signals


async def ingest_github_trending() -> List[Signal]:
    """Fetch trending repositories from GitHub."""
    import httpx
    signals = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": "created:>2026-01-01", "sort": "stars", "order": "desc", "per_page": 20},
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            r.raise_for_status()
            data = r.json()

            for repo in data.get("items", []):
                desc = repo.get("description", "") or ""
                signal_text = f"{repo['full_name']}: {desc} ({repo['stargazers_count']} stars, language: {repo.get('language', 'N/A')})"
                signals.append(Signal(
                    id=_make_id("github", signal_text),
                    source="github_trending",
                    content=signal_text,
                    entities=[repo["full_name"]],
                    sentiment=0.0,
                    relevance=0.6 if repo["stargazers_count"] > 100 else 0.4,
                    metadata={
                        "stars": repo["stargazers_count"],
                        "language": repo.get("language"),
                        "url": repo["html_url"],
                    },
                ))
        logger.info("GitHub: %d signals", len(signals))
    except Exception as e:
        logger.warning("GitHub ingestion failed: %s", e)
    return signals


async def ingest_tech_news() -> List[Signal]:
    """Fetch from TechCrunch, The Verge, Ars Technica RSS feeds."""
    import feedparser
    signals = []
    feeds = [
        ("https://techcrunch.com/feed/", "techcrunch"),
        ("https://www.theverge.com/rss/index.xml", "theverge"),
        ("https://feeds.arstechnica.com/arstechnica/index", "arstechnica"),
    ]
    for url, source_name in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                signals.append(Signal(
                    id=_make_id(source_name, title),
                    source=source_name,
                    content=title,
                    entities=_extract_entities(title),
                    sentiment=0.0,
                    relevance=0.5,
                    metadata={"link": entry.get("link", ""), "published": str(entry.get("published", ""))},
                ))
        except Exception as e:
            logger.warning("%s ingestion failed: %s", source_name, e)
    logger.info("Tech News: %d signals", len(signals))
    return signals


async def ingest_yfinance() -> List[Signal]:
    """Fetch news for major tickers using yfinance in a thread."""
    import concurrent.futures
    signals = []
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"]

    def _fetch_ticker(ticker):
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news = stock.news[:10] if hasattr(stock, 'news') and stock.news else []
            results = []
            for n in news:
                title = n.get("title", "")
                content_str = f"{ticker}: {title}"
                results.append(Signal(
                    id=_make_id("yfinance", content_str),
                    source="yfinance",
                    content=content_str,
                    entities=[ticker],
                    sentiment=0.0,
                    relevance=0.6,
                    metadata={"ticker": ticker, "url": n.get("link", "")},
                ))
            return results
        except Exception as e:
            logger.warning("yfinance %s failed: %s", ticker, e)
            return []

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [loop.run_in_executor(pool, _fetch_ticker, t) for t in tickers]
        results = await asyncio.gather(*futures)
        for r in results:
            signals.extend(r)

    logger.info("YFinance: %d signals", len(signals))
    return signals


# ── Combined ingestion (legacy + pipeline) ─────────────────────


async def ingest_all() -> List[Signal]:
    """Run all ingestion sources in parallel (legacy mode)."""
    results = await asyncio.gather(
        ingest_hackernews(),
        ingest_reddit(),
        ingest_github_trending(),
        ingest_tech_news(),
        ingest_yfinance(),
        return_exceptions=True,
    )
    all_signals = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error("Source %d failed: %s", i, r)
        elif isinstance(r, list):
            all_signals.extend(r)
    logger.info("Total ingested: %d signals from 5 sources", len(all_signals))
    return all_signals


async def ingest_pipeline(
    sources: Optional[List[str]] = None,
    max_signals: int = 200,
) -> List[Signal]:
    """Run the full automated ingestion pipeline (new pipeline mode).

    Uses the IngestionPipeline which ingests from gstack_tools providers
    (news, financial, social, github, arxiv) with deduplication and scoring.

    Args:
        sources: Specific sources. None = all.
        max_signals: Max signals to return.
    """
    from oracle.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    return await pipeline.run(sources=sources, max_signals=max_signals)


def _extract_entities(text: str) -> List[str]:
    """Simple entity extraction from text (company names, products)."""
    known = ["Apple", "Google", "Microsoft", "Meta", "Amazon", "Nvidia",
             "Tesla", "OpenAI", "Anthropic", "ChatGPT", "Claude", "Gemini",
             "iOS", "Android", "Windows", "Linux", "React", "Python",
             "TypeScript", "Rust", "Kubernetes", "Docker", "AWS", "Azure"]
    found = []
    text_lower = text.lower()
    for name in known:
        if name.lower() in text_lower:
            found.append(name)
    return found
