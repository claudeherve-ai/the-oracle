"""Automated ingestion pipeline — news, social, financial, GitHub, arXiv.

Orchestrates multi-source data ingestion and converts to Signal objects.

CITATION: Built for automated The Oracle signal ingestion.
Session: Hermes Agent, 2026-06-01.
BACK-LINK: /home/tedch/the-oracle/
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from gstack_tools.news import fetch_news, NewsArticle
from gstack_tools.social_media import fetch_social_posts, SocialPost
from gstack_tools.financial import get_stock_quote, get_crypto_quote
from gstack_tools.github_trends import get_trending_repos, search_github
from gstack_tools.arxiv_papers import get_latest_papers, search_arxiv

from oracle.models.prediction import Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trending topics to monitor
# ---------------------------------------------------------------------------

DEFAULT_TOPICS = [
    # Tech
    "artificial intelligence breakthrough",
    "large language model release",
    "quantum computing advance",
    # Markets
    "stock market rally",
    "tech IPO filing",
    "crypto regulation",
    # Products
    "Apple product launch",
    "Tesla new model",
    "NVIDIA earnings",
    # Science
    "CRISPR breakthrough",
    "fusion energy milestone",
    "climate technology",
]

DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AMZN"]
DEFAULT_CRYPTO = ["BTC", "ETH", "SOL"]

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """Multi-source signal ingestion pipeline.

    Pulls data from news, social media, financial APIs, GitHub, and arXiv,
    converts to Signal objects, deduplicates, and scores relevance.
    """

    def __init__(self, topics: List[str] | None = None):
        self._topics = topics or DEFAULT_TOPICS

    async def run(self, max_signals: int = 50) -> List[Signal]:
        """Run full ingestion pipeline across all sources."""
        logger.info("Starting ingestion pipeline (%d topics)...", len(self._topics))

        # Run all sources in parallel
        results = await asyncio.gather(
            self._ingest_news(max_signals),
            self._ingest_social(max_signals),
            self._ingest_financial(),
            self._ingest_github(),
            self._ingest_arxiv(),
            return_exceptions=True,
        )

        all_signals: List[Signal] = []
        for i, result in enumerate(results):
            source_name = ["news", "social", "financial", "github", "arxiv"][i]
            if isinstance(result, Exception):
                logger.warning("Ingestion failed for %s: %s", source_name, result)
            elif isinstance(result, list):
                all_signals.extend(result)
                logger.info("Ingested %d signals from %s", len(result), source_name)

        # Deduplicate
        deduped = self._deduplicate(all_signals)
        logger.info("Total signals: %d raw → %d deduplicated", len(all_signals), len(deduped))

        return deduped[:max_signals]

    async def _ingest_news(self, max_items: int) -> List[Signal]:
        signals = []
        for topic in self._topics[:3]:  # Limit to avoid rate limits
            try:
                articles = await fetch_news(query=topic, max_articles=5)
                for a in articles:
                    signals.append(Signal(
                        source=f"news:{a.source}",
                        content=f"{a.title}\n{a.summary}",
                        entities=self._extract_entities(a.title),
                        relevance=0.7 if a.published_at else 0.5,
                        metadata={
                            "url": a.url,
                            "published_at": a.published_at.isoformat() if a.published_at else None,
                            "source_type": "news",
                        },
                    ))
            except Exception as e:
                logger.debug("News ingest failed for topic '%s': %s", topic, e)
        return signals

    async def _ingest_social(self, max_items: int) -> List[Signal]:
        signals = []
        try:
            posts = await fetch_social_posts(
                query="technology OR AI OR startup",
                limit=15,
                providers=["hackernews", "reddit"],
            )
            for p in posts:
                signals.append(Signal(
                    source=f"social:{p.platform}",
                    content=f"{p.title}\n{p.content}",
                    entities=self._extract_entities(p.title),
                    relevance=min(1.0, p.score / 500) if p.score else 0.5,
                    metadata={
                        "url": p.url,
                        "score": p.score,
                        "comments": p.comments,
                        "platform": p.platform,
                        "created_at": p.created_at.isoformat() if p.created_at else None,
                        "source_type": "social",
                    },
                ))
        except Exception as e:
            logger.debug("Social ingest failed: %s", e)
        return signals

    async def _ingest_financial(self) -> List[Signal]:
        signals = []
        for ticker in DEFAULT_STOCKS[:5]:
            try:
                quote = await get_stock_quote(ticker)
                if quote:
                    direction = "up" if quote.change > 0 else "down"
                    signals.append(Signal(
                        source="financial:yahoo",
                        content=f"{ticker} ${quote.price} ({direction} {abs(quote.change_percent)}%)",
                        entities=[ticker],
                        relevance=0.8,
                        metadata={
                            "ticker": ticker,
                            "price": quote.price,
                            "change_percent": quote.change_percent,
                            "volume": quote.volume,
                            "source_type": "financial",
                        },
                    ))
            except Exception as e:
                logger.debug("Financial ingest failed for %s: %s", ticker, e)
        return signals

    async def _ingest_github(self) -> List[Signal]:
        signals = []
        try:
            repos = await get_trending_repos(language="", since="daily", limit=10)
            for repo in repos:
                signals.append(Signal(
                    source="github:trending",
                    content=f"{repo.name}: {repo.description} (⭐{repo.stars})",
                    entities=[repo.name.split("/")[0], repo.name],
                    relevance=min(1.0, repo.stars / 10000),
                    metadata={
                        "repo": repo.name,
                        "stars": repo.stars,
                        "language": repo.language,
                        "topics": repo.topics,
                        "url": repo.url,
                        "source_type": "github",
                    },
                ))
        except Exception as e:
            logger.debug("GitHub ingest failed: %s", e)
        return signals

    async def _ingest_arxiv(self) -> List[Signal]:
        signals = []
        categories = ["cs.AI", "cs.CL", "cs.LG", "q-fin.ST"]
        for cat in categories[:2]:
            try:
                papers = await get_latest_papers(category=cat, max_results=5)
                for paper in papers:
                    signals.append(Signal(
                        source="arxiv",
                        content=f"{paper.title}: {paper.summary[:200]}",
                        entities=paper.authors[:3] + [c for c in paper.categories if c],
                        relevance=0.6,
                        metadata={
                            "paper_id": paper.id,
                            "authors": paper.authors,
                            "categories": paper.categories,
                            "published": paper.published.isoformat() if paper.published else None,
                            "url": paper.abstract_url,
                            "source_type": "arxiv",
                        },
                    ))
            except Exception as e:
                logger.debug("arXiv ingest failed for %s: %s", cat, e)
        return signals

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Simple entity extraction from text."""
        import re
        # Find capitalized words/phrases (potential named entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Deduplicate
        seen = set()
        result = []
        for e in entities:
            if e.lower() not in seen and len(e) > 2:
                seen.add(e.lower())
                result.append(e)
        return result[:5]

    @staticmethod
    def _deduplicate(signals: List[Signal]) -> List[Signal]:
        """Deduplicate signals by content similarity."""
        seen_hashes: set[str] = set()
        unique = []
        for s in signals:
            # Simple hash of first 100 chars
            h = hash(s.content[:100].lower().strip())
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(s)
        return unique


__all__ = ["IngestionPipeline", "DEFAULT_TOPICS"]
