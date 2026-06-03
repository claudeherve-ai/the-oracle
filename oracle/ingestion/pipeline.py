"""Automated ingestion pipeline — web-search backed.

Orchestrates topic-based data ingestion and converts results to Signal objects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from oracle.tools import web_search

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
    """Topic-based signal ingestion pipeline.

    Runs a web search per topic, converts results to Signal objects,
    deduplicates, and scores relevance.
    """

    def __init__(self, topics: List[str] | None = None):
        self._topics = topics or DEFAULT_TOPICS

    async def run(self, sources: Optional[List[str]] = None,
                  max_signals: int = 50) -> List[Signal]:
        """Run ingestion across all topics.

        ``sources`` (if provided) overrides the configured topic list for this
        run; otherwise the pipeline's default topics are used.
        """
        topics = sources or self._topics
        logger.info("Starting ingestion pipeline (%d topics)...", len(topics))

        results = await asyncio.gather(
            *(self._ingest_topic(topic) for topic in topics),
            return_exceptions=True,
        )

        all_signals: List[Signal] = []
        for topic, result in zip(topics, results):
            if isinstance(result, Exception):
                logger.warning("Ingestion failed for '%s': %s", topic, result)
            elif isinstance(result, list):
                all_signals.extend(result)
                logger.info("Ingested %d signals from '%s'", len(result), topic)

        deduped = self._deduplicate(all_signals)
        logger.info("Total signals: %d raw → %d deduplicated", len(all_signals), len(deduped))

        return deduped[:max_signals]

    async def _ingest_topic(self, topic: str) -> List[Signal]:
        """Ingest signals for a single topic via web search."""
        signals: List[Signal] = []
        try:
            results = await web_search(topic, max_results=5)
            for r in results:
                signals.append(Signal(
                    source=f"web:{topic[:40]}",
                    content=f"{r.title}\n{r.snippet}",
                    entities=self._extract_entities(r.title),
                    relevance=0.6,
                    metadata={
                        "url": r.url,
                        "topic": topic,
                        "source_type": "web",
                    },
                ))
        except Exception as e:
            logger.debug("Ingest failed for topic '%s': %s", topic, e)
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
