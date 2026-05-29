"""Ingestion sources for The Oracle."""

from oracle.ingestion.sources import (
    ingest_hackernews,
    ingest_reddit,
    ingest_github_trending,
    ingest_tech_news,
    ingest_yfinance,
    ingest_all,
)

__all__ = [
    "ingest_hackernews",
    "ingest_reddit",
    "ingest_github_trending",
    "ingest_tech_news",
    "ingest_yfinance",
    "ingest_all",
]
