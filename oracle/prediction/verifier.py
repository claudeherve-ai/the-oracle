"""Canonical verification engine — fact-check predictions against real web sources.

Multi-source, multi-provider fact-checking against news, financial data,
GitHub, and arXiv, using natural-language-inference entailment with quoted
evidence spans (see ``oracle.tools.nli``) rather than keyword overlap.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from oracle.tools import research_topic, web_search, SearchResult
from oracle.tools.nli import EntailmentJudge, LLMEntailmentJudge

from oracle.llm import LLMProvider, LLMResponse
from oracle.models.prediction import Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verification types
# ---------------------------------------------------------------------------

@dataclass
class SourceEvidence:
    """Evidence from a single source, judged by natural-language inference."""
    url: str
    title: str
    snippet: str = ""
    supports: bool = False  # True = entails prediction (quote-verified)
    stance: str = "neutral"  # supports | contradicts | neutral
    quote: str = ""  # exact verbatim span the judge relied on
    relevance: float = 0.0  # 0-1
    credibility: float = 0.5  # 0-1 based on source authority

@dataclass
class VerificationResult:
    """Full verification of a prediction."""
    prediction_id: str
    statement: str
    original_confidence: float
    adjusted_confidence: float
    evidence: List[SourceEvidence] = field(default_factory=list)
    verdict: str = "unverifiable"  # supported, contradicted, unverifiable, mixed
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    summary: str = ""
    total_sources_checked: int = 0

# ---------------------------------------------------------------------------
# Source credibility scores
# ---------------------------------------------------------------------------

HIGH_CREDIBILITY_DOMAINS = {
    "reuters.com": 0.95, "bloomberg.com": 0.90, "bbc.com": 0.90,
    "apnews.com": 0.90, "nytimes.com": 0.85, "wsj.com": 0.85,
    "washingtonpost.com": 0.85, "ft.com": 0.85, "economist.com": 0.85,
    "nature.com": 0.95, "science.org": 0.95, "arxiv.org": 0.85,
    "github.com": 0.80, "techcrunch.com": 0.70, "theverge.com": 0.70,
    "arstechnica.com": 0.75, "wired.com": 0.70,
}

def _credibility(url: str) -> float:
    """Estimate source credibility from domain."""
    from urllib.parse import urlparse
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")
        return HIGH_CREDIBILITY_DOMAINS.get(domain, 0.4)
    except Exception:
        return 0.3


# ---------------------------------------------------------------------------
# Verifier Engine
# ---------------------------------------------------------------------------

class VerificationEngine:
    """Multi-source verification engine.

    Actually searches the web, news, financial data, GitHub, and arXiv
    to find supporting or contradicting evidence for each prediction.
    """

    def __init__(self, llm: LLMProvider, *, judge: Optional[EntailmentJudge] = None):
        self._llm = llm
        # NLI judge is the trust boundary; injectable for testing/mocking.
        self._judge: EntailmentJudge = judge or LLMEntailmentJudge(llm)

    async def verify(
        self,
        predictions: List[Prediction],
        *,
        deep_check: bool = False,
    ) -> List[VerificationResult]:
        """Verify all predictions against real sources."""
        if not predictions:
            return []

        logger.info("Verifying %d predictions (deep=%s)...", len(predictions), deep_check)
        results = await asyncio.gather(
            *[self._verify_one(p, deep_check) for p in predictions],
            return_exceptions=True,
        )

        verified = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Verification failed for prediction %d: %s", i, result)
                verified.append(VerificationResult(
                    prediction_id=predictions[i].id,
                    statement=predictions[i].statement,
                    original_confidence=predictions[i].confidence,
                    adjusted_confidence=predictions[i].confidence,
                    verdict="unverifiable",
                    summary=f"Verification error: {result}",
                ))
            else:
                verified.append(result)

        return verified

    async def _verify_one(self, pred: Prediction, deep: bool) -> VerificationResult:
        """Verify a single prediction using NLI-judged, quote-verified evidence."""
        statement = pred.statement
        logger.debug("Verifying: %s", statement[:80])

        # Run all source searches in parallel; each returns raw SearchResults.
        raw_lists = await asyncio.gather(
            self._check_web(statement),
            self._check_news(statement),
            self._check_financial(statement),
            self._check_github(statement),
            self._check_arxiv(statement),
            return_exceptions=True,
        )

        # Flatten + dedupe by URL (first occurrence wins).
        seen: set[str] = set()
        unique_results: List[SearchResult] = []
        for rl in raw_lists:
            if not isinstance(rl, list):
                continue
            for r in rl:
                if not getattr(r, "url", None) or r.url in seen:
                    continue
                seen.add(r.url)
                unique_results.append(r)

        # Judge every unique source with the NLI entailment judge.
        all_evidence = await self._judge_results(statement, unique_results)

        # Only quote-verified stances count toward the verdict; neutral excluded.
        supporting = [e for e in all_evidence if e.stance == "supports"]
        contradicting = [e for e in all_evidence if e.stance == "contradicts"]
        decisive = supporting + contradicting

        if not decisive:
            verdict = "unverifiable"
            adjusted_conf = pred.confidence
        elif len(contradicting) > len(supporting) * 2:
            verdict = "contradicted"
            adjusted_conf = max(0.01, pred.confidence - 0.25)
        elif len(supporting) > len(contradicting) * 3:
            verdict = "supported"
            total_cred = sum(e.credibility for e in supporting)
            boost = min(0.15, total_cred / max(len(supporting), 1) * 0.15)
            adjusted_conf = min(0.99, pred.confidence + boost)
        elif len(supporting) > len(contradicting):
            verdict = "mixed_supporting"
            adjusted_conf = pred.confidence
        else:
            verdict = "mixed"
            adjusted_conf = max(0.01, pred.confidence - 0.1)

        # Confidence interval: tighter with more decisive evidence.
        decisive_count = len(decisive)
        if decisive_count > 5:
            ci_half = 0.05
        elif decisive_count > 2:
            ci_half = 0.08
        else:
            ci_half = 0.12

        summary_parts = []
        if supporting:
            summary_parts.append(f"{len(supporting)} sources entail (quoted)")
        if contradicting:
            summary_parts.append(f"{len(contradicting)} sources contradict (quoted)")
        if not summary_parts:
            summary_parts.append("No quote-verified evidence found")

        return VerificationResult(
            prediction_id=pred.id,
            statement=statement,
            original_confidence=pred.confidence,
            adjusted_confidence=round(adjusted_conf, 4),
            evidence=all_evidence[:10],
            verdict=verdict,
            confidence_interval=(
                round(max(0.01, adjusted_conf - ci_half), 4),
                round(min(0.99, adjusted_conf + ci_half), 4),
            ),
            summary="; ".join(summary_parts),
            total_sources_checked=len(all_evidence),
        )

    async def _judge_results(
        self, statement: str, results: List[SearchResult]
    ) -> List[SourceEvidence]:
        """Run the NLI judge over each search result and build SourceEvidence.

        The judge decides ENTAILS/CONTRADICTS/NEUTRAL and must quote a verbatim
        span; an unquotable verdict is downgraded to neutral inside the judge.
        """
        if not results:
            return []

        async def _one(r: SearchResult) -> SourceEvidence:
            evidence_text = f"{r.title}. {r.snippet}".strip(". ")
            judgment = await self._judge.judge(
                statement, evidence_text, source_url=r.url
            )
            return SourceEvidence(
                url=r.url,
                title=r.title,
                snippet=r.snippet,
                supports=judgment.supports,
                stance=judgment.stance,
                quote=judgment.quote if judgment.quote_verified else "",
                relevance=1.0 if judgment.stance != "neutral" else 0.3,
                credibility=_credibility(r.url),
            )

        judged = await asyncio.gather(
            *[_one(r) for r in results], return_exceptions=True
        )
        return [e for e in judged if isinstance(e, SourceEvidence)]

    # ---- Source checkers (return raw SearchResults; judging is centralized) ----

    async def _check_web(self, statement: str) -> List[SearchResult]:
        """Check via web search."""
        try:
            context = await research_topic(statement, fetch_depth=2)
            return list(context.results[:5])
        except Exception as e:
            logger.debug("Web check failed: %s", e)
            return []

    async def _check_news(self, statement: str) -> List[SearchResult]:
        """Check via a targeted news-oriented web search."""
        try:
            return list(await web_search(f"{statement} news", max_results=5))
        except Exception as e:
            logger.debug("News check failed: %s", e)
            return []

    async def _check_financial(self, statement: str) -> List[SearchResult]:
        """Check via web search for market/stock predictions."""
        import re
        tickers = re.findall(r'\$?[A-Z]{1,5}\b', statement.upper())
        tickers = [t for t in tickers if t not in ("THE", "A", "I", "AT", "IN", "BY", "Q", "IS",
                                                     "AND", "OR", "BE", "TO", "IT", "WILL", "FOR",
                                                     "NEW", "CEO", "CFO", "CTO", "IPO", "AI", "ML")]
        if not tickers:
            return []

        results: List[SearchResult] = []
        for ticker in tickers[:3]:
            try:
                found = await web_search(f"{ticker} stock price forecast", max_results=2)
                results.extend(found)
            except Exception as e:
                logger.debug("Financial check failed for %s: %s", ticker, e)

        return results

    async def _check_github(self, statement: str) -> List[SearchResult]:
        """Check via web search for tech/open-source trend predictions."""
        tech_keywords = ["github", "open source", "repository", "framework", "library",
                         "package", "npm", "pypi", "cargo", "release"]
        if not any(kw in statement.lower() for kw in tech_keywords):
            return []

        try:
            return list(await web_search(f"{statement} github repository", max_results=3))
        except Exception as e:
            logger.debug("GitHub check failed: %s", e)
            return []

    async def _check_arxiv(self, statement: str) -> List[SearchResult]:
        """Check via web search for research/tech claims."""
        research_kw = ["study", "research", "paper", "breakthrough", "discovery",
                       "scientist", "researchers", "published", "journal"]
        if not any(kw in statement.lower() for kw in research_kw):
            return []

        try:
            return list(await web_search(f"{statement} research paper arxiv", max_results=3))
        except Exception as e:
            logger.debug("arXiv check failed: %s", e)
            return []

__all__ = ["VerificationEngine", "VerificationResult", "SourceEvidence"]
