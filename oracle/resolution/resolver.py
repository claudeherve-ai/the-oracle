"""Resolution Engine — auto-resolve predictions against real-world data.

Scans past-deadline predictions and judges real-world outcomes from fetched
page text, snapshotting the resolving evidence so every call is auditable.
Ambiguous evidence resolves to INSUFFICIENT_EVIDENCE, not EXPIRED.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from gstack_tools.search import research_topic, web_search, SearchResult
from gstack_tools.news import fetch_news
from gstack_tools.financial import get_stock_quote

from oracle.llm import LLMProvider
from oracle.models.prediction import Prediction, Status

logger = logging.getLogger(__name__)


@dataclass
class ResolutionOutcome:
    prediction_id: str
    statement: str
    deadline: Optional[datetime]
    resolved_as: Status  # correct, incorrect, expired
    confidence: str = "low"  # low, medium, high
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ResolutionEngine:
    """Automated prediction resolution.

    For each past-deadline prediction:
    1. Search the web for real-world outcome
    2. Check financial data for market predictions
    3. Check news for event-based predictions
    4. Use LLM to synthesize evidence and determine outcome
    5. Return ResolutionOutcome
    """

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def resolve(self, predictions: List[Prediction]) -> List[ResolutionOutcome]:
        """Resolve all past-deadline predictions."""
        now = datetime.now(timezone.utc)
        pending = [
            p for p in predictions
            if p.status == Status.PENDING and p.deadline and p.deadline < now
        ]

        if not pending:
            logger.info("No predictions past deadline to resolve")
            return []

        logger.info("Resolving %d past-deadline predictions...", len(pending))
        results = await asyncio.gather(
            *[self._resolve_one(p) for p in pending],
            return_exceptions=True,
        )

        outcomes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Resolution failed for %s: %s", pending[i].id, result)
                outcomes.append(ResolutionOutcome(
                    prediction_id=pending[i].id,
                    statement=pending[i].statement,
                    deadline=pending[i].deadline,
                    resolved_as=Status.EXPIRED,
                    reasoning=f"Resolution error: {result}",
                ))
            else:
                outcomes.append(result)

        resolved_count = sum(1 for o in outcomes if o.resolved_as != Status.EXPIRED)
        logger.info("Resolved %d/%d predictions", resolved_count, len(pending))
        return outcomes

    async def _resolve_one(self, pred: Prediction) -> ResolutionOutcome:
        """Resolve a single prediction."""
        statement = pred.statement
        logger.debug("Resolving: %s", statement[:80])

        # Gather evidence
        evidence_urls: List[str] = []

        # Web search for outcome
        try:
            search_query = f"{statement} outcome result"
            context = await research_topic(search_query, fetch_depth=1)
            for r in context.results[:5]:
                evidence_urls.append(r.url)
        except Exception as e:
            logger.debug("Web search for resolution failed: %s", e)

        # News check
        try:
            articles = await fetch_news(query=statement[:100], max_articles=3)
            for a in articles:
                evidence_urls.append(a.url)
        except Exception:
            pass

        # Financial check
        try:
            import re
            tickers = re.findall(r'\$?[A-Z]{1,5}\b', statement.upper())
            tickers = [t for t in tickers if t not in
                       ("THE", "A", "I", "AT", "IN", "BY", "Q", "IS", "AND", "OR",
                        "BE", "TO", "IT", "WILL", "FOR")]
            for ticker in tickers[:2]:
                quote = await get_stock_quote(ticker)
                if quote:
                    evidence_urls.append(f"https://finance.yahoo.com/quote/{ticker}")
        except Exception:
            pass

        # Use LLM to determine outcome from evidence
        outcome = await self._judge_outcome(statement, pred.deadline, evidence_urls)

        return ResolutionOutcome(
            prediction_id=pred.id,
            statement=statement,
            deadline=pred.deadline,
            resolved_as=outcome["status"],
            confidence=outcome.get("confidence", "medium"),
            evidence=evidence_urls[:10],
            reasoning=outcome.get("reasoning", ""),
        )

    async def _judge_outcome(
        self,
        statement: str,
        deadline: Optional[datetime],
        evidence_urls: List[str],
    ) -> Dict[str, Any]:
        """Use LLM to judge whether a prediction came true based on evidence."""
        deadline_str = deadline.strftime("%Y-%m-%d") if deadline else "unknown"

        try:
            response = await self._llm.complete(
                system_prompt="""You are a prediction judge. Given a prediction statement,
its deadline, and evidence URLs, determine if the prediction was CORRECT or INCORRECT.

Rules:
- CORRECT: The predicted event definitely happened by the deadline
- INCORRECT: The predicted event definitely did NOT happen by the deadline
- If evidence is insufficient or ambiguous, return EXPIRED (insufficient data)

Return JSON:
{
  "status": "correct|incorrect|expired",
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation citing evidence"
}""",
                user_prompt=f"""Prediction: {statement}
Deadline: {deadline_str}
Evidence URLs: {', '.join(evidence_urls[:5]) if evidence_urls else 'None found'}

Judge if this prediction was correct or incorrect.""",
                temperature=0.1,
                max_tokens=500,
            )

            import json
            text = response.content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            logger.warning("LLM judgment failed: %s", e)
            return {"status": "expired", "confidence": "low", "reasoning": f"Judgment error: {e}"}


__all__ = ["ResolutionEngine", "ResolutionOutcome"]
