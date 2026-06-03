"""Prediction engine for The Oracle.

Generates specific, verifiable, time-bound predictions from enriched signals.
Every prediction has: statement, confidence, reasoning, deadline, sources.

Uses LLMProvider for generation — fully testable with MockProvider.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from oracle.tools import research_topic, format_context_for_prompt, multi_source_grounding
from oracle.tools.structured import ParsedBatch, parse_prediction_batch

from oracle.audit import (
    AuditRecord,
    AuditSink,
    EvidenceSpan,
    PredictionAudit,
)
from oracle.llm import LLMProvider, LLMResponse
from oracle.calibration.tracker import ConfidenceContextualizer
from oracle.models.prediction import (
    Category,
    Prediction,
    Signal,
    Status,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from oracle.prediction.verifier import SourceEvidence, VerificationEngine, VerificationResult

logger = logging.getLogger("oracle.prediction.engine")

#: Verifier verdicts that, on their own, signal the evidence is too weak or too
#: conflicted to stand behind a forecast. When the abstain policy is enabled,
#: any prediction landing on one of these verdicts is converted into an explicit
#: ``INSUFFICIENT_EVIDENCE`` abstention rather than presented as a confident call.
_WEAK_VERDICTS = frozenset(
    {"contradicted", "insufficient_corroboration", "mixed_contradicting", "unverifiable"}
)

# ---------------------------------------------------------------------------
# Core prediction engine
# ---------------------------------------------------------------------------


class PredictionEngine:
    """Generates calibrated predictions from signals using an LLM.

    Usage:
        provider = OpenAIProvider()
        engine = PredictionEngine(provider)
        predictions = await engine.generate(signals)
        # Or with a specific question:
        predictions = await engine.generate(signals, question="Will Apple launch AR glasses?")
    """

    # Default prediction timeout: if no deadline specified, use this offset
    DEFAULT_DEADLINE_DAYS = 30

    #: Verification modes for ``generate()``.
    #:   "off"  — no external verification (default; what tests/MockProvider use).
    #:   "live" — run the canonical multi-source NLI verifier (real network).
    _VALID_VERIFICATION_MODES = ("off", "live")

    def __init__(
        self,
        llm: LLMProvider,
        *,
        verifier: "Optional[VerificationEngine]" = None,
        verification_mode: str = "off",
        abstain_on_weak_evidence: bool = False,
        abstain_threshold: float = 0.35,
        audit_sink: Optional[AuditSink] = None,
    ):
        """Create a prediction engine.

        Args:
            llm: The LLM provider used for generation.
            verifier: Optional canonical :class:`VerificationEngine`. When
                ``verification_mode == "live"`` and this is ``None``, one is
                lazily constructed around ``llm``. Inject a stub in tests.
            verification_mode: ``"off"`` (default) disables external
                verification entirely — this is what unit tests and
                ``MockProvider`` rely on, so no network is ever touched.
                ``"live"`` enables the canonical multi-source, quote-verified
                NLI verifier and lets it adjust confidence based on real,
                corroborated evidence.
            abstain_on_weak_evidence: When ``True`` (and verification is live),
                a prediction whose verdict is weak/contradictory, or whose
                verified confidence falls below ``abstain_threshold``, is
                converted into an explicit ``INSUFFICIENT_EVIDENCE`` abstention
                instead of being surfaced as a confident forecast. This is the
                system's "I don't know" path — a forecaster that refuses bad
                questions is more trustworthy than one that always answers.
                Defaults to ``False`` so existing behaviour is unchanged.
            abstain_threshold: Verified-confidence floor (inclusive lower bound
                is *not* abstained; strictly below abstains) used by the abstain
                policy. Ignored unless ``abstain_on_weak_evidence`` is enabled.
            audit_sink: Optional :class:`~oracle.audit.AuditSink`. When provided,
                every ``generate()`` call emits a structured
                :class:`~oracle.audit.AuditRecord` capturing the prompts, model,
                sources fetched, evidence spans, and confidence *before and
                after* verification — a replayable, auditable trail.
                Reproducibility is trust. ``None`` (default) means zero overhead
                and unchanged behaviour.
        """
        if verification_mode not in self._VALID_VERIFICATION_MODES:
            raise ValueError(
                f"verification_mode must be one of {self._VALID_VERIFICATION_MODES}, "
                f"got {verification_mode!r}"
            )
        if not 0.0 <= abstain_threshold <= 1.0:
            raise ValueError(
                f"abstain_threshold must be in [0.0, 1.0], got {abstain_threshold!r}"
            )
        self._llm = llm
        self._verifier = verifier
        self._verification_mode = verification_mode
        self._abstain_on_weak_evidence = abstain_on_weak_evidence
        self._abstain_threshold = abstain_threshold
        self._audit_sink = audit_sink

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        signals: List[Signal],
        *,
        question: Optional[str] = None,
        categories: Optional[List[Category]] = None,
        max_predictions: int = 5,
        extra_context: Optional[str] = None,
    ) -> List[Prediction]:
        """Generate predictions from signals.

        Args:
            signals: Enriched signals (ideally from SignalExtractor).
            question: Optional specific question to answer.
            categories: Optional filter — only generate for these categories.
            max_predictions: Maximum number of predictions to generate.
            extra_context: Additional context to include in the prompt.

        Returns:
            List of Prediction objects.
        """
        if not signals and not question:
            logger.warning("generate() called with empty signals and no question")
            return []

        # Step 0: Multi-source grounding — MCP + web for anti-hallucination
        web_grounding = ""
        if question:
            # Use multi-source grounding (MCP servers + web search)
            web_grounding = await multi_source_grounding(question)
            if not web_grounding:
                # Fallback to basic web research
                web_context = await research_topic(question)
                web_grounding = format_context_for_prompt(web_context)
            if web_grounding:
                logger.info("Predictions grounded with multi-source data for: %s",
                           question[:80])

        # Build the prompt
        system_prompt = _build_system_prompt(categories)
        combined_context = extra_context or ""
        if web_grounding:
            combined_context = web_grounding + ("\n\n" + combined_context if combined_context else "")
        user_prompt = _build_user_prompt(
            signals,
            question=question,
            categories=categories,
            max_predictions=max_predictions,
            extra_context=combined_context or None,
        )

        # Call LLM
        logger.info(
            "Generating predictions from %d signals (question=%s, max=%d)",
            len(signals),
            question[:80] if question else "auto",
            max_predictions,
        )
        response = await self._llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=max(4000, 800 * max_predictions),  # More tokens for grounded responses
        )

        # Check for empty response (Azure deployment name mismatch)
        if not response.content.strip():
            msg = (
                f"LLM returned empty response. Model: {self._llm.model_name}. "
                "Check AZURE_OPENAI_DEPLOYMENT env var matches your Azure Foundry deployment name."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # Parse and validate through the Pydantic structured-output boundary (A3).
        parsed = self._parse_predictions(response)
        predictions = self._validate_predictions(parsed, max_predictions)

        # Snapshot pre-verification confidence for the audit trail (D14): we want
        # to record how much verification *moved* each number, including drops.
        confidence_pre = [p.confidence for p in predictions]

        # Canonical verification path (A4): the multi-source, quote-verified NLI
        # verifier is the ONE source of evidence-driven confidence adjustment.
        # Gated behind verification_mode == "live" so default/test runs touch no
        # network and behaviour is unchanged unless explicitly enabled.
        verification_results: List["Optional[VerificationResult]"] = [None] * len(predictions)
        if predictions and self._verification_mode == "live":
            verifier = self._get_verifier()
            results = await verifier.verify(predictions)
            for idx, (p, r) in enumerate(zip(predictions, results)):
                verification_results[idx] = r
                p.confidence = max(0.01, min(0.99, r.adjusted_confidence))
                if r.summary:
                    p.reasoning = (p.reasoning or "") + f" [Verified: {r.summary}]"
                # Abstain path (C10): when the evidence is too weak/conflicted to
                # stand behind, refuse to forecast rather than emit a confident
                # number. Opt-in so default behaviour is unchanged.
                if self._abstain_on_weak_evidence and self._should_abstain(r):
                    p.status = Status.INSUFFICIENT_EVIDENCE
                    verdict = getattr(r, "verdict", "unverifiable")
                    p.reasoning = (p.reasoning or "") + (
                        f" [Abstained: insufficient evidence — verdict={verdict}, "
                        f"verified_confidence={p.confidence:.2f} "
                        f"(threshold={self._abstain_threshold:.2f})]"
                    )

        # Per-prediction audit record (D14): emitted only when a sink is wired in,
        # so the default/test path is zero-overhead and behaviourally unchanged.
        if self._audit_sink is not None:
            self._emit_audit_record(
                question=question,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                web_grounding=web_grounding,
                predictions=predictions,
                confidence_pre=confidence_pre,
                verification_results=verification_results,
            )

        logger.info("Generated %d predictions", len(predictions))
        return predictions

    def contextualize(
        self,
        predictions: List[Prediction],
        history: List[Prediction],
        *,
        min_samples: int = 5,
    ) -> List[Prediction]:
        """Attach a historical track record to each prediction's confidence (E17).

        Never show a number without its track record. Given the resolved
        ``history`` (audited, scored predictions), this populates each
        prediction's ``track_record`` with how often the system has actually
        been right for the same category at the same confidence band — turning a
        bare ``0.70`` into an auditable "70% confident, and here's the receipts".

        Pure, synchronous, no network. Mutates and returns ``predictions``.

        Args:
            predictions: Fresh predictions to contextualize (mutated in place).
            history: Resolved predictions to compute the track record from.
            min_samples: Minimum resolved samples in a bucket before its accuracy
                is treated as *proven* rather than provisional.
        """
        contextualizer = ConfidenceContextualizer(min_samples=min_samples)
        return contextualizer.contextualize(predictions, history)

    # ------------------------------------------------------------------
    # Audit trail (D14)
    # ------------------------------------------------------------------

    def _emit_audit_record(
        self,
        *,
        question: Optional[str],
        system_prompt: str,
        user_prompt: str,
        response: LLMResponse,
        web_grounding: str,
        predictions: List[Prediction],
        confidence_pre: List[float],
        verification_results: "List[Optional[VerificationResult]]",
    ) -> None:
        """Build and hand a structured :class:`AuditRecord` to the sink.

        Failures here must never break generation — auditing is observability,
        not a hard dependency of the result — so the whole body is defensive.
        """
        try:
            audited: List[PredictionAudit] = []
            for idx, pred in enumerate(predictions):
                pre = confidence_pre[idx] if idx < len(confidence_pre) else pred.confidence
                result = verification_results[idx] if idx < len(verification_results) else None
                spans: List[EvidenceSpan] = []
                verdict: Optional[str] = None
                if result is not None:
                    verdict = getattr(result, "verdict", None)
                    for ev in getattr(result, "evidence", []) or []:
                        spans.append(
                            EvidenceSpan(
                                url=getattr(ev, "url", "") or "",
                                quote=getattr(ev, "quote", "") or "",
                                stance=getattr(ev, "stance", "neutral") or "neutral",
                                credibility=float(getattr(ev, "credibility", 0.5) or 0.5),
                            )
                        )
                audited.append(
                    PredictionAudit(
                        prediction_id=getattr(pred, "id", "") or "",
                        statement=pred.statement,
                        category=getattr(pred.category, "value", str(pred.category)),
                        status=getattr(pred.status, "value", str(pred.status)),
                        confidence_pre=float(pre),
                        confidence_post=float(pred.confidence),
                        verdict=verdict,
                        sources=list(getattr(pred, "sources", []) or []),
                        evidence_spans=spans,
                    )
                )

            record = AuditRecord(
                model=getattr(self._llm, "model_name", "unknown"),
                question=question,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=response.content,
                verification_mode=self._verification_mode,
                grounding_present=bool(web_grounding),
                grounding_chars=len(web_grounding or ""),
                predictions=audited,
            )
            self._audit_sink.record(record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Audit record emission failed (non-fatal): %s", exc)

    def _should_abstain(self, result: Any) -> bool:
        """Decide whether a verified prediction should become an abstention.

        Abstain when the verifier's verdict is itself weak/contradictory, OR
        when the verified confidence falls strictly below ``abstain_threshold``.
        """
        verdict = getattr(result, "verdict", "unverifiable")
        if verdict in _WEAK_VERDICTS:
            return True
        adjusted = max(0.01, min(0.99, result.adjusted_confidence))
        return adjusted < self._abstain_threshold

    def _get_verifier(self) -> "VerificationEngine":
        """Return the canonical verifier, lazily building one if needed."""
        if self._verifier is None:
            from oracle.prediction.verifier import VerificationEngine

            self._verifier = VerificationEngine(self._llm)
        return self._verifier

    async def generate_from_question(
        self,
        question: str,
        signals: Optional[List[Signal]] = None,
        *,
        max_predictions: int = 5,
    ) -> List[Prediction]:
        """Generate predictions for a specific question.

        This is the primary user-facing method — users ask a question
        and get predictions back.

        Args:
            question: The user's question (e.g., "Will Apple release a new MacBook?")
            signals: Optional signals to ground predictions in current data.
            max_predictions: Max predictions to return.

        Returns:
            List of Prediction objects.
        """
        return await self.generate(
            signals or [],
            question=question,
            max_predictions=max_predictions,
        )

    async def scan(self, signals: List[Signal], max_predictions: int = 10) -> List[Prediction]:
        """Auto-scan signals and generate predictions without a specific question.

        The LLM identifies the most interesting trends and generates
        predictions from them.
        """
        return await self.generate(
            signals,
            question=None,
            max_predictions=max_predictions,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_predictions(self, response: LLMResponse) -> ParsedBatch:
        """Parse the LLM response into a validated :class:`ParsedBatch`.

        Delegates to the structured-output boundary which validates every
        candidate against the :class:`PredictionDraft` schema. Items that fail
        validation are *rejected with a reason* (logged + surfaced on the
        batch), never silently dropped, and a non-JSON payload yields a
        ``parse_error`` rather than a confusing empty list.
        """
        batch = parse_prediction_batch(response.content)

        if not batch.ok:
            logger.warning(
                "Failed to parse LLM response (%s): %s",
                batch.parse_error,
                response.content[:200],
            )
        elif batch.rejected:
            logger.warning(
                "Rejected %d malformed prediction(s) at the structured boundary: %s",
                batch.rejection_count,
                "; ".join(f"[{r.index}] {r.reason}" for r in batch.rejected),
            )

        return batch

    def _validate_predictions(
        self,
        batch: ParsedBatch,
        max_predictions: int,
    ) -> List[Prediction]:
        """Convert validated drafts into domain :class:`Prediction` objects.

        The structured boundary has already enforced the schema, so each draft
        is fed to ``_make_prediction`` (which still applies domain logic:
        confidence clamping, category fallback, deadline parsing, source
        normalisation).
        """
        predictions: List[Prediction] = []

        for draft in batch.valid[:max_predictions]:
            item = draft.model_dump()
            try:
                pred = self._make_prediction(item)
                if pred:
                    predictions.append(pred)
            except Exception as exc:
                logger.warning("Skipping invalid prediction: %s — %s", item.get("statement", ""), exc)

        return predictions

    def _make_prediction(self, item: Dict[str, Any]) -> Optional[Prediction]:
        """Create a Prediction from a parsed dict. Returns None if invalid."""
        statement = str(item.get("statement", "")).strip()
        if len(statement) < 10 or len(statement) > 1000:
            logger.debug("Prediction statement too short or long: %d chars", len(statement))
            return None

        # Parse category
        category_str = str(item.get("category", "tech_trend")).lower()
        try:
            category = Category(category_str)
        except ValueError:
            category = Category.TECH_TREND

        # Parse confidence — enforce 1-99% range (never 0% or 100%)
        confidence = float(item.get("confidence", 0.5))
        if confidence <= 0.01:
            confidence = 0.01
        elif confidence >= 0.99:
            confidence = 0.99
        confidence = max(0.01, min(0.99, confidence))

        # Parse deadline
        deadline = self._parse_deadline(item.get("deadline", ""))

        # Reasoning
        reasoning = str(item.get("reasoning", "")).strip()

        # Sources
        sources = item.get("sources", [])
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",") if s.strip()]
        sources = [str(s) for s in sources if s]

        return Prediction(
            category=category,
            statement=statement,
            confidence=confidence,
            reasoning=reasoning,
            sources=sources,
            deadline=deadline,
        )

    def _parse_deadline(self, deadline_raw: str) -> Optional[datetime]:
        """Parse a deadline string into a datetime.

        Supports:
        - ISO format: "2026-06-15"
        - Relative: "in 14 days", "in 1 month", "3 months"
        - Quarter: "Q3 2026", "Q4 2026"
        - Named months: "June 2026", "by July 15, 2026"
        """
        if not deadline_raw or str(deadline_raw).strip() in ("null", "none", ""):
            return datetime.now(timezone.utc) + timedelta(days=self.DEFAULT_DEADLINE_DAYS)

        deadline_str = str(deadline_raw).strip()

        # ISO date: "2026-06-15"
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", deadline_str)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)

        # ISO datetime: "2026-06-15T00:00:00Z"
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})T", deadline_str)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)

        # Relative: "in N days/weeks/months"
        m = re.match(r"in\s+(\d+)\s+(day|week|month)s?", deadline_str, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if unit == "day":
                return datetime.now(timezone.utc) + timedelta(days=num)
            elif unit == "week":
                return datetime.now(timezone.utc) + timedelta(weeks=num)
            elif unit == "month":
                return datetime.now(timezone.utc) + timedelta(days=num * 30)

        # Quarter: "Q3 2026"
        m = re.match(r"Q(\d)\s*(\d{4})", deadline_str, re.IGNORECASE)
        if m:
            quarter = int(m.group(1))
            year = int(m.group(2))
            month = (quarter - 1) * 3 + 1
            return datetime(year, month, 1, tzinfo=timezone.utc)

        # Month name: "June 2026", "by July 15, 2026"
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        for name, month_num in month_names.items():
            pattern = rf"\b{name}\b\s+(\d{{1,2}})[,\s]+(\d{{4}})|\b{name}\b\s+(\d{{4}})"
            m = re.search(pattern, deadline_str, re.IGNORECASE)
            if m:
                if m.group(3):  # "June 2026" — no day
                    day = 1
                    year = int(m.group(3))
                else:  # "June 15, 2026"
                    day = int(m.group(1))
                    year = int(m.group(2))
                return datetime(year, month_num, min(day, 28), tzinfo=timezone.utc)

        # Default fallback
        return datetime.now(timezone.utc) + timedelta(days=self.DEFAULT_DEADLINE_DAYS)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def _build_system_prompt(categories: Optional[List[Category]] = None) -> str:
    """Build the system prompt for prediction generation."""
    cat_list = ""
    if categories:
        cat_names = [c.value for c in categories]
        cat_list = f"\nFocus ONLY on these categories: {', '.join(cat_names)}"

    return f"""You are The Oracle — a predictive intelligence engine that makes SPECIFIC, VERIFIABLE, TIME-BOUND predictions.

Your predictions must follow these STRICT rules:

1. **SPECIFIC**: "X will happen by Y date" — never "X might happen" or "X could happen eventually"
   - GOOD: "Apple will announce a new MacBook Pro with M4 chip at WWDC in June 2026"
   - BAD: "Apple might release new hardware at some point"

2. **VERIFIABLE**: Every prediction must be objectively resolvable as correct or incorrect
   - GOOD: "NVDA stock will close above $150 on June 15, 2026"
   - BAD: "NVDA will continue to be a market leader"

3. **TIME-BOUND**: Every prediction MUST include a deadline
   - Use specific dates: "by June 15, 2026", "Q3 2026", "within 30 days"
   - Never use "eventually", "soon", "in the near future"

4. **CALIBRATED CONFIDENCE**: Use specific percentages
   - 50% = pure coin flip, you have no edge
   - 60-70% = you see some evidence pointing this way
   - 70-85% = strong evidence, but real uncertainty remains
   - 85-95% = very strong evidence, consensus expectation
   - NEVER use 100% or 0% — absolute certainty is impossible

5. **SOURCED**: Cite the signals, data, or patterns that support your prediction

6. **REASONED**: Explain WHY you made this prediction — what signals, what logic

Categories:{cat_list if cat_list else " tech_trend, product_launch, market_move, regulatory, startup_success, culture, github_trend"}

Output ONLY valid JSON. Format your response as:
{{
  "predictions": [
    {{
      "statement": "Specific, verifiable prediction with deadline",
      "category": "one_of_the_categories",
      "confidence": 0.75,
      "reasoning": "Why this prediction — cite specific signals and logic",
      "deadline": "2026-06-15",
      "sources": ["source_1_url_or_description", "source_2"]
    }}
  ]
}}

Do NOT include predictions you are not confident about. Quality over quantity."""


def _build_user_prompt(
    signals: List[Signal],
    *,
    question: Optional[str] = None,
    categories: Optional[List[Category]] = None,
    max_predictions: int = 5,
    extra_context: Optional[str] = None,
) -> str:
    """Build the user prompt with signal data and the question."""
    lines = []

    # Header
    if question:
        lines.append(f"# QUESTION\n{question}\n")
        lines.append(f"Generate at most {max_predictions} predictions that answer this question.\n")
    else:
        lines.append(f"# AUTO-SCAN\nAnalyze the signals below and generate at most {max_predictions} predictions about the most interesting emerging trends.\n")

    # Extra context
    if extra_context:
        lines.append(f"# ADDITIONAL CONTEXT\n{extra_context}\n")

    # Category constraint
    if categories:
        cat_names = [c.value for c in categories]
        lines.append(f"# CATEGORY CONSTRAINT\nOnly generate predictions in: {', '.join(cat_names)}\n")

    # Signals
    if signals:
        lines.append(f"# SIGNALS ({len(signals)} total)")
        lines.append("These are enriched signals extracted from news, social media, and tech discussions:\n")

        # Show top signals by relevance, plus pattern summaries
        sorted_signals = sorted(signals, key=lambda s: s.relevance, reverse=True)
        for i, sig in enumerate(sorted_signals[:30]):  # Cap at 30 signals
            anomaly = sig.metadata.get("anomaly_score", 0.0)
            patterns = sig.metadata.get("patterns_detected", [])
            keywords = sig.metadata.get("keywords", [])

            extra = ""
            if anomaly > 0.0:
                extra += f" [ANOMALY: {anomaly:.2f}]"
            if patterns:
                extra += f" [PATTERNS: {', '.join(patterns)}]"

            lines.append(f"\n[{i+1}] Source: {sig.source} | Sentiment: {sig.sentiment:.2f} | Relevance: {sig.relevance:.2f}{extra}")
            if sig.entities:
                lines.append(f"    Entities: {', '.join(sig.entities[:5])}")
            if keywords:
                lines.append(f"    Keywords: {', '.join(keywords[:5])}")
            content_preview = sig.content[:200]
            lines.append(f"    Content: {content_preview}{'...' if len(sig.content) > 200 else ''}")

        # Pattern summary
        all_patterns: List[str] = []
        for sig in sorted_signals:
            for p in sig.metadata.get("all_patterns", []):
                if p not in all_patterns:
                    all_patterns.append(p)

        if all_patterns:
            lines.append(f"\n# DETECTED PATTERNS ({len(all_patterns)})")
            for p in all_patterns[:10]:
                lines.append(f"- {p}")
    else:
        lines.append("# NO SIGNALS\nNo signals provided. Generate predictions based on your knowledge. Be conservative.\n")

    lines.append("\n# INSTRUCTIONS")
    lines.append(f"Generate at most {max_predictions} predictions.")
    lines.append("Each prediction must be specific, verifiable, time-bound, and calibrated.")
    lines.append("Output valid JSON only.")

    return "\n".join(lines)


__all__ = ["PredictionEngine"]
