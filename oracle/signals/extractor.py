"""Signal extraction engine for The Oracle.

Takes raw signals from ingestion sources and enriches them with
entities, sentiment analysis, anomaly scores, and emerging pattern detection.

All LLM calls go through the injected LLMProvider — fully testable.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from oracle.llm import LLMProvider, LLMResponse
from oracle.models.prediction import Signal

logger = logging.getLogger("oracle.signals.extractor")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ExtractedEntity:
    """A named entity found in a signal."""

    name: str
    type: str  # "company", "person", "product", "technology", "event", "other"

    def __init__(self, name: str, type: str = "other"):
        self.name = name
        self.type = type

    def __repr__(self) -> str:
        return f"ExtractedEntity({self.name!r}, {self.type})"


class EmergingPattern:
    """An emerging pattern detected across signals."""

    name: str
    type: str  # "mention_spike", "sentiment_shift", "new_entity", "cross_source_burst"
    entity: str
    description: str
    confidence: float  # 0.0 - 1.0

    def __init__(
        self,
        name: str,
        type: str,
        entity: str,
        description: str,
        confidence: float = 0.5,
    ):
        self.name = name
        self.type = type
        self.entity = entity
        self.description = description
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"EmergingPattern({self.name!r}, type={self.type}, "
            f"entity={self.entity!r}, confidence={self.confidence:.2f})"
        )


class ExtractionResult:
    """Result of extracting from a single signal."""

    entities: List[ExtractedEntity]
    sentiment: float  # -1.0 to 1.0
    anomaly_score: float  # 0.0 to 1.0
    relevance: float  # 0.0 to 1.0
    keywords: List[str]
    category_hints: List[str]

    def __init__(
        self,
        entities: Optional[List[ExtractedEntity]] = None,
        sentiment: float = 0.0,
        anomaly_score: float = 0.0,
        relevance: float = 0.5,
        keywords: Optional[List[str]] = None,
        category_hints: Optional[List[str]] = None,
    ):
        self.entities = entities or []
        self.sentiment = max(-1.0, min(1.0, sentiment))
        self.anomaly_score = max(0.0, min(1.0, anomaly_score))
        self.relevance = max(0.0, min(1.0, relevance))
        self.keywords = keywords or []
        self.category_hints = category_hints or []

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(entities={len(self.entities)}, "
            f"sentiment={self.sentiment:.2f}, anomaly={self.anomaly_score:.2f})"
        )


# ---------------------------------------------------------------------------
# Signal Extractor
# ---------------------------------------------------------------------------


class SignalExtractor:
    """Extracts entities, sentiment, anomalies, and patterns from signals.

    Usage:
        provider = OpenAIProvider()
        extractor = SignalExtractor(provider)
        enriched = await extractor.extract(signals)
    """

    # Batch size for LLM calls — combines multiple signals into one prompt
    LLM_BATCH_SIZE = 5

    def __init__(self, llm: LLMProvider):
        self._llm = llm
        self._entity_history: Dict[str, int] = {}  # entity -> count seen before
        self._sentiment_history: Dict[str, List[float]] = {}  # entity -> recent sentiments

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, signals: List[Signal]) -> List[Signal]:
        """Extract entities, sentiment, and metadata from a batch of signals.

        Returns enriched signals with populated entities, sentiment,
        relevance, anomaly scores, and extended metadata.
        """
        if not signals:
            return []

        # Step 1: LLM extraction (entity, sentiment, relevance) in batches
        all_results: List[ExtractionResult] = []
        for i in range(0, len(signals), self.LLM_BATCH_SIZE):
            batch = signals[i : i + self.LLM_BATCH_SIZE]
            batch_results = await self._extract_batch(batch)
            all_results.extend(batch_results)

        # Step 2: Compute anomaly scores and detect patterns
        patterns = self._detect_patterns(signals, all_results)

        # Step 3: Assign anomaly scores based on patterns
        self._assign_anomaly_scores(signals, all_results, patterns)

        # Step 4: Enrich signal objects
        enriched = self._apply_extraction(signals, all_results, patterns)

        # Step 5: Update history
        self._update_history(enriched)

        logger.info(
            "Extracted %d signals, found %d entities, %d patterns",
            len(enriched),
            sum(len(r.entities) for r in all_results),
            len(patterns),
        )
        return enriched

    async def extract_single(self, signal: Signal) -> Signal:
        """Extract from a single signal (convenience wrapper)."""
        results = await self.extract([signal])
        return results[0]

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def _extract_batch(self, signals: List[Signal]) -> List[ExtractionResult]:
        """Use LLM to extract entities, sentiment, and keywords from a batch."""
        user_prompt = self._build_extraction_prompt(signals)
        response = await self._llm.complete(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=min(4000, 800 * len(signals)),
        )
        return self._parse_extraction_response(response, len(signals))

    def _build_extraction_prompt(self, signals: List[Signal]) -> str:
        """Build the user prompt for entity/sentiment extraction."""
        lines = ["Extract entities, sentiment, and keywords from these signals:\n"]
        for idx, sig in enumerate(signals):
            lines.append(f"--- Signal {idx} (source: {sig.source}) ---")
            lines.append(sig.content[:1200])  # Truncate long content
            lines.append("")
        return "\n".join(lines)

    def _parse_extraction_response(
        self, response: LLMResponse, expected_count: int
    ) -> List[ExtractionResult]:
        """Parse the JSON response from the LLM into ExtractionResults."""
        content = response.content.strip()

        # Try to extract JSON from the response
        try:
            # Find JSON array in response (handle markdown code fences)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM extraction response as JSON, falling back")
            return [ExtractionResult() for _ in range(expected_count)]

        results: List[ExtractionResult] = []
        items = data if isinstance(data, list) else data.get("signals", [])

        for item in items:
            entities = [
                ExtractedEntity(name=e["name"], type=e.get("type", "other"))
                for e in item.get("entities", [])
                if isinstance(e, dict) and "name" in e
            ]
            sentiment = float(item.get("sentiment", 0.0))
            relevance = float(item.get("relevance", 0.5))
            keywords = item.get("keywords", [])
            category_hints = item.get("category_hints", [])

            results.append(
                ExtractionResult(
                    entities=entities,
                    sentiment=sentiment,
                    relevance=relevance,
                    keywords=keywords,
                    category_hints=category_hints,
                )
            )

        # Pad if we got fewer results than expected
        while len(results) < expected_count:
            results.append(ExtractionResult())

        return results[:expected_count]

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def _detect_patterns(
        self,
        signals: List[Signal],
        results: List[ExtractionResult],
    ) -> List[EmergingPattern]:
        """Detect emerging patterns across the signal batch.

        Detects:
        - mention_spike: Entity appears much more than historically
        - sentiment_shift: Entity sentiment changed significantly
        - new_entity: Entity not seen before
        - cross_source_burst: Entity appears across multiple sources
        """
        patterns: List[EmergingPattern] = []

        # Count entity occurrences in this batch
        entity_counts: Counter = Counter()
        entity_sources: Dict[str, Set[str]] = {}
        entity_sentiments: Dict[str, List[float]] = {}

        for sig, result in zip(signals, results):
            for entity in result.entities:
                key = entity.name.lower()
                entity_counts[key] += 1
                if key not in entity_sources:
                    entity_sources[key] = set()
                entity_sources[key].add(sig.source)
                if key not in entity_sentiments:
                    entity_sentiments[key] = []
                entity_sentiments[key].append(result.sentiment)

        total_signals = len(signals)
        if total_signals == 0:
            return []

        for entity_key, count in entity_counts.most_common():
            entity_name = entity_key
            proportion = count / total_signals

            # Pattern: New entity
            if entity_key not in self._entity_history:
                patterns.append(
                    EmergingPattern(
                        name=f"New entity: {entity_name}",
                        type="new_entity",
                        entity=entity_name,
                        description=f"{entity_name} appeared for the first time in this batch",
                        confidence=0.8 if proportion > 0.1 else 0.5,
                    )
                )
                continue

            # Pattern: Mention spike
            historical_count = self._entity_history.get(entity_key, 0)
            if count >= 3 and count > historical_count * 2:
                confidence = min(0.9, 0.5 + 0.1 * (count - historical_count))
                patterns.append(
                    EmergingPattern(
                        name=f"Mention spike: {entity_name}",
                        type="mention_spike",
                        entity=entity_name,
                        description=(
                            f"{entity_name} mentioned {count} times "
                            f"(up from ~{historical_count} historically)"
                        ),
                        confidence=confidence,
                    )
                )

            # Pattern: Sentiment shift
            if entity_key in self._sentiment_history and len(self._sentiment_history[entity_key]) >= 3:
                historical = self._sentiment_history[entity_key]
                avg_historical = sum(historical) / len(historical)
                if entity_key in entity_sentiments:
                    current_sentiments = entity_sentiments[entity_key]
                    avg_current = sum(current_sentiments) / len(current_sentiments)
                    shift = abs(avg_current - avg_historical)
                    if shift > 0.4:
                        direction = "positive" if avg_current > avg_historical else "negative"
                        patterns.append(
                            EmergingPattern(
                                name=f"Sentiment shift: {entity_name}",
                                type="sentiment_shift",
                                entity=entity_name,
                                description=(
                                    f"{entity_name} sentiment shifted {direction} "
                                    f"({avg_historical:.2f} -> {avg_current:.2f})"
                                ),
                                confidence=min(0.95, 0.5 + shift),
                            )
                        )

            # Pattern: Cross-source burst
            sources = entity_sources.get(entity_key, set())
            if len(sources) >= 3 and count >= 2:
                patterns.append(
                    EmergingPattern(
                        name=f"Cross-source burst: {entity_name}",
                        type="cross_source_burst",
                        entity=entity_name,
                        description=(
                            f"{entity_name} appears across {len(sources)} sources "
                            f"({', '.join(sorted(sources))})"
                        ),
                        confidence=min(0.9, 0.4 + 0.15 * len(sources)),
                    )
                )

        # Deduplicate and sort by confidence
        seen: Set[Tuple[str, str]] = set()
        deduped: List[EmergingPattern] = []
        for p in patterns:
            key = (p.type, p.entity)
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        deduped.sort(key=lambda p: p.confidence, reverse=True)
        return deduped

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def _assign_anomaly_scores(
        self,
        signals: List[Signal],
        results: List[ExtractionResult],
        patterns: List[EmergingPattern],
    ) -> None:
        """Assign anomaly scores to signals based on detected patterns."""
        # Build a map of entity -> pattern type
        entity_pattern_map: Dict[str, float] = {}
        for p in patterns:
            key = p.entity.lower()
            entity_pattern_map[key] = max(entity_pattern_map.get(key, 0.0), p.confidence)

        for result in results:
            if not result.entities:
                result.anomaly_score = 0.0
                continue

            # Anomaly score = max pattern confidence for any entity in this signal
            scores = [
                entity_pattern_map.get(e.name.lower(), 0.0)
                for e in result.entities
            ]
            result.anomaly_score = max(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Apply extraction to signals
    # ------------------------------------------------------------------

    def _apply_extraction(
        self,
        signals: List[Signal],
        results: List[ExtractionResult],
        patterns: List[EmergingPattern],
    ) -> List[Signal]:
        """Apply extracted data back onto the Signal objects."""
        enriched: List[Signal] = []

        pattern_names = [p.name for p in patterns]
        pattern_summary = "\n".join(f"- {p}" for p in patterns)

        for sig, result in zip(signals, results):
            entities = [e.name for e in result.entities]
            entity_types = {e.name: e.type for e in result.entities}

            # Determine which patterns this signal triggered
            signal_patterns = [
                p.name
                for p in patterns
                if any(p.entity.lower() == e.name.lower() for e in result.entities)
            ]

            enriched_sig = Signal(
                id=sig.id,
                source=sig.source,
                content=sig.content,
                entities=entities,
                sentiment=result.sentiment,
                relevance=result.relevance,
                metadata={
                    **sig.metadata,
                    "extracted_entity_types": entity_types,
                    "keywords": result.keywords,
                    "category_hints": result.category_hints,
                    "anomaly_score": result.anomaly_score,
                    "patterns_detected": signal_patterns,
                    "all_patterns": pattern_names,
                    "pattern_summary": pattern_summary,
                },
                captured_at=sig.captured_at,
            )
            enriched.append(enriched_sig)

        return enriched

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _update_history(self, signals: List[Signal]) -> None:
        """Update internal entity history from enriched signals."""
        for sig in signals:
            for entity in sig.entities:
                key = entity.lower()
                self._entity_history[key] = self._entity_history.get(key, 0) + 1
                if key not in self._sentiment_history:
                    self._sentiment_history[key] = []
                self._sentiment_history[key].append(sig.sentiment)
                # Keep only last 20 sentiments
                if len(self._sentiment_history[key]) > 20:
                    self._sentiment_history[key] = self._sentiment_history[key][-20:]

    def reset_history(self) -> None:
        """Clear entity history. Useful for testing."""
        self._entity_history.clear()
        self._sentiment_history.clear()


# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a signal extraction engine for The Oracle, a predictive intelligence system.

Your job: analyze signals (news snippets, social media posts, tech discussions) and extract structured information.

For each signal, extract:
1. **entities**: Companies, people, products, technologies mentioned. Each with "name" and "type" (company, person, product, technology, event, other).
2. **sentiment**: Score from -1.0 (very negative) to 1.0 (very positive). Use 0.0 for neutral/factual.
3. **relevance**: Score from 0.0 to 1.0 for how relevant this is to making predictions about tech, markets, or products.
4. **keywords**: 3-5 key terms that capture the topic.
5. **category_hints**: Which prediction categories this might relate to: tech_trend, product_launch, market_move, regulatory, startup_success, culture, github_trend.

Output ONLY valid JSON. Format:
[
  {
    "entities": [{"name": "Apple", "type": "company"}],
    "sentiment": 0.3,
    "relevance": 0.7,
    "keywords": ["AI", "launch", "chip"],
    "category_hints": ["product_launch", "tech_trend"]
  }
]

Be precise. Only extract entities explicitly mentioned. Be conservative with sentiment — default to 0.0 for factual reporting."""


__all__ = [
    "SignalExtractor",
    "ExtractedEntity",
    "EmergingPattern",
    "ExtractionResult",
]
