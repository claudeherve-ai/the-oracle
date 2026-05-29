"""Tests for signal extraction."""

import json
import pytest
from datetime import datetime, timezone

from oracle.llm import MockProvider
from oracle.models.prediction import Signal
from oracle.signals.extractor import SignalExtractor, ExtractedEntity, EmergingPattern


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def make_signal(
    content: str,
    source: str = "hackernews",
    entities: list | None = None,
    sentiment: float = 0.0,
    relevance: float = 0.5,
) -> Signal:
    return Signal(
        source=source,
        content=content,
        entities=entities or [],
        sentiment=sentiment,
        relevance=relevance,
    )


def make_llm_response(signals_count: int) -> str:
    """Build a realistic LLM extraction response."""
    items = []
    for i in range(signals_count):
        items.append({
            "entities": [
                {"name": f"Company{i}", "type": "company"},
                {"name": "AI Chip", "type": "product"},
            ],
            "sentiment": 0.3 + i * 0.1,
            "relevance": 0.7,
            "keywords": ["AI", "launch", "chip"],
            "category_hints": ["product_launch"],
        })
    return json.dumps(items)


# ---------------------------------------------------------------------------
# Tests — basic extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_empty_signals():
    """Extracting empty signals returns empty list."""
    provider = MockProvider()
    extractor = SignalExtractor(provider)
    result = await extractor.extract([])
    assert result == []


@pytest.mark.asyncio
async def test_extract_single_signal():
    """Extract entities and sentiment from a single signal."""
    provider = MockProvider()
    provider.set_response(json.dumps([{
        "entities": [
            {"name": "OpenAI", "type": "company"},
            {"name": "Sam Altman", "type": "person"},
            {"name": "GPT-5", "type": "product"},
        ],
        "sentiment": 0.6,
        "relevance": 0.9,
        "keywords": ["GPT-5", "OpenAI", "launch"],
        "category_hints": ["product_launch", "tech_trend"],
    }]))

    extractor = SignalExtractor(provider)
    signal = make_signal("OpenAI is expected to launch GPT-5 later this year.")
    result = await extractor.extract([signal])

    assert len(result) == 1
    enriched = result[0]
    assert "OpenAI" in enriched.entities
    assert "Sam Altman" in enriched.entities
    assert "GPT-5" in enriched.entities
    assert enriched.sentiment == 0.6
    assert enriched.relevance == 0.9
    assert enriched.metadata["keywords"] == ["GPT-5", "OpenAI", "launch"]
    assert "product_launch" in enriched.metadata["category_hints"]


@pytest.mark.asyncio
async def test_extract_batch():
    """Extract from multiple signals in a batch."""
    provider = MockProvider()
    provider.set_response(make_llm_response(3))

    extractor = SignalExtractor(provider)
    signals = [
        make_signal(f"Signal {i} about Company{i} launching AI chip.")
        for i in range(3)
    ]
    result = await extractor.extract(signals)

    assert len(result) == 3
    for enriched in result:
        assert len(enriched.entities) == 2
        assert enriched.sentiment != 0.0
        assert enriched.relevance == 0.7
        assert enriched.metadata["anomaly_score"] >= 0.0


@pytest.mark.asyncio
async def test_extract_handles_malformed_llm_response():
    """Fallback gracefully when LLM returns invalid JSON."""
    provider = MockProvider()
    provider.set_response("This is not valid JSON at all!")

    extractor = SignalExtractor(provider)
    signals = [make_signal("Some signal")]
    result = await extractor.extract(signals)

    # Should return signals without crashing — entities/sentiment may be defaults
    assert len(result) == 1
    assert isinstance(result[0], Signal)


@pytest.mark.asyncio
async def test_extract_markdown_wrapped_json():
    """Parse JSON wrapped in markdown code fences."""
    response_content = """```json
[
    {
        "entities": [{"name": "Apple", "type": "company"}],
        "sentiment": 0.5,
        "relevance": 0.8,
        "keywords": ["iPhone", "launch"],
        "category_hints": ["product_launch"]
    }
]
```"""
    provider = MockProvider()
    provider.set_response(response_content)

    extractor = SignalExtractor(provider)
    signals = [make_signal("Apple is launching a new iPhone")]
    result = await extractor.extract(signals)

    assert len(result) == 1
    assert "Apple" in result[0].entities
    assert result[0].sentiment == 0.5


# ---------------------------------------------------------------------------
# Tests — pattern detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pattern_new_entity():
    """Detect when a new entity appears."""
    provider = MockProvider()
    provider.set_response(json.dumps([{
        "entities": [{"name": "NewStartup", "type": "company"}],
        "sentiment": 0.7,
        "relevance": 0.9,
        "keywords": ["startup", "funding"],
        "category_hints": ["startup_success"],
    }]))

    extractor = SignalExtractor(provider)  # No history — everything is new
    signals = [make_signal("NewStartup raises $100M Series A")]
    result = await extractor.extract(signals)

    patterns = result[0].metadata.get("patterns_detected", [])
    assert any("new_entity" in p.lower() or "new entity" in p.lower() for p in result[0].metadata.get("all_patterns", []))


@pytest.mark.asyncio
async def test_pattern_mention_spike():
    """Detect when entity mentions spike."""
    provider = MockProvider()

    def _response_for_n(n: int) -> str:
        items = []
        for i in range(n):
            items.append({
                "entities": [{"name": "SpikingCorp", "type": "company"}],
                "sentiment": 0.5,
                "relevance": 0.8,
                "keywords": ["spike"],
                "category_hints": ["tech_trend"],
            })
        return json.dumps(items)

    provider.set_response(
        _response_for_n(1),  # First extract: 1 signal
        _response_for_n(3),  # Second extract: 3 signals
    )

    extractor = SignalExtractor(provider)

    # First batch: establish normal baseline (1 mention)
    await extractor.extract([make_signal("SpikingCorp report")])

    # Second batch: 3 mentions = spike
    signals = [
        make_signal("SpikingCorp announcement 1"),
        make_signal("SpikingCorp announcement 2"),
        make_signal("SpikingCorp announcement 3"),
    ]
    result = await extractor.extract(signals)

    all_patterns = []
    for sig in result:
        all_patterns.extend(sig.metadata.get("all_patterns", []))

    assert any("spike" in p.lower() for p in all_patterns)


# ---------------------------------------------------------------------------
# Tests — anomaly scoring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anomaly_scoring():
    """New entities get higher anomaly scores."""
    provider = MockProvider()
    provider.set_response(json.dumps([{
        "entities": [{"name": "MysteriousNewCorp", "type": "company"}],
        "sentiment": 0.0,
        "relevance": 0.9,
        "keywords": ["mystery", "new"],
        "category_hints": ["tech_trend"],
    }]))

    extractor = SignalExtractor(provider)  # Clean history
    signals = [make_signal("MysteriousNewCorp emerges from stealth")]
    result = await extractor.extract(signals)

    assert result[0].metadata["anomaly_score"] > 0.0


# ---------------------------------------------------------------------------
# Tests — history management
# ---------------------------------------------------------------------------

def test_reset_history():
    """Reset clears entity tracking."""
    provider = MockProvider()
    provider.set_response(make_llm_response(2))

    extractor = SignalExtractor(provider)
    extractor._entity_history["known_entity"] = 5
    extractor._sentiment_history["known_entity"] = [0.1, 0.2, 0.3]

    extractor.reset_history()

    assert len(extractor._entity_history) == 0
    assert len(extractor._sentiment_history) == 0
