"""Tests for prediction engine."""

import json
import pytest
from datetime import datetime, timezone, timedelta

from oracle.llm import MockProvider
from oracle.models.prediction import (
    Category,
    Prediction,
    Signal,
    Status,
)
from oracle.prediction.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(
    content: str = "Test signal",
    source: str = "hackernews",
    entities: list | None = None,
    sentiment: float = 0.0,
    relevance: float = 0.7,
    anomaly: float = 0.0,
    patterns: list | None = None,
    keywords: list | None = None,
) -> Signal:
    return Signal(
        source=source,
        content=content,
        entities=entities or [],
        sentiment=sentiment,
        relevance=relevance,
        metadata={
            "anomaly_score": anomaly,
            "patterns_detected": patterns or [],
            "keywords": keywords or [],
            "category_hints": [],
            "all_patterns": patterns or [],
        },
    )


def make_valid_prediction_response(count: int = 3) -> str:
    """Build a valid prediction JSON response."""
    preds = []
    for i in range(count):
        preds.append({
            "statement": f"Company{i} will announce Product{i} by Q4 2026",
            "category": "product_launch",
            "confidence": 0.55 + i * 0.1,
            "reasoning": f"Multiple signals indicate Company{i} is preparing a launch. "
                         f"Hiring activity up, domain registered, trademark filed.",
            "deadline": f"2026-{(10+i):02d}-01",
            "sources": [
                f"https://news.ycombinator.com/item?id={i}",
                f"https://techcrunch.com/article-{i}",
            ],
        })
    return json.dumps({"predictions": preds})


# ---------------------------------------------------------------------------
# Tests — basic generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_empty_signals():
    """Generating from empty signals returns empty list."""
    provider = MockProvider()
    engine = PredictionEngine(provider)
    result = await engine.generate([])
    assert result == []


@pytest.mark.asyncio
async def test_generate_returns_predictions():
    """Valid LLM response yields valid Prediction objects."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(3))

    engine = PredictionEngine(provider)
    signals = [
        make_signal("Company0 is hiring engineers"),
        make_signal("Company1 registered a trademark"),
        make_signal("Company2 domain WHOIS updated"),
    ]
    result = await engine.generate(signals)

    assert len(result) == 3
    for pred in result:
        assert isinstance(pred, Prediction)
        assert len(pred.statement) >= 10
        assert pred.confidence > 0.0
        assert pred.confidence < 1.0
        assert pred.deadline is not None
        assert pred.status == Status.PENDING


@pytest.mark.asyncio
async def test_generate_from_question():
    """Generate predictions from a specific question."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(2))

    engine = PredictionEngine(provider)
    result = await engine.generate_from_question(
        "Will Apple release AR glasses in 2026?",
        signals=[
            make_signal("Apple hires AR engineers", entities=["Apple"]),
            make_signal("Apple Vision Pro updates expected", entities=["Apple"]),
        ],
    )

    assert len(result) == 2
    # Verify the call recorded the question (may have 2 calls: generation + verification)
    assert len(provider.calls) >= 1
    assert "Will Apple release AR glasses" in provider.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_scan_mode():
    """Scan mode generates predictions without a question."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(5))

    engine = PredictionEngine(provider)
    signals = [make_signal(f"Signal {i}") for i in range(10)]
    result = await engine.scan(signals, max_predictions=5)

    assert len(result) <= 5
    # Verify no question in prompt
    assert "AUTO-SCAN" in provider.calls[0]["user_prompt"]


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confidence_bounds():
    """Confidence is forced into 1-99% range."""
    provider = MockProvider()
    provider.set_response(json.dumps({
        "predictions": [
            {
                "statement": "Something will definitely happen by June 2026",
                "category": "tech_trend",
                "confidence": 1.0,  # LLM said 100% — should be capped
                "reasoning": "Test",
                "deadline": "2026-06-01",
                "sources": [],
            },
            {
                "statement": "Something will definitely not happen by June 2026",
                "category": "tech_trend",
                "confidence": 0.0,  # LLM said 0% — should be raised
                "reasoning": "Test",
                "deadline": "2026-06-01",
                "sources": [],
            },
        ]
    }))

    engine = PredictionEngine(provider)
    result = await engine.generate([make_signal("test")])

    assert len(result) == 2
    assert result[0].confidence <= 0.99
    assert result[1].confidence >= 0.01


@pytest.mark.asyncio
async def test_short_statements_filtered():
    """Statements that are too short are filtered out."""
    provider = MockProvider()
    provider.set_response(json.dumps({
        "predictions": [
            {
                "statement": "Short",  # Too short (< 10 chars)
                "category": "tech_trend",
                "confidence": 0.7,
                "reasoning": "Test",
                "deadline": "2026-06-01",
                "sources": [],
            },
            {
                "statement": "This is a properly long prediction statement about a real thing happening",
                "category": "product_launch",
                "confidence": 0.65,
                "reasoning": "Test",
                "deadline": "2026-06-01",
                "sources": [],
            },
        ]
    }))

    engine = PredictionEngine(provider)
    result = await engine.generate([make_signal("test")])

    assert len(result) == 1
    assert "properly long" in result[0].statement


@pytest.mark.asyncio
async def test_malformed_json_handled():
    """Malformed LLM response doesn't crash."""
    provider = MockProvider()
    provider.set_response("I don't feel like making predictions today, sorry.")

    engine = PredictionEngine(provider)
    result = await engine.generate([make_signal("test")])

    assert result == []  # Should gracefully return empty


@pytest.mark.asyncio
async def test_markdown_wrapped_json():
    """Parse predictions from markdown-wrapped JSON."""
    response = """Here are my predictions:

```json
{
  "predictions": [
    {
      "statement": "Apple will announce M4 MacBook Pro at WWDC June 2026",
      "category": "product_launch",
      "confidence": 0.78,
      "reasoning": "Supply chain signals and historical pattern",
      "deadline": "2026-06-15",
      "sources": ["https://example.com/supply-chain"]
    }
  ]
}
```"""
    provider = MockProvider()
    provider.set_response(response)

    engine = PredictionEngine(provider)
    result = await engine.generate([make_signal("Apple supply chain")])

    assert len(result) == 1
    assert "M4 MacBook Pro" in result[0].statement
    assert result[0].confidence == 0.78


# ---------------------------------------------------------------------------
# Tests — category filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_category_filter():
    """Category filter is passed to the LLM prompt."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(2))

    engine = PredictionEngine(provider)
    result = await engine.generate(
        [make_signal("test")],
        categories=[Category.PRODUCT_LAUNCH, Category.MARKET_MOVE],
    )

    assert len(result) == 2
    # Check that the system prompt mentions the categories
    system = provider.calls[0]["system_prompt"]
    assert "product_launch" in system
    assert "market_move" in system


# ---------------------------------------------------------------------------
# Tests — deadline parsing
# ---------------------------------------------------------------------------

def test_parse_deadline_iso():
    engine = PredictionEngine(MockProvider())
    result = engine._parse_deadline("2026-06-15")
    assert result == datetime(2026, 6, 15, tzinfo=timezone.utc)


def test_parse_deadline_relative_days():
    engine = PredictionEngine(MockProvider())
    result = engine._parse_deadline("in 14 days")
    expected = datetime.now(timezone.utc) + timedelta(days=14)
    assert abs((result - expected).total_seconds()) < 60


def test_parse_deadline_quarter():
    engine = PredictionEngine(MockProvider())
    result = engine._parse_deadline("Q3 2026")
    assert result == datetime(2026, 7, 1, tzinfo=timezone.utc)


def test_parse_deadline_month():
    engine = PredictionEngine(MockProvider())
    result = engine._parse_deadline("June 2026")
    assert result == datetime(2026, 6, 1, tzinfo=timezone.utc)


def test_parse_deadline_month_with_day():
    engine = PredictionEngine(MockProvider())
    result = engine._parse_deadline("by July 15, 2026")
    assert result == datetime(2026, 7, 15, tzinfo=timezone.utc)


def test_parse_deadline_default():
    engine = PredictionEngine(MockProvider())
    result = engine._parse_deadline("")
    expected = datetime.now(timezone.utc) + timedelta(days=30)
    assert abs((result - expected).total_seconds()) < 60


# ---------------------------------------------------------------------------
# Tests — max predictions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_predictions_enforced():
    """More predictions than max are truncated."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(10))

    engine = PredictionEngine(provider)
    result = await engine.generate(
        [make_signal("test")],
        max_predictions=3,
    )

    assert len(result) <= 3


# ---------------------------------------------------------------------------
# Tests — A4 canonical verification wiring (one verification path)
# ---------------------------------------------------------------------------

class _StubVerifier:
    """Records whether verify() ran and returns deterministic results."""

    def __init__(self, adjusted: float = 0.42, summary: str = "stub-verified"):
        self.called = False
        self.received = None
        self._adjusted = adjusted
        self._summary = summary

    async def verify(self, predictions, *, deep_check: bool = False):
        from oracle.prediction.verifier import VerificationResult

        self.called = True
        self.received = list(predictions)
        return [
            VerificationResult(
                prediction_id=p.id,
                statement=p.statement,
                original_confidence=p.confidence,
                adjusted_confidence=self._adjusted,
                summary=self._summary,
            )
            for p in predictions
        ]


def test_invalid_verification_mode_raises():
    """Constructing with an unknown verification mode fails loudly."""
    with pytest.raises(ValueError):
        PredictionEngine(MockProvider(), verification_mode="sometimes")


@pytest.mark.asyncio
async def test_verification_off_does_not_call_verifier():
    """Default mode 'off' must NOT invoke the verifier (zero behaviour change)."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(2))
    stub = _StubVerifier(adjusted=0.42)

    engine = PredictionEngine(provider, verifier=stub)  # mode defaults to 'off'
    result = await engine.generate([make_signal("test")])

    assert len(result) == 2
    assert stub.called is False
    # Confidence untouched by the verifier.
    assert all(p.confidence != 0.42 for p in result)


@pytest.mark.asyncio
async def test_verification_live_adjusts_confidence():
    """Mode 'live' routes through the canonical verifier and adjusts confidence."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(2))
    stub = _StubVerifier(adjusted=0.42, summary="2 independent sources support")

    engine = PredictionEngine(provider, verifier=stub, verification_mode="live")
    result = await engine.generate([make_signal("test")])

    assert len(result) == 2
    assert stub.called is True
    for p in result:
        assert p.confidence == 0.42
        assert "[Verified: 2 independent sources support]" in p.reasoning


@pytest.mark.asyncio
async def test_verification_live_clamps_confidence():
    """Verifier output is clamped into the [0.01, 0.99] band."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(1))
    stub = _StubVerifier(adjusted=5.0)  # absurd value must be clamped

    engine = PredictionEngine(provider, verifier=stub, verification_mode="live")
    result = await engine.generate([make_signal("test")])

    assert len(result) == 1
    assert result[0].confidence == 0.99


@pytest.mark.asyncio
async def test_self_consistency_check_is_non_evidentiary():
    """Demoted verify.py never returns confidence or verdict labels."""
    from oracle.prediction.verify import self_consistency_check

    provider = MockProvider()
    provider.set_response(json.dumps({
        "self_consistency_warnings": [
            {"statement": "X will happen", "self_consistency_warning": "too vague"}
        ],
        "summary": "one vague statement",
    }))

    out = await self_consistency_check(provider, [{"statement": "X will happen"}])

    assert "self_consistency_warnings" in out
    # Must NOT carry any evidence/calibration fields.
    assert "adjusted_confidence" not in out
    assert "verified_predictions" not in out
    assert "verdict" not in out


# ---------------------------------------------------------------------------
# A3 — Structured-output validation at the LLM boundary (#3)
# ---------------------------------------------------------------------------


def test_structured_valid_batch_parses():
    """A well-formed batch validates every item with no rejections."""
    from oracle.tools.structured import parse_prediction_batch

    payload = make_valid_prediction_response(3)
    batch = parse_prediction_batch(payload)

    assert batch.ok
    assert batch.parse_error is None
    assert len(batch.valid) == 3
    assert batch.rejection_count == 0
    # The validated draft exposes the schema fields directly.
    assert all(d.statement for d in batch.valid)
    assert all(0.0 <= d.confidence <= 1.0 for d in batch.valid)


def test_structured_invalid_item_rejected_not_dropped():
    """A malformed item is REJECTED WITH A REASON, never silently dropped."""
    from oracle.tools.structured import parse_prediction_batch

    payload = json.dumps({
        "predictions": [
            {  # valid
                "statement": "Apple will announce an M4 MacBook Pro by June 2026",
                "category": "product_launch",
                "confidence": 0.7,
            },
            {  # invalid: statement too short (< 10 chars)
                "statement": "Nope",
                "category": "tech_trend",
                "confidence": 0.5,
            },
            {  # invalid: confidence out of [0, 1]
                "statement": "Something big will absolutely happen next quarter",
                "category": "market_move",
                "confidence": 7.5,
            },
        ]
    })

    batch = parse_prediction_batch(payload)

    assert batch.ok  # the container itself was valid JSON
    assert len(batch.valid) == 1
    assert batch.rejection_count == 2
    # Rejections preserve which item failed and a human-readable reason.
    rejected_indices = {r.index for r in batch.rejected}
    assert rejected_indices == {1, 2}
    assert all(r.reason for r in batch.rejected)
    # The original raw item is retained for auditing.
    assert all(r.raw is not None for r in batch.rejected)


def test_structured_malformed_payload_sets_parse_error():
    """Non-JSON prose yields a parse_error, not a confusing empty success."""
    from oracle.tools.structured import parse_prediction_batch

    batch = parse_prediction_batch("I don't feel like predicting today.")

    assert not batch.ok
    assert batch.parse_error is not None
    assert batch.valid == []


def test_structured_non_string_payload_accepted():
    """Production structured-output mode passes an already-decoded dict."""
    from oracle.tools.structured import parse_prediction_batch

    batch = parse_prediction_batch({
        "predictions": [
            {
                "statement": "Regulators will publish new AI rules by Q4 2026",
                "category": "regulatory",
                "confidence": 0.6,
            }
        ]
    })

    assert batch.ok
    assert len(batch.valid) == 1
    assert batch.valid[0].category == "regulatory"


def test_structured_sources_coercion():
    """A comma-joined string of sources is coerced into a clean list."""
    from oracle.tools.structured import PredictionDraft

    draft = PredictionDraft.model_validate({
        "statement": "GitHub will surpass 200M repos by end of 2026",
        "sources": "https://a.example, https://b.example ,",
    })

    assert draft.sources == ["https://a.example", "https://b.example"]


def test_structured_response_format_is_schema_dict():
    """The production response_format payload is a strict JSON-schema dict."""
    from oracle.tools.structured import (
        prediction_json_schema,
        prediction_response_format,
    )

    rf = prediction_response_format()
    assert isinstance(rf, dict)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert isinstance(rf["json_schema"]["schema"], dict)

    schema = prediction_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema


@pytest.mark.asyncio
async def test_engine_rejects_invalid_predictions_at_boundary():
    """Engine surfaces only schema-valid predictions; junk is rejected."""
    provider = MockProvider()
    provider.set_response(json.dumps({
        "predictions": [
            {
                "statement": "A genuinely valid prediction about the future of tech",
                "category": "tech_trend",
                "confidence": 0.6,
                "deadline": "2026-06-01",
                "sources": [],
            },
            {  # too short — must be rejected, not silently dropped
                "statement": "bad",
                "category": "tech_trend",
                "confidence": 0.6,
                "deadline": "2026-06-01",
                "sources": [],
            },
        ]
    }))

    engine = PredictionEngine(provider)
    result = await engine.generate([make_signal("test")])

    # Only the valid prediction survives the boundary.
    assert len(result) == 1
    assert result[0].statement.startswith("A genuinely valid")
