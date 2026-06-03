"""Tests for contextualized confidence (E17) and the advanced metrics
serializer (E15).

These cover the "never show a number without its track record" guarantee:

* a proven confidence bucket yields ``basis="bucket"`` with the right accuracy,
* a sparse bucket falls back to the whole-category record (``basis="category"``),
* no applicable history is reported *honestly* (``basis="none"``, not proven),
* category coercion accepts both ``Category`` enums and plain strings,
* ``INSUFFICIENT_EVIDENCE`` history never inflates the denominators,
* ``PredictionEngine.contextualize`` mutates and returns the predictions, and
* the advanced calibration report serializes to a stable, JSON-safe payload
  that includes the derived ``ece`` / ``is_well_calibrated`` fields.
"""

import json
from datetime import datetime, timezone

import pytest

from oracle.models.prediction import Category, Prediction, Status
from oracle.calibration.tracker import (
    ConfidenceContextualizer,
    TrackRecord,
    _confidence_bucket,
)
from oracle.calibration.metrics import compute_full_report, serialize_full_report
from oracle.prediction.engine import PredictionEngine
from oracle.llm import MockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction(
    category: Category = Category.TECH_TREND,
    confidence: float = 0.7,
    status: Status = Status.CORRECT,
) -> Prediction:
    return Prediction(
        category=category,
        statement=f"Test prediction with {confidence:.0%} confidence",
        confidence=confidence,
        reasoning="Test reasoning",
        sources=[],
        deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
        status=status,
        resolution="Test resolution" if status != Status.PENDING else None,
        resolved_at=datetime.now(timezone.utc),
    )


def resolved_bucket(
    category: Category,
    confidence: float,
    n_correct: int,
    n_incorrect: int,
) -> list[Prediction]:
    """Build a batch of resolved predictions at one confidence level."""
    out = [
        make_prediction(category, confidence, Status.CORRECT)
        for _ in range(n_correct)
    ]
    out += [
        make_prediction(category, confidence, Status.INCORRECT)
        for _ in range(n_incorrect)
    ]
    return out


# ---------------------------------------------------------------------------
# ConfidenceContextualizer — proven bucket
# ---------------------------------------------------------------------------

def test_proven_bucket_track_record():
    ctx = ConfidenceContextualizer(min_samples=5)
    # 7 correct + 3 incorrect at 0.70 -> 70% accuracy in the 0.7-0.8 bucket.
    history = resolved_bucket(Category.TECH_TREND, 0.70, n_correct=7, n_incorrect=3)

    record = ctx.track_record_for(Category.TECH_TREND, 0.70, history)

    assert record.is_proven is True
    assert record.basis == "bucket"
    assert record.bucket == "0.7-0.8"
    assert record.bucket_sample_size == 10
    assert record.bucket_correct == 7
    assert record.bucket_accuracy == 0.7
    assert "70%" in record.phrase
    # The headline phrase must cite the actual hit rate, not the raw confidence.
    assert "70% of the time" in record.phrase


def test_bucket_only_counts_same_category():
    ctx = ConfidenceContextualizer(min_samples=3)
    history = resolved_bucket(Category.TECH_TREND, 0.70, 3, 0)
    history += resolved_bucket(Category.MARKET_MOVE, 0.70, 5, 0)

    record = ctx.track_record_for(Category.TECH_TREND, 0.70, history)

    assert record.bucket_sample_size == 3
    assert record.category == Category.TECH_TREND.value


# ---------------------------------------------------------------------------
# Category fallback
# ---------------------------------------------------------------------------

def test_category_fallback_when_bucket_sparse():
    ctx = ConfidenceContextualizer(min_samples=5)
    # Spread across buckets so no single bucket reaches 5, but category does.
    history = resolved_bucket(Category.TECH_TREND, 0.62, 2, 0)  # 0.6-0.7
    history += resolved_bucket(Category.TECH_TREND, 0.75, 3, 1)  # 0.7-0.8
    history += resolved_bucket(Category.TECH_TREND, 0.85, 1, 1)  # 0.8-0.9

    record = ctx.track_record_for(Category.TECH_TREND, 0.72, history)

    assert record.is_proven is False
    assert record.basis == "category"
    assert record.bucket_sample_size < ctx.min_samples
    assert record.category_sample_size == 8
    assert record.category_correct == 6
    assert record.category_accuracy == pytest.approx(6 / 8, abs=1e-4)
    assert "overall" in record.phrase


# ---------------------------------------------------------------------------
# Honest "no record" path
# ---------------------------------------------------------------------------

def test_no_history_is_honest():
    ctx = ConfidenceContextualizer(min_samples=5)
    record = ctx.track_record_for(Category.STARTUP_SUCCESS, 0.80, [])

    assert record.is_proven is False
    assert record.basis == "none"
    assert record.bucket_accuracy is None
    assert record.category_accuracy is None
    assert "no proven track record" in record.phrase.lower()
    assert "provisional" in record.phrase.lower()


def test_other_category_history_does_not_count():
    ctx = ConfidenceContextualizer(min_samples=3)
    history = resolved_bucket(Category.MARKET_MOVE, 0.80, 10, 0)

    record = ctx.track_record_for(Category.STARTUP_SUCCESS, 0.80, history)

    assert record.basis == "none"
    assert record.category_sample_size == 0


# ---------------------------------------------------------------------------
# Category coercion + INSUFFICIENT_EVIDENCE exclusion
# ---------------------------------------------------------------------------

def test_string_category_coercion():
    ctx = ConfidenceContextualizer(min_samples=3)
    history = resolved_bucket(Category.TECH_TREND, 0.70, 3, 0)

    record = ctx.track_record_for("tech_trend", 0.70, history)

    assert record.basis == "bucket"
    assert record.category == "tech_trend"


def test_insufficient_evidence_excluded_from_denominator():
    ctx = ConfidenceContextualizer(min_samples=3)
    history = resolved_bucket(Category.TECH_TREND, 0.70, 3, 0)
    # Abstentions / unscorable outcomes must never pad the track record.
    history += [
        make_prediction(Category.TECH_TREND, 0.70, Status.INSUFFICIENT_EVIDENCE)
        for _ in range(4)
    ]
    history += [make_prediction(Category.TECH_TREND, 0.70, Status.PENDING)]

    record = ctx.track_record_for(Category.TECH_TREND, 0.70, history)

    assert record.bucket_sample_size == 3
    assert record.bucket_correct == 3
    assert record.bucket_accuracy == 1.0


def test_to_dict_round_trips_all_fields():
    record = TrackRecord(
        confidence=0.7,
        confidence_pct=70,
        category="tech_trend",
        bucket="0.7-0.8",
        min_samples=5,
    )
    d = record.to_dict()
    for key in (
        "confidence", "confidence_pct", "category", "bucket", "min_samples",
        "bucket_sample_size", "bucket_correct", "bucket_accuracy",
        "category_sample_size", "category_correct", "category_accuracy",
        "is_proven", "basis", "phrase",
    ):
        assert key in d


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_engine_contextualize_mutates_and_returns():
    engine = PredictionEngine(MockProvider())
    history = resolved_bucket(Category.TECH_TREND, 0.70, 6, 4)
    fresh = [make_prediction(Category.TECH_TREND, 0.70, Status.PENDING)]

    returned = engine.contextualize(fresh, history)

    assert returned is fresh
    assert fresh[0].track_record is not None
    assert fresh[0].track_record["basis"] == "bucket"
    assert fresh[0].track_record["bucket_accuracy"] == 0.6


def test_engine_contextualize_min_samples_threshold():
    engine = PredictionEngine(MockProvider())
    history = resolved_bucket(Category.TECH_TREND, 0.70, 3, 0)
    fresh = [make_prediction(Category.TECH_TREND, 0.70, Status.PENDING)]

    # With a high bar the same history is no longer "proven".
    engine.contextualize(fresh, history, min_samples=10)
    assert fresh[0].track_record["is_proven"] is False
    assert fresh[0].track_record["basis"] == "none"


def test_track_record_is_json_serializable():
    ctx = ConfidenceContextualizer(min_samples=2)
    history = resolved_bucket(Category.TECH_TREND, 0.70, 5, 5)
    record = ctx.track_record_for(Category.TECH_TREND, 0.70, history)
    # Must survive the API boundary (model_dump -> JSON).
    json.dumps(record.to_dict())


# ---------------------------------------------------------------------------
# E15 — advanced metrics serializer
# ---------------------------------------------------------------------------

def test_serialize_full_report_includes_derived_fields():
    history = resolved_bucket(Category.TECH_TREND, 0.70, 7, 3)
    history += resolved_bucket(Category.MARKET_MOVE, 0.90, 4, 1)

    report = compute_full_report(history)
    payload = serialize_full_report(report)

    # Derived @property fields that dataclasses.asdict would otherwise drop.
    assert "ece" in payload
    assert "is_well_calibrated" in payload
    assert isinstance(payload["ece"], float)
    assert isinstance(payload["is_well_calibrated"], bool)

    # Core fields are present.
    assert payload["overall_total"] == 15
    assert payload["overall_correct"] == 11
    assert "brier_score" in payload
    assert "calibration_curve" in payload
    assert "decomposition" in payload
    assert "by_category" in payload

    # generated_at must be ISO-encoded, and the whole payload JSON-safe.
    assert isinstance(payload["generated_at"], str)
    json.dumps(payload)


def test_serialize_full_report_respects_category_filter():
    history = resolved_bucket(Category.TECH_TREND, 0.70, 7, 3)
    history += resolved_bucket(Category.MARKET_MOVE, 0.90, 4, 1)

    payload = serialize_full_report(
        compute_full_report(history, category=Category.TECH_TREND)
    )

    assert payload["overall_total"] == 10
    assert payload["overall_correct"] == 7


def test_serialize_full_report_empty_history():
    payload = serialize_full_report(compute_full_report([]))

    assert payload["overall_total"] == 0
    assert payload["overall_accuracy"] == 0.0
    json.dumps(payload)
