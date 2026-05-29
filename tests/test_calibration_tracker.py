"""Tests for calibration tracker."""

import pytest
from datetime import datetime, timezone, timedelta

from oracle.models.prediction import (
    CalibrationBucket,
    CalibrationReport,
    Category,
    Prediction,
    Status,
)
from oracle.calibration.tracker import CalibrationTracker, _confidence_bucket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prediction(
    category: Category = Category.TECH_TREND,
    confidence: float = 0.7,
    status: Status = Status.CORRECT,
    resolved_at: datetime | None = None,
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
        resolved_at=resolved_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests — confidence bucketing
# ---------------------------------------------------------------------------

def test_confidence_bucket():
    assert _confidence_bucket(0.55) == "0.5-0.6"
    assert _confidence_bucket(0.0) == "0.0-0.1"
    assert _confidence_bucket(0.99) == "0.9-1.0"
    assert _confidence_bucket(1.0) == "0.9-1.0"  # Clamped
    assert _confidence_bucket(0.70) == "0.7-0.8"
    assert _confidence_bucket(0.799) == "0.7-0.8"


# ---------------------------------------------------------------------------
# Tests — basic computation
# ---------------------------------------------------------------------------

def test_compute_empty():
    tracker = CalibrationTracker()
    report = tracker.compute([])
    assert report.overall_total == 0
    assert report.overall_correct == 0
    assert report.overall_accuracy == 0.0
    assert report.buckets == []


def test_compute_single_correct():
    tracker = CalibrationTracker()
    preds = [make_prediction(confidence=0.7, status=Status.CORRECT)]
    report = tracker.compute(preds)

    assert report.overall_total == 1
    assert report.overall_correct == 1
    assert report.overall_accuracy == 1.0


def test_compute_single_incorrect():
    tracker = CalibrationTracker()
    preds = [make_prediction(confidence=0.7, status=Status.INCORRECT)]
    report = tracker.compute(preds)

    assert report.overall_total == 1
    assert report.overall_correct == 0
    assert report.overall_accuracy == 0.0


def test_compute_ignores_pending():
    """Pending predictions are ignored in calibration."""
    tracker = CalibrationTracker()
    preds = [
        make_prediction(status=Status.CORRECT),
        make_prediction(status=Status.INCORRECT),
        make_prediction(status=Status.PENDING),  # Should be ignored
        make_prediction(status=Status.EXPIRED),  # Should be ignored
    ]
    report = tracker.compute(preds)

    assert report.overall_total == 2
    assert report.overall_correct == 1
    assert report.overall_accuracy == 0.5


# ---------------------------------------------------------------------------
# Tests — buckets
# ---------------------------------------------------------------------------

def test_compute_buckets():
    """Predictions are grouped into confidence buckets per category."""
    tracker = CalibrationTracker()
    preds = [
        make_prediction(Category.TECH_TREND, confidence=0.75, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.75, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.75, status=Status.INCORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.85, status=Status.CORRECT),
        make_prediction(Category.MARKET_MOVE, confidence=0.55, status=Status.INCORRECT),
    ]
    report = tracker.compute(preds)

    assert len(report.buckets) == 3  # 3 unique (category, bucket) pairs

    # TECH_TREND, 0.7-0.8: 3 total, 2 correct → 0.667 accuracy
    tech_7_8 = [b for b in report.buckets if b.category == Category.TECH_TREND and b.confidence_range == "0.7-0.8"]
    assert len(tech_7_8) == 1
    assert tech_7_8[0].total == 3
    assert tech_7_8[0].correct == 2
    assert round(tech_7_8[0].accuracy, 2) == 0.67

    # TECH_TREND, 0.8-0.9: 1 total, 1 correct → 1.0 accuracy
    tech_8_9 = [b for b in report.buckets if b.category == Category.TECH_TREND and b.confidence_range == "0.8-0.9"]
    assert len(tech_8_9) == 1
    assert tech_8_9[0].total == 1
    assert tech_8_9[0].correct == 1

    # MARKET_MOVE, 0.5-0.6: 1 total, 0 correct → 0.0 accuracy
    market_5_6 = [b for b in report.buckets if b.category == Category.MARKET_MOVE]
    assert len(market_5_6) == 1
    assert market_5_6[0].total == 1
    assert market_5_6[0].correct == 0


def test_bucket_accuracy_zero_when_empty():
    """Empty bucket has 0.0 accuracy."""
    bucket = CalibrationBucket(
        category=Category.TECH_TREND,
        confidence_range="0.5-0.6",
        total=0,
        correct=0,
    )
    assert bucket.accuracy == 0.0


def test_bucket_accuracy():
    """Bucket accuracy is correct/total."""
    bucket = CalibrationBucket(
        category=Category.TECH_TREND,
        confidence_range="0.5-0.6",
        total=10,
        correct=7,
    )
    assert bucket.accuracy == 0.7


# ---------------------------------------------------------------------------
# Tests — Brier score
# ---------------------------------------------------------------------------

def test_brier_score_perfect():
    """Perfect predictions have Brier score 0."""
    tracker = CalibrationTracker()
    preds = [
        make_prediction(confidence=0.9, status=Status.CORRECT),
        make_prediction(confidence=0.1, status=Status.INCORRECT),
    ]
    # Perfect: 90% correct was correct, 10% confident was wrong (so 10% was right that it'd be wrong)
    score = tracker.compute_brier_score(preds)
    # Brier = ((0.9-1)^2 + (0.1-0)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
    assert round(score, 4) == 0.01


def test_brier_score_terrible():
    """Terrible predictions have high Brier score."""
    tracker = CalibrationTracker()
    preds = [
        make_prediction(confidence=0.9, status=Status.INCORRECT),
        make_prediction(confidence=0.1, status=Status.CORRECT),
    ]
    score = tracker.compute_brier_score(preds)
    # Brier = ((0.9-0)^2 + (0.1-1)^2) / 2 = (0.81 + 0.81) / 2 = 0.81
    assert round(score, 4) == 0.81


def test_brier_score_fifty_fifty():
    """Always predicting 50% yields Brier = 0.25."""
    tracker = CalibrationTracker()
    preds = [
        make_prediction(confidence=0.5, status=Status.CORRECT),
        make_prediction(confidence=0.5, status=Status.INCORRECT),
        make_prediction(confidence=0.5, status=Status.CORRECT),
        make_prediction(confidence=0.5, status=Status.INCORRECT),
    ]
    score = tracker.compute_brier_score(preds)
    # Each: (0.5 - outcome)^2 = 0.25
    assert round(score, 4) == 0.25


# ---------------------------------------------------------------------------
# Tests — by-category stats
# ---------------------------------------------------------------------------

def test_by_category_stats():
    """Per-category stats are computed correctly."""
    tracker = CalibrationTracker()
    preds = [
        # TECH_TREND: 3 correct out of 4 = 0.75
        make_prediction(Category.TECH_TREND, confidence=0.8, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.8, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.8, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.8, status=Status.INCORRECT),
        # MARKET_MOVE: 1 correct out of 2 = 0.5
        make_prediction(Category.MARKET_MOVE, confidence=0.6, status=Status.CORRECT),
        make_prediction(Category.MARKET_MOVE, confidence=0.6, status=Status.INCORRECT),
    ]
    report = tracker.compute(preds)

    tech = report.by_category["tech_trend"]
    assert tech["total"] == 4
    assert tech["correct"] == 3
    assert tech["accuracy"] == 0.75
    assert tech["avg_confidence"] == 0.8
    assert "brier_score" in tech

    market = report.by_category["market_move"]
    assert market["total"] == 2
    assert market["correct"] == 1
    assert market["accuracy"] == 0.5


# ---------------------------------------------------------------------------
# Tests — calibration curve
# ---------------------------------------------------------------------------

def test_calibration_curve():
    """Calibration curve data maps confidence to accuracy."""
    tracker = CalibrationTracker()
    preds = [
        make_prediction(Category.TECH_TREND, confidence=0.55, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.55, status=Status.INCORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.75, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.75, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.75, status=Status.CORRECT),
    ]
    report = tracker.compute(preds)

    curve = report.__dict__.get("calibration_curve", [])
    assert len(curve) == 2  # Two buckets

    # 0.5-0.6: 50% accuracy
    bucket_5_6 = [c for c in curve if c["confidence_range"] == "0.5-0.6"][0]
    assert bucket_5_6["accuracy"] == 0.5
    assert bucket_5_6["mid_confidence"] == 0.55  # Near 0.55

    # 0.7-0.8: 100% accuracy
    bucket_7_8 = [c for c in curve if c["confidence_range"] == "0.7-0.8"][0]
    assert bucket_7_8["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Tests — category filter
# ---------------------------------------------------------------------------

def test_category_filter():
    """Filter calibration to a specific category."""
    tracker = CalibrationTracker()
    preds = [
        make_prediction(Category.TECH_TREND, confidence=0.7, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.7, status=Status.INCORRECT),
        make_prediction(Category.MARKET_MOVE, confidence=0.7, status=Status.CORRECT),
    ]
    report = tracker.compute(preds, category_filter=Category.MARKET_MOVE)

    assert report.overall_total == 1
    assert report.overall_correct == 1
    assert len(report.by_category) == 1
    assert "market_move" in report.by_category


# ---------------------------------------------------------------------------
# Tests — trend analysis
# ---------------------------------------------------------------------------

def test_compute_trends():
    """Trend analysis groups by week."""
    tracker = CalibrationTracker()
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)

    preds = [
        make_prediction(confidence=0.7, status=Status.CORRECT,
                        resolved_at=base),
        make_prediction(confidence=0.7, status=Status.INCORRECT,
                        resolved_at=base + timedelta(days=1)),
        make_prediction(confidence=0.7, status=Status.CORRECT,
                        resolved_at=base + timedelta(days=8)),
        make_prediction(confidence=0.7, status=Status.CORRECT,
                        resolved_at=base + timedelta(days=9)),
    ]

    trends = tracker.compute_trends(preds)

    assert "trends" in trends
    assert len(trends["trends"]) >= 1  # At least one week
    assert "direction" in trends


def test_compute_trends_empty():
    """No resolved predictions → empty trends."""
    tracker = CalibrationTracker()
    trends = tracker.compute_trends([])
    assert "trends" in trends
    assert len(trends["trends"]) == 0


# ---------------------------------------------------------------------------
# Tests — calibration bias
# ---------------------------------------------------------------------------

def test_calibration_bias():
    """Overconfident predictions show positive calibration bias."""
    tracker = CalibrationTracker()
    preds = [
        # All 80% confident, but only 50% correct = overconfident
        make_prediction(Category.TECH_TREND, confidence=0.8, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.8, status=Status.INCORRECT),
    ]
    report = tracker.compute(preds)

    tech = report.by_category["tech_trend"]
    # accuracy = 0.5, avg_confidence = 0.8, bias = 0.8 - 0.5 = 0.3
    assert tech["calibration_bias"] > 0
    assert round(tech["calibration_bias"], 1) == 0.3


def test_underconfidence_bias():
    """Underconfident predictions show negative calibration bias."""
    tracker = CalibrationTracker()
    preds = [
        # 60% confident, but 100% correct = underconfident
        make_prediction(Category.TECH_TREND, confidence=0.6, status=Status.CORRECT),
        make_prediction(Category.TECH_TREND, confidence=0.6, status=Status.CORRECT),
    ]
    report = tracker.compute(preds)

    tech = report.by_category["tech_trend"]
    assert tech["calibration_bias"] < 0
    assert round(tech["calibration_bias"], 1) == -0.4
