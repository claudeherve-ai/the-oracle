"""Tests for the backtest harness (E16).

A backtest is only worth anything if it cannot quietly flatter the system. These
tests pin the two properties that make it trustworthy: (1) only resolved
predictions ever count toward calibration, and (2) the rolling replay never peeks
at the future — each prediction's advertised track record is computed from the
strictly-earlier resolved history only.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from oracle.backtest import (
    BacktestReport,
    ReliabilityRow,
    RollingPoint,
    run_backtest,
)
from oracle.models.prediction import Category, Prediction, Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def make_pred(
    confidence: float,
    status: Status,
    *,
    category: Category = Category.TECH_TREND,
    order: int = 0,
) -> Prediction:
    """Construct a prediction with a deterministic created_at for ordering."""
    return Prediction(
        category=category,
        statement="A specific, verifiable, time-bound claim about the future.",
        confidence=confidence,
        status=status,
        created_at=_BASE + timedelta(days=order),
    )


def resolved_bucket(
    confidence: float,
    n_correct: int,
    n_incorrect: int,
    *,
    category: Category = Category.TECH_TREND,
    start: int = 0,
) -> list[Prediction]:
    preds: list[Prediction] = []
    order = start
    for _ in range(n_correct):
        preds.append(make_pred(confidence, Status.CORRECT, category=category, order=order))
        order += 1
    for _ in range(n_incorrect):
        preds.append(make_pred(confidence, Status.INCORRECT, category=category, order=order))
        order += 1
    return preds


# ---------------------------------------------------------------------------
# Scoring discipline
# ---------------------------------------------------------------------------

def test_empty_predictions_has_no_track_record():
    report = run_backtest([])
    assert report.total == 0
    assert report.resolved == 0
    assert report.skipped_unresolved == 0
    assert report.well_calibrated is False
    assert report.rolling_honesty_gap is None
    assert "no track record" in report.verdict.lower()


def test_only_correct_incorrect_are_scored():
    preds = [
        make_pred(0.7, Status.CORRECT, order=0),
        make_pred(0.7, Status.INCORRECT, order=1),
        make_pred(0.7, Status.PENDING, order=2),
        make_pred(0.7, Status.EXPIRED, order=3),
        make_pred(0.7, Status.INSUFFICIENT_EVIDENCE, order=4),
    ]
    report = run_backtest(preds)
    assert report.total == 5
    assert report.resolved == 2
    assert report.skipped_unresolved == 3


def test_abstentions_never_enter_a_denominator():
    # An abstention sitting next to one correct call must not change accuracy.
    preds = [
        make_pred(0.9, Status.CORRECT, order=0),
        make_pred(0.9, Status.INSUFFICIENT_EVIDENCE, order=1),
    ]
    report = run_backtest(preds)
    assert report.resolved == 1
    assert report.report.overall_accuracy == 1.0


# ---------------------------------------------------------------------------
# Reliability table (predicted vs realized)
# ---------------------------------------------------------------------------

def test_reliability_row_compares_predicted_to_realized():
    # 10 predictions all at 0.70, 6 correct -> realized 0.60, gap +0.10.
    preds = resolved_bucket(0.70, n_correct=6, n_incorrect=4)
    report = run_backtest(preds)
    assert len(report.reliability) == 1
    row = report.reliability[0]
    assert isinstance(row, ReliabilityRow)
    assert row.n == 10
    assert row.predicted_confidence == pytest.approx(0.70)
    assert row.realized_accuracy == pytest.approx(0.60)
    assert row.gap == pytest.approx(0.10)


def test_mean_calibration_gap_is_sample_weighted():
    # Bucket A: 0.70 conf, 0.70 realized (gap 0), 10 samples.
    # Bucket B: 0.90 conf, 0.50 realized (gap 0.40), 2 samples.
    # Weighted mean |gap| = (0*10 + 0.40*2) / 12 = 0.0667.
    preds = (
        resolved_bucket(0.70, n_correct=7, n_incorrect=3, start=0)
        + resolved_bucket(0.90, n_correct=1, n_incorrect=1, start=100)
    )
    report = run_backtest(preds)
    assert report.mean_calibration_gap == pytest.approx(0.0667, abs=1e-3)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_preliminary_verdict_below_trust_threshold():
    preds = resolved_bucket(0.70, n_correct=7, n_incorrect=3)  # 10 resolved
    report = run_backtest(preds, min_for_trust=20)
    assert "preliminary" in report.verdict.lower()


def test_well_calibrated_verdict_when_confidence_matches_reality():
    # The calibration curve uses equal-frequency bins (bin_size = n // 10). With
    # n = 40 and a 0.75 confidence, each bin holds 4 predictions; a repeating
    # [C, C, C, I] pattern makes every bin exactly 75% accurate, so ECE -> 0 and
    # the system is genuinely well-calibrated rather than accidentally so.
    preds = [
        make_pred(
            0.75,
            Status.CORRECT if (i % 4 != 3) else Status.INCORRECT,
            order=i,
        )
        for i in range(40)
    ]
    report = run_backtest(preds, min_for_trust=20)
    assert report.resolved == 40
    assert report.report.overall_accuracy == pytest.approx(0.75)
    assert report.mean_calibration_gap == pytest.approx(0.0, abs=1e-6)
    assert report.well_calibrated is True
    assert "well-calibrated" in report.verdict.lower()


# ---------------------------------------------------------------------------
# Rolling replay (no hindsight)
# ---------------------------------------------------------------------------

def test_rolling_replay_is_chronological_and_blind_to_the_future():
    preds = resolved_bucket(0.70, n_correct=3, n_incorrect=2)
    report = run_backtest(preds)
    points = report.rolling_points
    assert len(points) == 5
    # First prediction had zero prior history.
    assert points[0].prior_scored_samples == 0
    assert points[0].claimed_basis == "none"
    assert points[0].claimed_accuracy is None
    # Prior-sample count strictly increases with chronological position.
    assert [pt.prior_scored_samples for pt in points] == [0, 1, 2, 3, 4]


def test_no_claims_until_min_samples_history_exists():
    # Only 4 resolved in the category; min_samples=5 -> never enough to claim.
    preds = resolved_bucket(0.70, n_correct=2, n_incorrect=2)
    report = run_backtest(preds, min_samples=5)
    assert report.rolling_claim_coverage == 0
    assert report.rolling_honesty_gap is None
    assert all(pt.claimed_accuracy is None for pt in report.rolling_points)


def test_rolling_honesty_gap_is_computed_once_history_is_deep_enough():
    # 12 resolved in one category; after 5 priors the contextualizer can claim.
    preds = resolved_bucket(0.70, n_correct=8, n_incorrect=4)
    report = run_backtest(preds, min_samples=5)
    assert report.rolling_claim_coverage > 0
    assert report.rolling_honesty_gap is not None
    assert 0.0 <= report.rolling_honesty_gap <= 1.0
    # Every claim-bearing point must expose the basis it used.
    claimed = [pt for pt in report.rolling_points if pt.claimed_accuracy is not None]
    assert all(pt.claimed_basis in ("bucket", "category") for pt in claimed)


# ---------------------------------------------------------------------------
# Category filtering
# ---------------------------------------------------------------------------

def test_category_filter_scopes_resolved_set():
    preds = (
        resolved_bucket(0.70, n_correct=3, n_incorrect=0, category=Category.TECH_TREND, start=0)
        + resolved_bucket(0.70, n_correct=0, n_incorrect=2, category=Category.MARKET_MOVE, start=50)
    )
    report = run_backtest(preds, category=Category.TECH_TREND)
    # Only the 3 tech-trend predictions are scored; both market-move are excluded.
    assert report.resolved == 3
    assert report.report.overall_accuracy == 1.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_to_dict_is_json_serializable_and_complete():
    preds = resolved_bucket(0.70, n_correct=6, n_incorrect=4)
    report = run_backtest(preds)
    payload = report.to_dict()
    # Must round-trip through JSON with no custom encoder.
    text = json.dumps(payload)
    again = json.loads(text)
    for key in (
        "total",
        "resolved",
        "skipped_unresolved",
        "report",
        "reliability",
        "rolling_points",
        "mean_calibration_gap",
        "well_calibrated",
        "verdict",
        "generated_at",
    ):
        assert key in again
    assert isinstance(again["reliability"], list)
    assert isinstance(again["rolling_points"], list)


def test_report_is_an_advanced_calibration_report():
    preds = resolved_bucket(0.70, n_correct=6, n_incorrect=4)
    report = run_backtest(preds)
    assert isinstance(report, BacktestReport)
    # The embedded report is the same type the live dashboard serves.
    assert report.report.overall_total == 10
    assert report.report.overall_correct == 6
