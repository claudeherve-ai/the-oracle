"""Calibration tracking and reporting for The Oracle.

Tracks resolved predictions, computes accuracy per confidence bucket,
Brier scores, and generates calibration reports.

All computation is pure — no LLM calls needed for calibration math.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from oracle.models.prediction import (
    CalibrationBucket,
    CalibrationReport,
    Category,
    Prediction,
    Status,
)

logger = logging.getLogger("oracle.calibration.tracker")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confidence_bucket(confidence: float) -> str:
    """Map a confidence value to a bucket string like '0.5-0.6'.

    Buckets are in 0.1 increments:
    0.0-0.1, 0.1-0.2, ..., 0.8-0.9, 0.9-1.0
    """
    # Floor to 0.1 (use math.floor — round() uses banker's rounding)
    import math
    lower = math.floor(confidence * 10) / 10
    if lower >= 1.0:
        lower = 0.9
    upper = min(lower + 0.1, 1.0)
    return f"{lower:.1f}-{upper:.1f}"


# ---------------------------------------------------------------------------
# Calibration Tracker
# ---------------------------------------------------------------------------


class CalibrationTracker:
    """Computes calibration metrics from resolved predictions.

    Usage:
        tracker = CalibrationTracker()
        report = tracker.compute(resolved_predictions)
        print(f"Overall accuracy: {report.overall_accuracy:.1%}")
        print(f"Brier score: {report.brier_score:.4f}")
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        predictions: List[Prediction],
        *,
        category_filter: Optional[Category] = None,
    ) -> CalibrationReport:
        """Compute a full calibration report from resolved predictions.

        Only predictions with status CORRECT or INCORRECT are included.
        PENDING and EXPIRED predictions are ignored.

        Args:
            predictions: List of predictions (ideally only resolved ones).
            category_filter: Optional — only compute for this category.

        Returns:
            CalibrationReport with buckets, Brier score, and accuracy.
        """
        # Filter to resolved predictions only
        resolved = [
            p for p in predictions
            if p.status in (Status.CORRECT, Status.INCORRECT)
        ]
        if category_filter:
            resolved = [p for p in resolved if p.category == category_filter]

        if not resolved:
            logger.warning("No resolved predictions to compute calibration")
            return CalibrationReport(
                overall_total=0,
                overall_correct=0,
                buckets=[],
                by_category={},
            )

        # Overall stats
        overall_total = len(resolved)
        overall_correct = sum(1 for p in resolved if p.status == Status.CORRECT)

        # Compute by-category stats
        by_category = self._compute_by_category(resolved)

        # Compute buckets (by category + confidence range)
        buckets = self._compute_buckets(resolved)

        # Compute Brier score
        brier = self._compute_brier_score(resolved)

        # Compute calibration curve data
        calibration_curve = self._compute_calibration_curve(buckets)

        report = CalibrationReport(
            overall_total=overall_total,
            overall_correct=overall_correct,
            buckets=buckets,
            by_category=by_category,
        )

        # Attach computed fields via __dict__ (they're not in the model but
        # we add them for API/dashboard consumers)
        report.__dict__["brier_score"] = brier
        report.__dict__["calibration_curve"] = calibration_curve

        logger.info(
            "Calibration: %d/%d (%.1f%%) correct, Brier=%.4f, %d buckets",
            overall_correct, overall_total,
            (overall_correct / overall_total * 100) if overall_total > 0 else 0,
            brier,
            len(buckets),
        )

        return report

    def compute_brier_score(self, predictions: List[Prediction]) -> float:
        """Compute Brier score for a set of resolved predictions.

        Brier = (1/N) * sum((confidence - outcome)^2)
        Outcome = 1 for CORRECT, 0 for INCORRECT.

        Lower is better. 0 = perfect calibration, 0.25 = random guessing at 50%.
        """
        return self._compute_brier_score(predictions)

    # ------------------------------------------------------------------
    # Bucket computation
    # ------------------------------------------------------------------

    def _compute_buckets(
        self,
        predictions: List[Prediction],
    ) -> List[CalibrationBucket]:
        """Group predictions into confidence buckets per category."""
        # Key: (category, confidence_bucket) -> (total, correct)
        bucket_data: Dict[Tuple[Category, str], Tuple[int, int]] = defaultdict(lambda: (0, 0))

        for p in predictions:
            bucket_str = _confidence_bucket(p.confidence)
            key = (p.category, bucket_str)
            total, correct = bucket_data[key]
            total += 1
            if p.status == Status.CORRECT:
                correct += 1
            bucket_data[key] = (total, correct)

        buckets = []
        for (category, conf_range), (total, correct) in sorted(
            bucket_data.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        ):
            buckets.append(
                CalibrationBucket(
                    category=category,
                    confidence_range=conf_range,
                    total=total,
                    correct=correct,
                )
            )

        return buckets

    # ------------------------------------------------------------------
    # Category stats
    # ------------------------------------------------------------------

    def _compute_by_category(
        self,
        predictions: List[Prediction],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-category accuracy and stats."""
        cat_data: Dict[Category, List[Prediction]] = defaultdict(list)
        for p in predictions:
            cat_data[p.category].append(p)

        result: Dict[str, Dict[str, Any]] = {}
        for category, preds in cat_data.items():
            total = len(preds)
            correct = sum(1 for p in preds if p.status == Status.CORRECT)
            accuracy = correct / total if total > 0 else 0.0
            avg_confidence = sum(p.confidence for p in preds) / total if total > 0 else 0.0
            brier = self._compute_brier_score(preds)

            result[category.value] = {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "brier_score": brier,
                # Calibration bias: positive = overconfident, negative = underconfident
                "calibration_bias": avg_confidence - accuracy,
            }

        return result

    # ------------------------------------------------------------------
    # Brier score
    # ------------------------------------------------------------------

    def _compute_brier_score(self, predictions: List[Prediction]) -> float:
        """Compute the Brier score.

        Brier = (1/N) * sum((confidence_i - outcome_i)^2)

        Where:
        - outcome_i = 1 if CORRECT, 0 if INCORRECT
        - PENDING/EXPIRED predictions are skipped

        A perfect forecaster has Brier score = 0.
        Always predicting 50% yields Brier = 0.25.
        """
        resolved = [
            p for p in predictions
            if p.status in (Status.CORRECT, Status.INCORRECT)
        ]
        if not resolved:
            return 0.0

        total_squared_error = 0.0
        for p in resolved:
            outcome = 1.0 if p.status == Status.CORRECT else 0.0
            error = p.confidence - outcome
            total_squared_error += error * error

        return total_squared_error / len(resolved)

    # ------------------------------------------------------------------
    # Calibration curve
    # ------------------------------------------------------------------

    def _compute_calibration_curve(
        self,
        buckets: List[CalibrationBucket],
    ) -> List[Dict[str, Any]]:
        """Build calibration curve data for plotting.

        X-axis: predicted confidence (bucket midpoint)
        Y-axis: actual accuracy in that bucket

        A perfectly calibrated system follows the diagonal.
        """
        curve = []
        for bucket in buckets:
            mid_confidence = self._bucket_midpoint(bucket.confidence_range)
            curve.append({
                "confidence_range": bucket.confidence_range,
                "mid_confidence": mid_confidence,
                "accuracy": bucket.accuracy,
                "total": bucket.total,
                "correct": bucket.correct,
                # Gap from perfect calibration
                "calibration_gap": mid_confidence - bucket.accuracy,
            })

        curve.sort(key=lambda x: x["mid_confidence"])
        return curve

    @staticmethod
    def _bucket_midpoint(bucket_range: str) -> float:
        """Get the midpoint of a confidence bucket string.

        "0.5-0.6" -> 0.55
        """
        parts = bucket_range.split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return 0.5

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def compute_trends(
        self,
        predictions: List[Prediction],
    ) -> Dict[str, Any]:
        """Compute calibration trend data over time.

        Useful for tracking whether the system is getting better or worse.
        """
        resolved = [
            p for p in predictions
            if p.status in (Status.CORRECT, Status.INCORRECT) and p.resolved_at
        ]
        if not resolved:
            return {"trends": [], "summary": "No resolved predictions with timestamps"}

        # Group by week
        resolved.sort(key=lambda p: p.resolved_at)
        weeks: Dict[str, List[Prediction]] = defaultdict(list)

        for p in resolved:
            # ISO week
            week_key = p.resolved_at.strftime("%Y-W%V")
            weeks[week_key].append(p)

        trends = []
        for week_key in sorted(weeks.keys()):
            week_preds = weeks[week_key]
            total = len(week_preds)
            correct = sum(1 for p in week_preds if p.status == Status.CORRECT)
            accuracy = correct / total if total > 0 else 0.0
            brier = self._compute_brier_score(week_preds)
            trends.append({
                "week": week_key,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "brier_score": brier,
            })

        # Simple trend direction
        if len(trends) >= 2:
            recent_accuracy = trends[-1]["accuracy"]
            earlier_accuracy = trends[0]["accuracy"]
            delta = recent_accuracy - earlier_accuracy
            if delta > 0.02:
                direction = "improving"
            elif delta < -0.02:
                direction = "declining"
            else:
                direction = "stable"
        else:
            direction = "insufficient_data"

        return {
            "trends": trends,
            "direction": direction,
            "earliest_week": trends[0]["week"] if trends else None,
            "latest_week": trends[-1]["week"] if trends else None,
        }


# ---------------------------------------------------------------------------
# Contextualized confidence (E17) — "never show a number without its record"
# ---------------------------------------------------------------------------


@dataclass
class TrackRecord:
    """The historical track record behind a single confidence number.

    Built from resolved history so a raw ``0.70`` can be presented as
    "70% confident — this category has been right 68% of the time across 142
    resolved predictions at this confidence level". When there is not yet
    enough history the record is *honest about that*: ``is_proven`` is False and
    the phrase says so rather than inventing a reassuring statistic.

    ``basis`` records WHICH evidence the headline accuracy came from:

    * ``"bucket"`` — the exact confidence bucket for this category (strongest).
    * ``"category"`` — the whole category (used when the bucket is too sparse).
    * ``"none"`` — no resolved history applies yet; treat as provisional.
    """

    confidence: float
    confidence_pct: int
    category: str
    bucket: str
    min_samples: int
    # Exact confidence bucket within the category
    bucket_sample_size: int = 0
    bucket_correct: int = 0
    bucket_accuracy: Optional[float] = None
    # Whole-category fallback
    category_sample_size: int = 0
    category_correct: int = 0
    category_accuracy: Optional[float] = None
    is_proven: bool = False
    basis: str = "none"
    phrase: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "confidence_pct": self.confidence_pct,
            "category": self.category,
            "bucket": self.bucket,
            "min_samples": self.min_samples,
            "bucket_sample_size": self.bucket_sample_size,
            "bucket_correct": self.bucket_correct,
            "bucket_accuracy": self.bucket_accuracy,
            "category_sample_size": self.category_sample_size,
            "category_correct": self.category_correct,
            "category_accuracy": self.category_accuracy,
            "is_proven": self.is_proven,
            "basis": self.basis,
            "phrase": self.phrase,
        }


class ConfidenceContextualizer:
    """Attaches a historical :class:`TrackRecord` to each prediction.

    Pure logic — no LLM, no network. Given a list of *resolved* predictions
    (the audited history), it looks up how often the system has actually been
    right for the same category at the same confidence level, and renders an
    honest, human-readable phrase. This is the difference between a forecaster
    saying "70%" and one saying "70%, and here is my receipts for that 70%".
    """

    def __init__(self, *, min_samples: int = 5) -> None:
        #: Minimum resolved samples in the exact bucket before its accuracy is
        #: treated as a *proven* track record rather than a provisional hint.
        self.min_samples = max(1, int(min_samples))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_record_for(
        self,
        category: Any,
        confidence: float,
        resolved: List[Prediction],
    ) -> TrackRecord:
        """Compute the track record for a (category, confidence) pair."""
        cat = self._coerce_category(category)
        bucket = _confidence_bucket(confidence)

        scored = [
            p for p in resolved
            if p.status in (Status.CORRECT, Status.INCORRECT)
            and p.category == cat
        ]

        # Exact confidence bucket within the category.
        in_bucket = [p for p in scored if _confidence_bucket(p.confidence) == bucket]
        bucket_n = len(in_bucket)
        bucket_correct = sum(1 for p in in_bucket if p.status == Status.CORRECT)
        bucket_acc = (bucket_correct / bucket_n) if bucket_n else None

        # Whole-category fallback.
        cat_n = len(scored)
        cat_correct = sum(1 for p in scored if p.status == Status.CORRECT)
        cat_acc = (cat_correct / cat_n) if cat_n else None

        is_proven = bucket_n >= self.min_samples
        if is_proven:
            basis = "bucket"
        elif cat_n >= self.min_samples:
            basis = "category"
        else:
            basis = "none"

        record = TrackRecord(
            confidence=round(float(confidence), 4),
            confidence_pct=round(float(confidence) * 100),
            category=cat.value,
            bucket=bucket,
            min_samples=self.min_samples,
            bucket_sample_size=bucket_n,
            bucket_correct=bucket_correct,
            bucket_accuracy=round(bucket_acc, 4) if bucket_acc is not None else None,
            category_sample_size=cat_n,
            category_correct=cat_correct,
            category_accuracy=round(cat_acc, 4) if cat_acc is not None else None,
            is_proven=is_proven,
            basis=basis,
        )
        record.phrase = self._phrase(record)
        return record

    def contextualize(
        self,
        predictions: List[Prediction],
        resolved: List[Prediction],
    ) -> List[Prediction]:
        """Populate ``track_record`` on each prediction in-place; return them."""
        for p in predictions:
            record = self.track_record_for(p.category, p.confidence, resolved)
            p.track_record = record.to_dict()
        return predictions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_category(category: Any) -> Category:
        if isinstance(category, Category):
            return category
        return Category(category)

    @staticmethod
    def _phrase(record: TrackRecord) -> str:
        pct = record.confidence_pct
        cat_label = record.category.replace("_", " ")
        if record.basis == "bucket" and record.bucket_accuracy is not None:
            acc = round(record.bucket_accuracy * 100)
            return (
                f"{pct}% confident — '{cat_label}' predictions in this confidence "
                f"band have resolved correct {acc}% of the time across "
                f"{record.bucket_sample_size} resolved cases."
            )
        if record.basis == "category" and record.category_accuracy is not None:
            acc = round(record.category_accuracy * 100)
            return (
                f"{pct}% confident — not enough history yet at this exact "
                f"confidence band ({record.bucket_sample_size} of "
                f"{record.min_samples} needed), but '{cat_label}' predictions "
                f"overall have resolved correct {acc}% of the time across "
                f"{record.category_sample_size} resolved cases."
            )
        return (
            f"{pct}% confident — no proven track record yet for '{cat_label}' "
            f"at this confidence level ({record.bucket_sample_size} resolved, "
            f"{record.min_samples} needed). Treat as provisional until the "
            f"system has resolved more predictions in this category."
        )


__all__ = [
    "CalibrationTracker",
    "_confidence_bucket",
    "TrackRecord",
    "ConfidenceContextualizer",
]
