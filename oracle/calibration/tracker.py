"""Calibration tracking and reporting for The Oracle.

Tracks resolved predictions, computes accuracy per confidence bucket,
Brier scores, and generates calibration reports.

All computation is pure — no LLM calls needed for calibration math.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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


__all__ = ["CalibrationTracker", "_confidence_bucket"]
