"""Advanced calibration metrics — Brier score, calibration curves, decomposition.

Pure, dependency-free statistics: reliability curve, ECE/MCE, Murphy
decomposition, and coverage of calibration outcomes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from oracle.models.prediction import Category, Prediction, Status


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CalibrationCurve:
    """Calibration curve data — maps confidence bins to actual accuracy."""
    bins: List[Tuple[float, float]] = field(default_factory=list)  # [(confidence, accuracy)]
    bin_counts: List[int] = field(default_factory=list)
    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0

@dataclass
class DecompositionResult:
    """Discrimination vs calibration decomposition."""
    brier_score: float = 0.0
    refinement: float = 0.0  # Discrimination component
    calibration_component: float = 0.0  # Calibration component
    uncertainty: float = 0.0  # Inherent uncertainty

@dataclass
class AdvancedCalibrationReport:
    """Full calibration report with advanced metrics."""
    overall_total: int = 0
    overall_correct: int = 0
    overall_accuracy: float = 0.0
    brier_score: float = 0.0
    calibration_curve: CalibrationCurve = field(default_factory=CalibrationCurve)
    decomposition: DecompositionResult = field(default_factory=DecompositionResult)
    confidence_coverage: Dict[str, float] = field(default_factory=dict)  # e.g., "80": 0.78
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_well_calibrated(self) -> bool:
        return self.calibration_curve.expected_calibration_error < 0.10

    @property
    def ece(self) -> float:
        """Convenience alias for expected calibration error."""
        return self.calibration_curve.expected_calibration_error


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def compute_brier_score(predictions: List[Prediction]) -> float:
    """Compute Brier score: mean squared error between confidence and outcome.

    Brier = (1/N) * Σ(confidence_i - outcome_i)²
    where outcome_i is 1 if correct, 0 if incorrect.

    Lower is better. 0 = perfect, 0.25 = worst (always 50% wrong).
    """
    resolved = [p for p in predictions if p.status in (Status.CORRECT, Status.INCORRECT)]
    if not resolved:
        return 0.0

    total = 0.0
    for p in resolved:
        outcome = 1.0 if p.status == Status.CORRECT else 0.0
        total += (p.confidence - outcome) ** 2

    return total / len(resolved)


def compute_calibration_curve(
    predictions: List[Prediction],
    num_bins: int = 10,
) -> CalibrationCurve:
    """Compute calibration curve with equal-frequency binning.

    Groups predictions into confidence bins and measures actual accuracy
    in each bin. A well-calibrated model has accuracy ≈ confidence in each bin.

    Also computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    resolved = [p for p in predictions if p.status in (Status.CORRECT, Status.INCORRECT)]
    if not resolved:
        return CalibrationCurve()

    # Sort by confidence
    sorted_preds = sorted(resolved, key=lambda p: p.confidence)
    n = len(sorted_preds)
    bin_size = max(1, n // num_bins)

    bins: List[Tuple[float, float]] = []
    counts: List[int] = []
    ece_total = 0.0
    mce = 0.0

    for i in range(0, n, bin_size):
        bin_preds = sorted_preds[i:i + bin_size]
        if not bin_preds:
            continue

        avg_conf = sum(p.confidence for p in bin_preds) / len(bin_preds)
        accuracy = sum(1 for p in bin_preds if p.status == Status.CORRECT) / len(bin_preds)

        bins.append((round(avg_conf, 3), round(accuracy, 3)))
        counts.append(len(bin_preds))

        error = abs(avg_conf - accuracy)
        ece_total += error * len(bin_preds)
        mce = max(mce, error)

    ece = ece_total / n if n > 0 else 0.0

    return CalibrationCurve(
        bins=bins,
        bin_counts=counts,
        expected_calibration_error=round(ece, 4),
        max_calibration_error=round(mce, 4),
    )


def compute_decomposition(predictions: List[Prediction]) -> DecompositionResult:
    """Decompose Brier score into discrimination, calibration, and uncertainty.

    Uses the Murphy decomposition:
    Brier = Uncertainty - Resolution + Reliability
           = uncertainty - discrimination + calibration

    - Uncertainty: inherent difficulty (always predicting base rate)
    - Discrimination (Resolution): how well predictions separate outcomes
    - Calibration (Reliability): how well calibrated are the probabilities
    """
    resolved = [p for p in predictions if p.status in (Status.CORRECT, Status.INCORRECT)]
    if not resolved:
        return DecompositionResult()

    n = len(resolved)
    outcomes = [1.0 if p.status == Status.CORRECT else 0.0 for p in resolved]
    confidences = [p.confidence for p in resolved]

    # Base rate
    base_rate = sum(outcomes) / n

    # Brier score
    brier = sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / n

    # Uncertainty = base_rate * (1 - base_rate)
    uncertainty = base_rate * (1 - base_rate)

    # Bin predictions for reliability/resolution
    bins = defaultdict(list)
    for c, o in zip(confidences, outcomes):
        bin_key = round(c * 20) / 20  # 0.05-width bins
        bins[bin_key].append((c, o))

    # Reliability (calibration component)
    reliability = 0.0
    for bin_key, items in bins.items():
        n_k = len(items)
        if n_k == 0:
            continue
        avg_conf = sum(c for c, _ in items) / n_k
        avg_outcome = sum(o for _, o in items) / n_k
        reliability += n_k * (avg_conf - avg_outcome) ** 2
    reliability /= n

    # Resolution (discrimination)
    resolution = 0.0
    for bin_key, items in bins.items():
        n_k = len(items)
        if n_k == 0:
            continue
        avg_outcome = sum(o for _, o in items) / n_k
        resolution += n_k * (avg_outcome - base_rate) ** 2
    resolution /= n

    return DecompositionResult(
        brier_score=round(brier, 4),
        refinement=round(resolution, 4),
        calibration_component=round(reliability, 4),
        uncertainty=round(uncertainty, 4),
    )


def compute_confidence_coverage(predictions: List[Prediction]) -> Dict[str, float]:
    """Check if confidence intervals actually cover the true outcome rate.

    For example, if you say "80% confidence," your 80% predictions
    should be correct ~80% of the time.
    """
    resolved = [p for p in predictions if p.status in (Status.CORRECT, Status.INCORRECT)]
    if not resolved:
        return {}

    coverage = {}
    for level in [50, 60, 70, 75, 80, 85, 90, 95]:
        # Find predictions with confidence >= this level
        level_preds = [p for p in resolved if p.confidence * 100 >= level]
        if level_preds:
            accuracy = sum(1 for p in level_preds if p.status == Status.CORRECT) / len(level_preds)
            coverage[str(level)] = round(accuracy, 3)

    return coverage


def compute_full_report(
    predictions: List[Prediction],
    category: Optional[Category] = None,
) -> AdvancedCalibrationReport:
    """Compute a full advanced calibration report."""
    resolved = [p for p in predictions if p.status in (Status.CORRECT, Status.INCORRECT)]
    if category:
        resolved = [p for p in resolved if p.category == category]

    total = len(resolved)
    correct = sum(1 for p in resolved if p.status == Status.CORRECT)
    accuracy = correct / total if total > 0 else 0.0

    brier = compute_brier_score(predictions)
    curve = compute_calibration_curve(predictions)
    decomp = compute_decomposition(predictions)
    coverage = compute_confidence_coverage(predictions)

    # By category
    by_cat: Dict[str, Dict[str, float]] = {}
    for cat in {p.category for p in resolved}:
        cat_preds = [p for p in resolved if p.category == cat]
        cat_correct = sum(1 for p in cat_preds if p.status == Status.CORRECT)
        cat_brier = compute_brier_score(cat_preds)
        by_cat[cat.value] = {
            "total": len(cat_preds),
            "correct": cat_correct,
            "accuracy": round(cat_correct / len(cat_preds), 4) if cat_preds else 0,
            "brier_score": round(cat_brier, 4),
        }

    return AdvancedCalibrationReport(
        overall_total=total,
        overall_correct=correct,
        overall_accuracy=round(accuracy, 4),
        brier_score=round(brier, 4),
        calibration_curve=curve,
        decomposition=decomp,
        confidence_coverage=coverage,
        by_category=by_cat,
    )


__all__ = [
    "CalibrationCurve", "DecompositionResult", "AdvancedCalibrationReport",
    "compute_brier_score", "compute_calibration_curve",
    "compute_decomposition", "compute_confidence_coverage",
    "compute_full_report",
]
