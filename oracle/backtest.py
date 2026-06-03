"""Backtest harness — replay resolved historical questions through calibration.

The single most important rule of a trustworthy forecaster is: *prove it on the
past before you ask anyone to trust it on the future.* This module replays a set
of already-resolved predictions through the exact same calibration math the live
system uses, and answers two questions an auditor would actually ask:

1. **Static calibration** — over the whole resolved history, do the confidence
   numbers match realized accuracy? (Brier score, reliability curve, ECE, a
   per-confidence-bucket predicted-vs-realized table.) This is "is the system
   well-calibrated?".

2. **Rolling honesty** — at the moment each prediction was made, the system
   could attach a :class:`~oracle.calibration.tracker.TrackRecord` claiming a
   historical accuracy ("this category has been right 68% of the time"). Using
   ONLY the predictions resolved *strictly before* each one, we recompute that
   claim and check it against what actually happened next. If the system's
   advertised track record systematically over- or under-states reality, this
   surfaces it. This is "was the system honest about its own history, in real
   time, without hindsight?".

Pure and dependency-free: no network, no LLM, no clock dependence beyond the
``created_at`` already on each prediction. Fully deterministic, so it can run in
CI and back a public "audit me" dashboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from oracle.calibration.metrics import (
    AdvancedCalibrationReport,
    compute_full_report,
    serialize_full_report,
)
from oracle.calibration.tracker import ConfidenceContextualizer, _confidence_bucket
from oracle.models.prediction import Category, Prediction, Status

#: Statuses that contribute to calibration. An abstention / pending / expired
#: prediction is neither right nor wrong and never enters a denominator.
_SCORED = (Status.CORRECT, Status.INCORRECT)

#: Expected-calibration-error threshold below which the system is considered
#: well-calibrated. Mirrors ``AdvancedCalibrationReport.is_well_calibrated`` so
#: the backtest verdict and the live dashboard agree.
WELL_CALIBRATED_ECE = 0.10


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityRow:
    """One confidence bucket's predicted-vs-realized comparison."""

    bucket: str
    n: int
    predicted_confidence: float  # mean stated confidence in the bucket
    realized_accuracy: float     # fraction actually CORRECT
    gap: float                   # predicted_confidence - realized_accuracy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RollingPoint:
    """A single chronological replay step (no hindsight).

    Captures what the contextualizer *would have claimed* about historical
    accuracy at the instant this prediction was made, alongside the outcome that
    later materialized, so the two can be compared honestly.
    """

    prediction_id: str
    category: str
    confidence: float
    bucket: str
    outcome: int                       # 1 if CORRECT, 0 if INCORRECT
    claimed_accuracy: Optional[float]  # track-record accuracy known at the time
    claimed_basis: str                 # "bucket" | "category" | "none"
    prior_scored_samples: int          # resolved cases available before this one

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestReport:
    """Aggregate result of replaying resolved predictions through calibration."""

    total: int
    resolved: int
    skipped_unresolved: int
    report: AdvancedCalibrationReport
    reliability: List[ReliabilityRow] = field(default_factory=list)
    rolling_points: List[RollingPoint] = field(default_factory=list)
    mean_calibration_gap: float = 0.0          # n-weighted mean |conf - acc| (ECE-like)
    rolling_claim_coverage: int = 0            # rolling points that carried a claim
    rolling_honesty_gap: Optional[float] = None  # |mean(claim) - mean(outcome)| over claims
    well_calibrated: bool = False
    verdict: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict suitable for an audit endpoint or dashboard."""
        return {
            "total": self.total,
            "resolved": self.resolved,
            "skipped_unresolved": self.skipped_unresolved,
            "report": serialize_full_report(self.report),
            "reliability": [r.to_dict() for r in self.reliability],
            "rolling_points": [p.to_dict() for p in self.rolling_points],
            "mean_calibration_gap": round(self.mean_calibration_gap, 6),
            "rolling_claim_coverage": self.rolling_claim_coverage,
            "rolling_honesty_gap": (
                round(self.rolling_honesty_gap, 6)
                if self.rolling_honesty_gap is not None
                else None
            ),
            "well_calibrated": self.well_calibrated,
            "verdict": self.verdict,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _scored(predictions: List[Prediction], category: Optional[Category]) -> List[Prediction]:
    out = [p for p in predictions if p.status in _SCORED]
    if category is not None:
        out = [p for p in out if p.category == category]
    return out


def _reliability_table(scored: List[Prediction]) -> List[ReliabilityRow]:
    """Per-confidence-bucket predicted-vs-realized accuracy table."""
    buckets: Dict[str, List[Prediction]] = {}
    for p in scored:
        buckets.setdefault(_confidence_bucket(p.confidence), []).append(p)

    rows: List[ReliabilityRow] = []
    for bucket in sorted(buckets):
        members = buckets[bucket]
        n = len(members)
        predicted = sum(p.confidence for p in members) / n
        realized = sum(1 for p in members if p.status == Status.CORRECT) / n
        rows.append(
            ReliabilityRow(
                bucket=bucket,
                n=n,
                predicted_confidence=round(predicted, 4),
                realized_accuracy=round(realized, 4),
                gap=round(predicted - realized, 4),
            )
        )
    return rows


def _mean_calibration_gap(rows: List[ReliabilityRow]) -> float:
    """Sample-size-weighted mean absolute predicted-vs-realized gap (ECE-like)."""
    total_n = sum(r.n for r in rows)
    if total_n == 0:
        return 0.0
    return sum(abs(r.gap) * r.n for r in rows) / total_n


def _rolling_replay(
    scored: List[Prediction],
    *,
    min_samples: int,
) -> List[RollingPoint]:
    """Chronological no-hindsight replay of the contextualizer's claims.

    Predictions are processed in ``created_at`` order. For each one, the track
    record is recomputed from ONLY the predictions resolved before it, so the
    claimed historical accuracy never peeks at the outcome it is about to be
    judged against.
    """
    ordered = sorted(scored, key=lambda p: (p.created_at or datetime.min.replace(tzinfo=timezone.utc)))
    contextualizer = ConfidenceContextualizer(min_samples=min_samples)
    history: List[Prediction] = []
    points: List[RollingPoint] = []

    for p in ordered:
        record = contextualizer.track_record_for(p.category, p.confidence, history)
        if record.basis == "bucket":
            claimed = record.bucket_accuracy
        elif record.basis == "category":
            claimed = record.category_accuracy
        else:
            claimed = None

        points.append(
            RollingPoint(
                prediction_id=p.id,
                category=p.category.value,
                confidence=round(float(p.confidence), 4),
                bucket=_confidence_bucket(p.confidence),
                outcome=1 if p.status == Status.CORRECT else 0,
                claimed_accuracy=claimed,
                claimed_basis=record.basis,
                prior_scored_samples=len(history),
            )
        )
        history.append(p)

    return points


def _rolling_honesty_gap(points: List[RollingPoint]) -> Optional[float]:
    """Aggregate gap between advertised historical accuracy and reality.

    Over every replay step that actually advertised a track record, compares the
    mean claimed accuracy to the mean realized outcome. A near-zero gap means the
    system's self-reported history was honest in real time; a large gap means it
    systematically over- or under-promised.
    """
    claims = [pt for pt in points if pt.claimed_accuracy is not None]
    if not claims:
        return None
    mean_claimed = sum(pt.claimed_accuracy for pt in claims) / len(claims)  # type: ignore[arg-type]
    mean_outcome = sum(pt.outcome for pt in claims) / len(claims)
    return abs(mean_claimed - mean_outcome)


def _verdict(
    *,
    resolved: int,
    well_calibrated: bool,
    mean_gap: float,
    honesty_gap: Optional[float],
    min_for_trust: int,
) -> str:
    if resolved == 0:
        return (
            "No resolved predictions to backtest. The system has no track record "
            "yet — treat every live confidence as provisional."
        )
    if resolved < min_for_trust:
        return (
            f"Only {resolved} resolved predictions — below the {min_for_trust} "
            "needed for a statistically meaningful verdict. Calibration shown is "
            "preliminary; keep resolving before trusting it on live questions."
        )
    if well_calibrated:
        msg = (
            f"Well-calibrated over {resolved} resolved predictions "
            f"(mean confidence-vs-accuracy gap {mean_gap:.1%}, ECE under "
            f"{WELL_CALIBRATED_ECE:.0%})."
        )
    else:
        msg = (
            f"NOT well-calibrated over {resolved} resolved predictions "
            f"(mean confidence-vs-accuracy gap {mean_gap:.1%}, ECE at or above "
            f"{WELL_CALIBRATED_ECE:.0%}). Confidence numbers should be "
            "down-weighted until calibration improves."
        )
    if honesty_gap is not None:
        msg += (
            f" Real-time track-record honesty gap: {honesty_gap:.1%} between "
            "advertised historical accuracy and what actually happened next."
        )
    return msg


def run_backtest(
    predictions: List[Prediction],
    *,
    category: Optional[Category] = None,
    min_samples: int = 5,
    min_for_trust: int = 20,
) -> BacktestReport:
    """Replay resolved predictions through the live calibration math.

    Args:
        predictions: Any mix of predictions; only ``CORRECT``/``INCORRECT`` are
            scored. Pending, expired, and abstained predictions are counted as
            skipped, never as right or wrong.
        category: Optional filter to backtest a single category in isolation.
        min_samples: Track-record proof threshold passed to the contextualizer
            during the rolling replay (mirrors the live default).
        min_for_trust: Minimum resolved count before the verdict is allowed to
            declare the system trustworthy rather than preliminary.

    Returns:
        A :class:`BacktestReport` carrying the full advanced calibration report,
        a predicted-vs-realized reliability table, the chronological rolling
        replay, and an overall verdict.
    """
    total = len(predictions)
    scored = _scored(predictions, category)
    resolved = len(scored)
    skipped = total - resolved

    report = compute_full_report(predictions, category=category)
    reliability = _reliability_table(scored)
    mean_gap = _mean_calibration_gap(reliability)
    rolling = _rolling_replay(scored, min_samples=min_samples)
    honesty_gap = _rolling_honesty_gap(rolling)
    well_calibrated = bool(report.is_well_calibrated) and resolved > 0
    claim_coverage = sum(1 for pt in rolling if pt.claimed_accuracy is not None)

    verdict = _verdict(
        resolved=resolved,
        well_calibrated=well_calibrated,
        mean_gap=mean_gap,
        honesty_gap=honesty_gap,
        min_for_trust=min_for_trust,
    )

    return BacktestReport(
        total=total,
        resolved=resolved,
        skipped_unresolved=skipped,
        report=report,
        reliability=reliability,
        rolling_points=rolling,
        mean_calibration_gap=mean_gap,
        rolling_claim_coverage=claim_coverage,
        rolling_honesty_gap=honesty_gap,
        well_calibrated=well_calibrated,
        verdict=verdict,
    )


__all__ = [
    "WELL_CALIBRATED_ECE",
    "ReliabilityRow",
    "RollingPoint",
    "BacktestReport",
    "run_backtest",
]
