"""Oracle calibration tracking and reporting.

Computes accuracy per confidence bucket, Brier scores, and calibration
reports from resolved predictions. All computation is pure — no LLM needed.
"""

from oracle.calibration.tracker import CalibrationTracker, _confidence_bucket

__all__ = [
    "CalibrationTracker",
    "_confidence_bucket",
]
