"""Per-prediction audit trail for The Oracle.

Reproducibility is trust. Every prediction-generation run can emit a structured
:class:`AuditRecord` capturing *exactly* what the system saw and decided — the
prompts, the model, the sources it fetched, the evidence spans it relied on, and
the confidence **before and after** verification. Anyone can replay, inspect, or
challenge a call after the fact.

Design goals
------------
* **Opt-in, zero-overhead by default.** The engine only builds a record when an
  :class:`AuditSink` is wired in, so existing behaviour and tests are unchanged.
* **Structured, machine-readable.** Records serialise to plain dicts / single
  JSON lines so they drop straight into log pipelines or an audit table.
* **Honest.** It records what happened, including abstentions and
  confidence that *dropped* after verification — never a flattering summary.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "EvidenceSpan",
    "PredictionAudit",
    "AuditRecord",
    "AuditSink",
    "InMemoryAuditSink",
    "LoggingAuditSink",
    "new_audit_logger",
]

_AUDIT_LOGGER_NAME = "oracle.audit"


def new_audit_logger() -> logging.Logger:
    """Return the canonical audit logger (``oracle.audit``)."""
    return logging.getLogger(_AUDIT_LOGGER_NAME)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class EvidenceSpan:
    """A single quoted evidence span the verifier relied on.

    Quotes, never summaries — if the system can't point at the verbatim text and
    its URL, it doesn't count as evidence.
    """

    url: str = ""
    quote: str = ""
    stance: str = "neutral"  # supports | contradicts | neutral
    credibility: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "quote": self.quote,
            "stance": self.stance,
            "credibility": round(float(self.credibility), 4),
        }


@dataclass
class PredictionAudit:
    """Audit detail for a single prediction within a generation run."""

    prediction_id: str
    statement: str
    category: str
    status: str
    confidence_pre: float
    confidence_post: float
    verdict: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    evidence_spans: List[EvidenceSpan] = field(default_factory=list)

    @property
    def confidence_delta(self) -> float:
        """How much verification moved the confidence (post - pre)."""
        return round(self.confidence_post - self.confidence_pre, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "statement": self.statement,
            "category": self.category,
            "status": self.status,
            "confidence_pre": round(float(self.confidence_pre), 4),
            "confidence_post": round(float(self.confidence_post), 4),
            "confidence_delta": self.confidence_delta,
            "verdict": self.verdict,
            "sources": list(self.sources),
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
        }


@dataclass
class AuditRecord:
    """A complete, replayable record of one ``PredictionEngine.generate`` call."""

    model: str
    question: Optional[str] = None
    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    verification_mode: str = "off"
    grounding_present: bool = False
    grounding_chars: int = 0
    predictions: List[PredictionAudit] = field(default_factory=list)
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "created_at": self.created_at.isoformat(),
            "model": self.model,
            "question": self.question,
            "verification_mode": self.verification_mode,
            "grounding_present": self.grounding_present,
            "grounding_chars": self.grounding_chars,
            "prediction_count": len(self.predictions),
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response,
            "predictions": [p.to_dict() for p in self.predictions],
        }

    def to_json(self) -> str:
        """Serialise to a single JSON line for structured logging."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


@runtime_checkable
class AuditSink(Protocol):
    """Anything that can durably accept an :class:`AuditRecord`."""

    def record(self, record: AuditRecord) -> None:  # pragma: no cover - protocol
        ...


class InMemoryAuditSink:
    """Collects records in memory — ideal for tests and short-lived sessions."""

    def __init__(self) -> None:
        self.records: List[AuditRecord] = []

    def record(self, record: AuditRecord) -> None:
        self.records.append(record)

    @property
    def last(self) -> Optional[AuditRecord]:
        return self.records[-1] if self.records else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)


class LoggingAuditSink:
    """Emits each record as one structured JSON line via the audit logger."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        *,
        level: int = logging.INFO,
    ) -> None:
        self._logger = logger or new_audit_logger()
        self._level = level

    def record(self, record: AuditRecord) -> None:
        self._logger.log(self._level, "audit %s", record.to_json())
