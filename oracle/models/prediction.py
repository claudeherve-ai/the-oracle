"""Prediction models for The Oracle.

Pydantic models shared across the engine: predictions, evidence, ensemble
votes, resolution results, and verification reports.
"""

from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import uuid
from pydantic import BaseModel, ConfigDict, Field


class Category(str, Enum):
    TECH_TREND = "tech_trend"
    PRODUCT_LAUNCH = "product_launch"
    MARKET_MOVE = "market_move"
    REGULATORY = "regulatory"
    STARTUP_SUCCESS = "startup_success"
    CULTURE = "culture"
    GITHUB_TREND = "github_trend"


class Status(str, Enum):
    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    EXPIRED = "expired"
    #: The system explicitly declined to stand behind this prediction because
    #: the available evidence was insufficient, contradictory, or could not be
    #: corroborated. An abstention — NOT a forecast. It is deliberately excluded
    #: from every calibration denominator (a refusal is neither right nor wrong),
    #: which is what lets the system honestly say "I don't know."
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Prediction(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: Category
    statement: str = Field(..., min_length=10, max_length=1000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    sources: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None
    status: Status = Status.PENDING
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # NEW: ensemble fields
    model_disagreement: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Inter-model disagreement score (0=total agreement, 1=max disagreement)"
    )
    confidence_interval_lower: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Lower bound of ensemble confidence interval"
    )
    confidence_interval_upper: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Upper bound of ensemble confidence interval"
    )
    contributing_models: List[str] = Field(
        default_factory=list,
        description="Models that contributed to this prediction"
    )
    #: Contextualized track record for this prediction's confidence + category,
    #: computed from resolved history. Lets the UI show "70% confident — this
    #: category has been right 68% of the time across 142 resolved predictions
    #: at this confidence level" instead of a bare, unaudited number. ``None``
    #: until :meth:`oracle.prediction.engine.PredictionEngine.contextualize`
    #: (or the API) populates it. See ``oracle.calibration.tracker.TrackRecord``.
    track_record: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Historical accuracy context for this confidence/category",
    )

    @property
    def is_abstention(self) -> bool:
        """True when the system declined to forecast (insufficient evidence)."""
        return self.status == Status.INSUFFICIENT_EVIDENCE


class Signal(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    content: str
    entities: List[str] = Field(default_factory=list)
    sentiment: float = Field(0.0, ge=-1.0, le=1.0)
    relevance: float = Field(0.5, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Ensemble models ──────────────────────────────────────────────


class ModelVote(BaseModel):
    """A single model's vote in the ensemble."""
    model_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    prompt_variant: str = "balanced"  # conservative | balanced | aggressive


class EnsemblePrediction(BaseModel):
    """Aggregated prediction from multiple models."""
    statement: str
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_interval_lower: float = Field(..., ge=0.0, le=1.0)
    confidence_interval_upper: float = Field(..., ge=0.0, le=1.0)
    model_disagreement: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    sources: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None
    votes: List[ModelVote] = Field(default_factory=list)
    contributing_models: List[str] = Field(default_factory=list)


# ── Verification models ─────────────────────────────────────────


class EvidenceItem(BaseModel):
    """A piece of evidence from web/financial/news search."""
    source_url: str
    source_name: str = "web"
    title: str = ""
    snippet: str = ""
    supports: bool = False  # True=supporting, False=contradicting
    stance: str = "neutral"  # supports | contradicts | neutral (from NLI)
    quote: str = ""  # exact verbatim span proving support/contradiction
    credibility_score: float = Field(0.5, ge=0.0, le=1.0)


class VerifiedPrediction(BaseModel):
    """A prediction with verification results."""
    prediction_id: str
    statement: str
    original_confidence: float
    adjusted_confidence: float
    verdict: str = "unverifiable"  # supported | contradicted | unverifiable
    verification_note: str = ""
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Full verification report for a set of predictions."""
    verified_predictions: List[VerifiedPrediction] = Field(default_factory=list)
    overall_reliability: float = 0.0
    total_evidence_items: int = 0
    summary: str = ""


# ── Resolution models ───────────────────────────────────────────


class EvidenceSnapshot(BaseModel):
    """A point-in-time capture of one source used to resolve a prediction.

    This is what makes a resolution *auditable months later*: it records not
    just the URL but the exact text that was read, a tamper-evident hash of that
    text, the verbatim ``quote`` the judge stood on, and the surrounding
    context. Markets, repos, and news pages mutate or vanish — the snapshot is
    the frozen evidence the label was actually derived from.
    """

    url: str
    canonical_url: str = ""  # normalized URL used for de-duplication
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fetch_ok: bool = True
    fetch_error: str = ""
    stance: str = "neutral"  # supports | contradicts | neutral (verified NLI verdict)
    quote: str = ""  # exact verbatim span the judge relied on (verified in-text)
    snippet: str = ""  # surrounding context window around the quote
    content_hash: str = ""  # sha256 of the fetched text (tamper-evidence)
    content_chars: int = 0
    content_truncated: bool = False


class ResolutionResult(BaseModel):
    """Result of auto-resolving a single prediction.

    Carries the full audit trail: the exact normalized claim that was judged,
    a machine-readable ``resolution_reason`` code, and the evidence snapshots
    the label was derived from. ``new_status`` may be INSUFFICIENT_EVIDENCE — an
    honest abstention that is excluded from calibration rather than a guess.
    """
    prediction_id: str
    statement: str
    previous_status: str
    new_status: Status
    resolution: str = ""
    resolution_claim: str = ""  # the deadline-baked claim actually judged
    resolution_reason: str = ""  # audit code: resolved | all_neutral | etc.
    confidence: str = "low"  # low | medium | high
    reasoning: str = ""
    evidence_urls: List[str] = Field(default_factory=list)
    evidence_snapshots: List[EvidenceSnapshot] = Field(default_factory=list)
    requires_human_review: bool = False
    schema_version: int = 1
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResolutionReport(BaseModel):
    """Report from an auto-resolution run."""
    scanned: int = 0
    resolved: int = 0
    expired: int = 0
    results: List[ResolutionResult] = Field(default_factory=list)
    summary: str = ""


# ── Calibration models ──────────────────────────────────────────


class CalibrationBucket(BaseModel):
    category: Category
    confidence_range: str  # "0.5-0.6", "0.6-0.7", etc.
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class CalibrationCurve(BaseModel):
    """Full calibration curve data for plotting and analysis."""
    points: List[dict] = Field(default_factory=list)
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0
    discrimination_auc: float = 0.0  # Area under ROC — how well we separate correct/incorrect
    confidence_interval_coverage: float = 0.0  # Fraction where CI contains actual outcome
    sharpe_ratio: float = 0.0  # Sharpe-like ratio for prediction quality
    n_predictions: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationReport(BaseModel):
    overall_total: int = 0
    overall_correct: int = 0
    buckets: List[CalibrationBucket] = Field(default_factory=list)
    by_category: dict = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Advanced metrics (populated when advanced=True)
    advanced: Optional[CalibrationCurve] = None

    @property
    def overall_accuracy(self) -> float:
        return self.overall_correct / self.overall_total if self.overall_total > 0 else 0.0


# ── API request models ──────────────────────────────────────────


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000)
    categories: Optional[List[Category]] = None
    max_predictions: int = Field(default=5, ge=1, le=20)


class EnsemblePredictRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000)
    categories: Optional[List[Category]] = None
    max_predictions: int = Field(default=5, ge=1, le=20)
    models: Optional[List[str]] = None  # Model names to use, defaults to all available
    prompt_variants: Optional[List[str]] = None  # ["conservative", "balanced", "aggressive"]


class AutoResolveRequest(BaseModel):
    """Request to trigger auto-resolution of past-deadline predictions."""
    dry_run: bool = False  # If True, report what would change without committing
    max_to_resolve: int = Field(default=50, ge=1, le=200)


class AutoIngestRequest(BaseModel):
    """Request to trigger full ingestion pipeline."""
    sources: Optional[List[str]] = None  # e.g., ["news", "financial", "github", "arxiv", "social"]
    max_signals: int = Field(default=200, ge=1, le=1000)


class ResolveRequest(BaseModel):
    outcome: Status = Field(...)
    resolution: Optional[str] = None
