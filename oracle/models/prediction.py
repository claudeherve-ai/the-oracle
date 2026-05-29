"""Prediction models."""

from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List
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


class CalibrationBucket(BaseModel):
    category: Category
    confidence_range: str  # "0.5-0.6", "0.6-0.7", etc.
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class CalibrationReport(BaseModel):
    overall_total: int = 0
    overall_correct: int = 0
    buckets: List[CalibrationBucket] = Field(default_factory=list)
    by_category: dict = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def overall_accuracy(self) -> float:
        return self.overall_correct / self.overall_total if self.overall_total > 0 else 0.0


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000)
    categories: Optional[List[Category]] = None
    max_predictions: int = Field(default=5, ge=1, le=20)


class ResolveRequest(BaseModel):
    outcome: Status = Field(...)
    resolution: Optional[str] = None
