"""Repository layer for predictions and signals."""

from typing import Optional, List
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from oracle.storage.database import PredictionRecord, SignalRecord
from oracle.models.prediction import Prediction, Signal, Category, Status


class PredictionRepository:
    def __init__(self, session: Session):
        self.session = session

    async def create(self, p: Prediction) -> Prediction:
        record = PredictionRecord(
            id=p.id, category=p.category.value, statement=p.statement,
            confidence=p.confidence, reasoning=p.reasoning, sources=p.sources,
            deadline=p.deadline, status=p.status.value,
            resolution=p.resolution, resolved_at=p.resolved_at,
            created_at=p.created_at,
        )
        self.session.add(record)
        self.session.commit()
        return p

    async def get(self, pid: str) -> Optional[Prediction]:
        r = self.session.get(PredictionRecord, pid)
        return self._to_model(r) if r else None

    async def list_all(self, category: Optional[Category] = None,
                       status: Optional[Status] = None,
                       limit: int = 50) -> List[Prediction]:
        q = self.session.query(PredictionRecord)
        if category:
            q = q.filter(PredictionRecord.category == category.value)
        if status:
            q = q.filter(PredictionRecord.status == status.value)
        records = q.order_by(PredictionRecord.created_at.desc()).limit(limit).all()
        return [self._to_model(r) for r in records]

    async def get_resolved(self) -> List[Prediction]:
        records = (
            self.session.query(PredictionRecord)
            .filter(PredictionRecord.status.in_(["correct", "incorrect"]))
            .all()
        )
        return [self._to_model(r) for r in records]

    async def resolve(self, pid: str, outcome: Status, resolution: Optional[str] = None) -> Optional[Prediction]:
        r = self.session.get(PredictionRecord, pid)
        if not r:
            return None
        r.status = outcome.value
        r.resolution = resolution
        r.resolved_at = datetime.now(timezone.utc)
        self.session.commit()
        return self._to_model(r)

    @staticmethod
    def _to_model(r: PredictionRecord) -> Prediction:
        return Prediction(
            id=r.id, category=r.category, statement=r.statement,
            confidence=r.confidence, reasoning=r.reasoning or "",
            sources=r.sources or [], deadline=r.deadline, status=r.status,
            resolution=r.resolution, resolved_at=r.resolved_at,
            created_at=r.created_at,
        )


class SignalRepository:
    def __init__(self, session: Session):
        self.session = session

    async def create(self, s: Signal) -> Signal:
        record = SignalRecord(
            id=s.id, source=s.source, content=s.content,
            entities=s.entities, sentiment=s.sentiment,
            relevance=s.relevance, extra_data=s.metadata,
            captured_at=s.captured_at,
        )
        self.session.add(record)
        self.session.commit()
        return s

    async def create_batch(self, signals: List[Signal]) -> int:
        for s in signals:
            record = SignalRecord(
                id=s.id, source=s.source, content=s.content,
                entities=s.entities, sentiment=s.sentiment,
                relevance=s.relevance, metadata=s.metadata,
                captured_at=s.captured_at,
            )
            self.session.add(record)
        self.session.commit()
        return len(signals)

    async def list_recent(self, limit: int = 50, source: Optional[str] = None) -> List[Signal]:
        q = self.session.query(SignalRecord).order_by(SignalRecord.captured_at.desc())
        if source:
            q = q.filter(SignalRecord.source == source)
        records = q.limit(limit).all()
        return [self._to_model(r) for r in records]

    @staticmethod
    def _to_model(r: SignalRecord) -> Signal:
        return Signal(
            id=r.id, source=r.source, content=r.content,
            entities=r.entities or [], sentiment=r.sentiment or 0.0,
            relevance=r.relevance or 0.5, metadata=r.extra_data or {},
            captured_at=r.captured_at,
        )
