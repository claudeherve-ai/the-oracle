"""API routes for The Oracle."""

import asyncio
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query

from oracle.models.prediction import (
    Prediction, Signal, Category, Status,
    PredictRequest, EnsemblePredictRequest, ResolveRequest, CalibrationReport,
)
from oracle.storage.repository import PredictionRepository, SignalRepository
from oracle.prediction.engine import PredictionEngine
from oracle.prediction.ensemble import EnsembleEngine
from oracle.calibration.tracker import CalibrationTracker
from oracle.ingestion.sources import ingest_all
from oracle.signals.extractor import SignalExtractor
from oracle.api.dependencies import get_llm, get_prediction_repo, get_signal_repo

logger = logging.getLogger("oracle.api")
router = APIRouter(prefix="/v1")


# ── Predict ──────────────────────────────────────────────────


@router.post("/predict", status_code=201)
async def predict(
    body: PredictRequest,
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
    signal_repo: SignalRepository = Depends(get_signal_repo),
):
    """Generate predictions from a question + current signals."""
    llm = get_llm()
    engine = PredictionEngine(llm)
    signals = await signal_repo.list_recent(limit=50)

    if body.question:
        predictions = await engine.generate_from_question(
            body.question, signals, max_predictions=body.max_predictions
        )
    else:
        predictions = await engine.scan(signals, body.max_predictions)

    for p in predictions:
        await pred_repo.create(p)

    return {"predictions": [p.model_dump() for p in predictions], "count": len(predictions)}


@router.post("/predict/query", status_code=201)
async def predict_query(
    body: PredictRequest,
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
    signal_repo: SignalRepository = Depends(get_signal_repo),
):
    """Generate predictions for a specific question."""
    llm = get_llm()
    engine = PredictionEngine(llm)
    signals = await signal_repo.list_recent(limit=50)
    predictions = await engine.generate_from_question(
        body.question, signals, max_predictions=body.max_predictions
    )

    for p in predictions:
        await pred_repo.create(p)

    return {"predictions": [p.model_dump() for p in predictions], "count": len(predictions)}


@router.post("/predict/ensemble", status_code=201)
async def predict_ensemble(
    body: EnsemblePredictRequest,
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
    signal_repo: SignalRepository = Depends(get_signal_repo),
):
    """Generate ensemble predictions across N prompt variants / models.

    Disagreement between runs is surfaced as a first-class uncertainty signal:
    "4 of 5 runs agreed" is far more trustworthy than a single confident
    sample. Callers should treat a high ``disagreement_score`` as a reason to
    distrust the point estimate.
    """
    llm = get_llm()
    ensemble = EnsembleEngine(llm, prompt_variants=body.prompt_variants or None)
    signals = await signal_repo.list_recent(limit=50)

    result = await ensemble.generate(
        signals,
        question=body.question,
        categories=body.categories,
        max_predictions=body.max_predictions,
    )

    for p in result.predictions:
        await pred_repo.create(p)

    return {
        "predictions": [p.model_dump() for p in result.predictions],
        "count": len(result.predictions),
        "disagreement_score": result.disagreement_score,
        "models_used": result.models_used,
        "variants_used": result.variants_used,
        "model_details": result.model_details,
    }


# ── Predictions ──────────────────────────────────────────────


@router.get("/predictions")
async def list_predictions(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
):
    """List predictions with optional filters."""
    cat = Category(category) if category else None
    st = Status(status) if status else None
    predictions = await pred_repo.list_all(category=cat, status=st, limit=limit)
    return {"items": [p.model_dump() for p in predictions], "total": len(predictions)}


@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
):
    """Get a specific prediction."""
    p = await pred_repo.get(prediction_id)
    if not p:
        raise HTTPException(404, "Prediction not found")
    return p.model_dump()


@router.post("/predictions/{prediction_id}/resolve")
async def resolve_prediction(
    prediction_id: str,
    body: ResolveRequest,
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
):
    """Resolve a prediction as correct or incorrect."""
    p = await pred_repo.resolve(prediction_id, body.outcome, body.resolution)
    if not p:
        raise HTTPException(404, "Prediction not found")
    return p.model_dump()


# ── Calibration ──────────────────────────────────────────────


@router.get("/calibration")
async def calibration(
    category: Optional[str] = Query(None),
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
):
    """Get calibration report."""
    tracker = CalibrationTracker()
    resolved = await pred_repo.get_resolved()
    cat = Category(category) if category else None
    report = tracker.compute(resolved, category_filter=cat)
    return report.model_dump()


# ── Signals ──────────────────────────────────────────────────


@router.get("/signals")
async def list_signals(
    source: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    signal_repo: SignalRepository = Depends(get_signal_repo),
):
    """List recent signals."""
    signals = await signal_repo.list_recent(limit=limit, source=source)
    return {"items": [s.model_dump() for s in signals], "total": len(signals)}


# ── Ingest ───────────────────────────────────────────────────


@router.post("/ingest")
async def ingest(
    signal_repo: SignalRepository = Depends(get_signal_repo),
):
    """Trigger a fresh ingestion cycle from all sources."""
    signals = await ingest_all()
    count = await signal_repo.create_batch(signals)
    return {"ingested": count, "sources": list(set(s.source for s in signals))}


# ── Dashboard ────────────────────────────────────────────────


@router.get("/dashboard")
async def dashboard_data(
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
):
    """Get dashboard data (redirects to static HTML if available)."""
    resolved = await pred_repo.get_resolved()
    pending = await pred_repo.list_all(status=Status.PENDING, limit=100)
    tracker = CalibrationTracker()
    report = tracker.compute(resolved)
    return {
        "calibration": report.model_dump(),
        "recent_predictions": [p.model_dump() for p in pending[:20]],
    }
