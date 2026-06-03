"""Tests for the ensemble engine (C9).

The ensemble is only trustworthy if prompt variants genuinely differentiate the
runs — otherwise every run is identical and ``disagreement_score`` is always 0,
which would be a silent lie about uncertainty. These tests pin that behavior.
"""

import json

import pytest

from oracle.llm import MockProvider
from oracle.models.prediction import Category, Signal
from oracle.prediction.ensemble import EnsembleEngine, PROMPT_VARIANTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(content: str = "Test signal", source: str = "hackernews") -> Signal:
    return Signal(
        source=source,
        content=content,
        entities=[],
        sentiment=0.0,
        relevance=0.7,
        metadata={
            "anomaly_score": 0.0,
            "patterns_detected": [],
            "keywords": [],
            "category_hints": [],
            "all_patterns": [],
        },
    )


def make_single_prediction_response(confidence: float, statement: str) -> str:
    """One prediction with a fixed statement and a given confidence.

    A fixed statement guarantees all variant runs aggregate into one group so
    the spread of confidences drives ``disagreement_score``.
    """
    return json.dumps({
        "predictions": [
            {
                "statement": statement,
                "category": "product_launch",
                "confidence": confidence,
                "reasoning": "Multiple independent signals point to this outcome "
                             "within the stated horizon; treated per the stance.",
                "deadline": "2026-10-01",
                "sources": [
                    "https://news.ycombinator.com/item?id=1",
                    "https://techcrunch.com/article-1",
                ],
            }
        ]
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensemble_empty_returns_empty():
    provider = MockProvider()
    ensemble = EnsembleEngine(provider)
    result = await ensemble.generate([], question=None)
    assert result.predictions == []
    assert result.models_used == 0


@pytest.mark.asyncio
async def test_variant_guidance_is_actually_injected():
    """Each variant's stance must reach the prompt (the latent-bug regression).

    If the variant text never makes it into the prompt, the ensemble is
    theater. We assert every configured variant stance shows up in the
    captured prompts.
    """
    provider = MockProvider()
    stmt = "Company X will ship a major product by Q4 2026"
    provider.set_response(make_single_prediction_response(0.5, stmt))

    ensemble = EnsembleEngine(provider)
    await ensemble.generate([make_signal("Company X is hiring")], question="Will X ship?")

    prompts = "\n".join(c["user_prompt"] for c in provider.calls)
    assert "FORECASTER STANCE" in prompts
    for variant in PROMPT_VARIANTS:
        # The variant label appears in the injected stance header.
        assert variant in prompts


@pytest.mark.asyncio
async def test_disagreement_is_nonzero_when_variants_differ():
    """Differing confidences across variants must yield disagreement > 0."""
    provider = MockProvider()
    stmt = "Company X will ship a major product by Q4 2026"
    # One response per variant run, with a wide confidence spread.
    provider.set_response(
        make_single_prediction_response(0.30, stmt),
        make_single_prediction_response(0.55, stmt),
        make_single_prediction_response(0.90, stmt),
    )

    ensemble = EnsembleEngine(provider)  # 3 default variants → 3 runs
    result = await ensemble.generate(
        [make_signal("Company X is hiring")],
        question="Will X ship?",
    )

    assert result.disagreement_score > 0.0
    assert result.models_used == 1
    assert len(result.variants_used) == len(PROMPT_VARIANTS)
    # Same statement across runs → aggregated into a single prediction.
    assert len(result.predictions) == 1
    agg = result.predictions[0]
    # Mean of the spread, roughly centered.
    assert 0.4 < agg.confidence < 0.8


@pytest.mark.asyncio
async def test_agreement_yields_low_disagreement():
    """Identical confidences across variants → disagreement near zero."""
    provider = MockProvider()
    stmt = "Company X will ship a major product by Q4 2026"
    provider.set_response(make_single_prediction_response(0.60, stmt))  # repeats

    ensemble = EnsembleEngine(provider)
    result = await ensemble.generate(
        [make_signal("Company X is hiring")],
        question="Will X ship?",
    )

    assert result.disagreement_score == 0.0
    assert len(result.predictions) == 1


@pytest.mark.asyncio
async def test_explicit_single_variant():
    """Requesting one variant runs exactly once."""
    provider = MockProvider()
    stmt = "Company X will ship a major product by Q4 2026"
    provider.set_response(make_single_prediction_response(0.7, stmt))

    ensemble = EnsembleEngine(provider, prompt_variants=["conservative"])
    result = await ensemble.generate(
        [make_signal("Company X is hiring")],
        question="Will X ship?",
    )

    assert result.variants_used == ["conservative"]
    assert len(provider.calls) == 1
    assert result.disagreement_score == 0.0
