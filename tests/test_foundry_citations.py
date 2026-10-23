"""Tests for the Foundry LLM provider, citations, and fail-loud configuration."""

import pytest

from oracle.llm import OpenAIProvider, normalize_endpoint
from oracle.llm import MockProvider
from oracle.prediction.engine import PredictionEngine
from oracle.models.prediction import Status


@pytest.mark.parametrize(
    "raw",
    [
        "https://r.services.ai.azure.com",
        "https://r.services.ai.azure.com/",
        "https://r.services.ai.azure.com/openai/v1",
        "https://r.services.ai.azure.com/openai/v1/responses",
        "https://r.services.ai.azure.com/openai",
    ],
)
def test_normalize_endpoint(raw):
    assert normalize_endpoint(raw) == "https://r.services.ai.azure.com/openai/v1"


def test_default_deployment(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    assert OpenAIProvider().model_name == "gpt-5.6-terra"


async def test_missing_endpoint_fails_loudly(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        await OpenAIProvider().complete("s", "u")


def test_get_llm_never_falls_back_to_mock(monkeypatch):
    from oracle.api.dependencies import get_llm

    get_llm.cache_clear()
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("ORACLE_LLM", raising=False)
    with pytest.raises(RuntimeError):
        get_llm()
    get_llm.cache_clear()


_RESPONSE = (
    '[{"statement": "Company X will announce product Y by June 30 2026.", '
    '"category": "product_launch", "confidence": 0.7, "reasoning": "r", '
    '"sources": %s, "deadline": "2026-06-30"}]'
)


async def test_citations_attached_from_sources():
    llm = MockProvider()
    llm.set_response(_RESPONSE % '["https://example.com/a", "https://example.com/a"]')
    engine = PredictionEngine(llm)
    preds = await engine.generate([], question="Will X launch Y?")
    assert [c.url for c in preds[0].citations] == ["https://example.com/a"]
    assert preds[0].citations[0].stance == "unverified"


async def test_uncited_prediction_abstains_when_enabled():
    llm = MockProvider()
    llm.set_response(_RESPONSE % "[]")
    engine = PredictionEngine(llm, abstain_on_weak_evidence=True)
    preds = await engine.generate([], question="Will X launch Y?")
    assert preds[0].status == Status.INSUFFICIENT_EVIDENCE
