"""Tests for the canonical VerificationEngine with NLI-judged evidence.

All network is stubbed: web_search / research_topic are monkeypatched to async
fakes, and the entailment judge is a MockProvider returning canned JSON.
"""

import json

import pytest

from oracle.llm import MockProvider
from oracle.models.prediction import Category, Prediction
from oracle.tools import SearchResult, WebContext
from oracle.prediction import verifier as verifier_mod
from oracle.prediction.verifier import VerificationEngine


def _judge_resp(label, quote):
    return json.dumps({"label": label, "quote": quote, "reason": "r"})


def _pred(statement="OpenAI will release a new model in 2025.", conf=0.7):
    return Prediction(category=Category.PRODUCT_LAUNCH, statement=statement, confidence=conf)


@pytest.fixture
def stub_search(monkeypatch):
    """Stub all web access in the verifier module."""

    async def fake_web_search(query, max_results=5):
        return [
            SearchResult(
                title="OpenAI ships new model",
                url="https://reuters.com/openai-new-model",
                snippet="OpenAI released a new model in 2025 to broad availability.",
            )
        ]

    async def fake_research(topic, fetch_depth=2):
        return WebContext(
            query=topic,
            results=[
                SearchResult(
                    title="OpenAI ships new model",
                    url="https://reuters.com/openai-new-model",
                    snippet="OpenAI released a new model in 2025 to broad availability.",
                )
            ],
        )

    monkeypatch.setattr(verifier_mod, "web_search", fake_web_search)
    monkeypatch.setattr(verifier_mod, "research_topic", fake_research)


@pytest.mark.asyncio
async def test_supported_when_judge_entails_with_quote(stub_search):
    mock = MockProvider()
    mock.set_response(_judge_resp("ENTAILS", "OpenAI released a new model in 2025"))
    engine = VerificationEngine(mock, judge=verifier_mod.LLMEntailmentJudge(mock))

    [result] = await engine.verify([_pred()])

    assert result.verdict == "supported"
    assert result.adjusted_confidence >= 0.7
    # Every evidence item carries a verified quote, not a summary.
    assert any(e.quote for e in result.evidence)
    assert all(e.stance == "supports" for e in result.evidence if e.stance != "neutral")


@pytest.mark.asyncio
async def test_contradicted_when_judge_contradicts_with_quote(stub_search):
    mock = MockProvider()
    mock.set_response(_judge_resp("CONTRADICTS", "OpenAI released a new model in 2025"))
    engine = VerificationEngine(mock)

    [result] = await engine.verify([_pred()])

    assert result.verdict == "contradicted"
    assert result.adjusted_confidence < 0.7


@pytest.mark.asyncio
async def test_unverifiable_when_quote_cannot_be_verified(stub_search):
    """Fabricated quote → judge downgrades to neutral → no decisive evidence."""
    mock = MockProvider()
    mock.set_response(_judge_resp("ENTAILS", "this text is nowhere in the evidence"))
    engine = VerificationEngine(mock)

    [result] = await engine.verify([_pred()])

    assert result.verdict == "unverifiable"
    # Confidence is unchanged when nothing is decisive.
    assert result.adjusted_confidence == pytest.approx(0.7)
    assert all(not e.quote for e in result.evidence)


@pytest.mark.asyncio
async def test_dedupes_sources_by_url(stub_search):
    """Web + news return the same URL; it must be judged once."""
    mock = MockProvider()
    mock.set_response(_judge_resp("NEUTRAL", ""))
    engine = VerificationEngine(mock)

    [result] = await engine.verify([_pred()])

    urls = [e.url for e in result.evidence]
    assert len(urls) == len(set(urls))


# --- B6: independent-corroboration + credibility weighting -------------------

_QUOTE = "OpenAI released a new model in 2025"
_SNIPPET = "OpenAI released a new model in 2025 to broad availability."


def _stub_sources(monkeypatch, sources):
    """Make every web-backed checker return the given SearchResults."""

    async def fake_web_search(query, max_results=5):
        return list(sources)

    async def fake_research(topic, fetch_depth=2):
        return WebContext(query=topic, results=list(sources))

    monkeypatch.setattr(verifier_mod, "web_search", fake_web_search)
    monkeypatch.setattr(verifier_mod, "research_topic", fake_research)


@pytest.mark.asyncio
async def test_single_low_credibility_source_does_not_boost(monkeypatch):
    """One lone unknown-domain source must NOT move confidence (B6)."""
    _stub_sources(
        monkeypatch,
        [SearchResult(title="Blog post", url="https://randomblog.example/x", snippet=_SNIPPET)],
    )
    mock = MockProvider()
    mock.set_response(_judge_resp("ENTAILS", _QUOTE))
    engine = VerificationEngine(mock)

    [result] = await engine.verify([_pred()])

    assert result.verdict == "insufficient_corroboration"
    assert result.adjusted_confidence == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_two_independent_domains_boost(monkeypatch):
    """Two distinct mid-credibility domains agreeing earns a boost (B6)."""
    _stub_sources(
        monkeypatch,
        [
            SearchResult(title="TC", url="https://techcrunch.com/a", snippet=_SNIPPET),
            SearchResult(title="Verge", url="https://theverge.com/b", snippet=_SNIPPET),
        ],
    )
    mock = MockProvider()
    mock.set_response(_judge_resp("ENTAILS", _QUOTE))
    engine = VerificationEngine(mock)

    [result] = await engine.verify([_pred()])

    assert result.verdict == "supported"
    assert result.adjusted_confidence > 0.7
    assert len({e.url for e in result.evidence}) == 2


@pytest.mark.asyncio
async def test_two_same_domain_echoes_do_not_boost(monkeypatch):
    """Two articles from the SAME domain are not independent corroboration (B6)."""
    _stub_sources(
        monkeypatch,
        [
            SearchResult(title="A", url="https://randomblog.example/a", snippet=_SNIPPET),
            SearchResult(title="B", url="https://randomblog.example/b", snippet=_SNIPPET),
        ],
    )
    mock = MockProvider()
    mock.set_response(_judge_resp("ENTAILS", _QUOTE))
    engine = VerificationEngine(mock)

    [result] = await engine.verify([_pred()])

    assert result.verdict == "insufficient_corroboration"
    assert result.adjusted_confidence == pytest.approx(0.7)

