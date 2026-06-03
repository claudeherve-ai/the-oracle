"""Tests for the NLI entailment judge — the reliability trust boundary."""

import json

import pytest

from oracle.llm import MockProvider
from oracle.tools.nli import (
    Entailment,
    EntailmentJudgment,
    LLMEntailmentJudge,
    quote_in_evidence,
)


# ── quote_in_evidence (pure, the trust boundary) ───────────────────


def test_quote_exact_substring_matches():
    evidence = "The Federal Reserve raised rates by 25 basis points on Wednesday."
    assert quote_in_evidence("raised rates by 25 basis points", evidence)


def test_quote_whitespace_and_case_insensitive():
    evidence = "Apple   announced   record   revenue."
    assert quote_in_evidence("Apple announced record revenue", evidence)


def test_quote_punctuation_tolerant():
    evidence = "Sales grew 12%, beating estimates."
    assert quote_in_evidence("Sales grew 12 beating estimates", evidence)


def test_quote_near_verbatim_accepted():
    evidence = "The company reported strong quarterly earnings growth this year."
    assert quote_in_evidence("reported strong quarterly earnings growth", evidence)


def test_fabricated_quote_rejected():
    evidence = "The weather is sunny today in Seattle."
    assert not quote_in_evidence("the stock price doubled overnight", evidence)


def test_too_short_quote_rejected():
    evidence = "Yes it is true."
    assert not quote_in_evidence("Yes", evidence)


def test_empty_inputs_rejected():
    assert not quote_in_evidence("", "something")
    assert not quote_in_evidence("something", "")


# ── LLMEntailmentJudge ─────────────────────────────────────────────


def _resp(label, quote, reason="r"):
    return json.dumps({"label": label, "quote": quote, "reason": reason})


@pytest.mark.asyncio
async def test_entails_with_valid_quote_supports():
    evidence = "OpenAI released GPT-5 to the public in 2025."
    mock = MockProvider()
    mock.set_response(_resp("ENTAILS", "OpenAI released GPT-5 to the public"))
    judge = LLMEntailmentJudge(mock)

    j = await judge.judge("OpenAI released GPT-5", evidence)
    assert j.label is Entailment.ENTAILS
    assert j.quote_verified
    assert j.supports
    assert not j.contradicts
    assert j.stance == "supports"


@pytest.mark.asyncio
async def test_contradicts_with_valid_quote():
    evidence = "The merger was officially cancelled by both companies."
    mock = MockProvider()
    mock.set_response(_resp("CONTRADICTS", "The merger was officially cancelled"))
    judge = LLMEntailmentJudge(mock)

    j = await judge.judge("The merger will be completed", evidence)
    assert j.label is Entailment.CONTRADICTS
    assert j.contradicts
    assert j.stance == "contradicts"


@pytest.mark.asyncio
async def test_fabricated_quote_downgraded_to_neutral():
    """The keystone: ENTAILS without a verifiable quote must NOT count."""
    evidence = "The report covered general market trends in Europe."
    mock = MockProvider()
    # Model claims entailment but quotes text that is NOT in the evidence.
    mock.set_response(_resp("ENTAILS", "profits tripled in a single quarter"))
    judge = LLMEntailmentJudge(mock)

    j = await judge.judge("Profits tripled", evidence)
    assert j.label is Entailment.NEUTRAL
    assert not j.quote_verified
    assert not j.supports
    assert j.stance == "neutral"


@pytest.mark.asyncio
async def test_neutral_label_passes_through():
    evidence = "The article discussed unrelated topics."
    mock = MockProvider()
    mock.set_response(_resp("NEUTRAL", ""))
    judge = LLMEntailmentJudge(mock)

    j = await judge.judge("Something specific happened", evidence)
    assert j.label is Entailment.NEUTRAL
    assert j.stance == "neutral"


@pytest.mark.asyncio
async def test_empty_evidence_is_neutral_without_calling_llm():
    mock = MockProvider()
    mock.set_response(_resp("ENTAILS", "anything"))
    judge = LLMEntailmentJudge(mock)

    j = await judge.judge("claim", "   ")
    assert j.label is Entailment.NEUTRAL
    assert len(mock.calls) == 0  # short-circuited, no provider call


@pytest.mark.asyncio
async def test_unparseable_output_is_neutral():
    mock = MockProvider()
    mock.set_response("not json at all")
    judge = LLMEntailmentJudge(mock)

    j = await judge.judge("claim", "some real evidence text here")
    assert j.label is Entailment.NEUTRAL


@pytest.mark.asyncio
async def test_judge_uses_temperature_zero():
    evidence = "Confirmed: the launch happened on schedule."
    mock = MockProvider()
    mock.set_response(_resp("ENTAILS", "the launch happened on schedule"))
    judge = LLMEntailmentJudge(mock)

    await judge.judge("The launch happened", evidence)
    assert mock.calls[0]["temperature"] == 0.0


def test_judgment_dataclass_properties():
    j = EntailmentJudgment(Entailment.ENTAILS, quote="x", quote_verified=True)
    assert j.supports and not j.contradicts
    j2 = EntailmentJudgment(Entailment.ENTAILS, quote="x", quote_verified=False)
    assert not j2.supports  # unverified never supports
