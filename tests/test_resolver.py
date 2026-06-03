"""Tests for the resolution keystone (C8/B7).

These verify the *trust-critical* behavior of automated resolution: that labels
are derived from real fetched text + verified quotes, that verdicts map to the
correct calibration-safe status, that conflicting evidence abstains and flags for
review, and that every source is snapshotted for later audit.

All network is stubbed — an injected judge plus async search/fetch stubs — so the
autouse network ban in conftest is satisfied.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from oracle.llm import MockProvider
from oracle.models.prediction import Category, Prediction, Status
from oracle.resolution.resolver import (
    REASON_ALL_FETCHES_FAILED,
    REASON_ALL_NEUTRAL,
    REASON_CONFLICTING,
    REASON_NO_SOURCES,
    REASON_RESOLVED,
    ResolutionEngine,
    _build_claim,
    _canonicalize_url,
)
from oracle.tools.nli import Entailment, EntailmentJudgment


# --------------------------------------------------------------------------- #
# Stubs                                                                        #
# --------------------------------------------------------------------------- #


class _SearchResult:
    def __init__(self, url: str):
        self.url = url
        self.title = url
        self.snippet = ""


def make_search_fn(url_map):
    """Return an async search stub.

    ``url_map`` maps a substring of the query to a list of URLs to return.
    """

    async def _search(query, max_results=5):
        for needle, urls in url_map.items():
            if needle in query:
                return [_SearchResult(u) for u in urls][:max_results]
        return []

    return _search


def make_fetch_fn(text_map, *, errors=None):
    """Return an async fetch stub mapping url -> page text."""
    errors = errors or {}

    async def _fetch(url, max_chars=4000):
        if url in errors:
            raise RuntimeError(errors[url])
        return text_map.get(url, "")[:max_chars]

    return _fetch


class StubJudge:
    """Injectable entailment judge returning canned verdicts keyed by text.

    ``verdicts`` maps a substring of the evidence text -> EntailmentJudgment.
    Anything unmatched is NEUTRAL.
    """

    def __init__(self, verdicts):
        self._verdicts = verdicts
        self.calls = []

    async def judge(self, claim, evidence_text, *, source_url=""):
        self.calls.append((claim, evidence_text, source_url))
        for needle, judgment in self._verdicts.items():
            if needle in evidence_text:
                return judgment
        return EntailmentJudgment(label=Entailment.NEUTRAL)


def _entails(quote):
    return EntailmentJudgment(
        label=Entailment.ENTAILS, quote=quote, quote_verified=True, reason="entails"
    )


def _contradicts(quote):
    return EntailmentJudgment(
        label=Entailment.CONTRADICTS, quote=quote, quote_verified=True, reason="contra"
    )


def _make_prediction(statement="Acme Corp will ship product X to general availability") -> Prediction:
    return Prediction(
        id="pred-1",
        category=Category.PRODUCT_LAUNCH,
        statement=statement,
        confidence=0.7,
        deadline=datetime.now(timezone.utc) - timedelta(days=1),
        status=Status.PENDING,
    )


def _engine(judge, search_fn, fetch_fn, **kw):
    return ResolutionEngine(
        MockProvider(),
        judge=judge,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_canonicalize_strips_tracking_and_www():
    a = _canonicalize_url("https://www.Example.com/path/?utm_source=x&id=7#frag")
    b = _canonicalize_url("https://example.com/path?id=7")
    assert a == b
    assert "utm_source" not in a
    assert "#frag" not in a
    assert "www." not in a


def test_canonicalize_protocol_relative():
    assert _canonicalize_url("//example.com/x").startswith("https://example.com")


def test_build_claim_bakes_deadline_and_happened_framing():
    when = datetime(2030, 1, 2, tzinfo=timezone.utc)
    claim = _build_claim("Acme ships X", when)
    assert "2030-01-02" in claim
    assert "ACTUALLY HAPPENED" in claim
    assert "Acme ships X" in claim


def test_build_claim_handles_no_deadline():
    claim = _build_claim("Acme ships X", None)
    assert "unspecified date" in claim


# --------------------------------------------------------------------------- #
# Status mapping                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_entails_resolves_correct():
    pred = _make_prediction()
    search = make_search_fn({"outcome result": ["https://reuters.com/a"]})
    fetch = make_fetch_fn({"https://reuters.com/a": "Acme shipped product X today. It is now generally available."})
    judge = StubJudge({"Acme shipped product X": _entails("Acme shipped product X today")})

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.resolved_as is Status.CORRECT
    assert o.resolution_reason == REASON_RESOLVED
    assert o.requires_human_review is False
    snap = o.evidence_snapshots[0]
    assert snap.stance == "supports"
    assert snap.quote == "Acme shipped product X today"
    assert snap.content_hash  # sha256 populated
    assert snap.snippet
    assert snap.canonical_url.startswith("https://reuters.com")


@pytest.mark.asyncio
async def test_contradicts_resolves_incorrect():
    pred = _make_prediction()
    search = make_search_fn({"outcome result": ["https://reuters.com/a"]})
    fetch = make_fetch_fn({"https://reuters.com/a": "Acme cancelled product X. It will not ship."})
    judge = StubJudge({"Acme cancelled": _contradicts("Acme cancelled product X")})

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.INCORRECT
    assert o.resolution_reason == REASON_RESOLVED
    assert o.evidence_snapshots[0].stance == "contradicts"


@pytest.mark.asyncio
async def test_neutral_resolves_insufficient():
    pred = _make_prediction()
    search = make_search_fn({"outcome result": ["https://blog.example.com/a"]})
    fetch = make_fetch_fn({"https://blog.example.com/a": "Some unrelated commentary about the weather."})
    judge = StubJudge({})  # everything NEUTRAL

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.INSUFFICIENT_EVIDENCE
    assert o.resolution_reason == REASON_ALL_NEUTRAL
    assert o.confidence == "low"


@pytest.mark.asyncio
async def test_conflict_equal_abstains_and_flags_review():
    pred = _make_prediction()
    search = make_search_fn(
        {"outcome result": ["https://reuters.com/yes", "https://bloomberg.com/no"]}
    )
    fetch = make_fetch_fn(
        {
            "https://reuters.com/yes": "Acme shipped product X today.",
            "https://bloomberg.com/no": "Acme cancelled product X entirely.",
        }
    )
    judge = StubJudge(
        {
            "shipped product X": _entails("Acme shipped product X today"),
            "cancelled product X": _contradicts("Acme cancelled product X entirely"),
        }
    )

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.INSUFFICIENT_EVIDENCE
    assert o.resolution_reason == REASON_CONFLICTING
    assert o.requires_human_review is True


@pytest.mark.asyncio
async def test_majority_wins_but_still_flags_review_on_conflict():
    pred = _make_prediction()
    search = make_search_fn(
        {
            "outcome result": [
                "https://reuters.com/yes",
                "https://apnews.com/yes2",
                "https://bloomberg.com/no",
            ]
        }
    )
    fetch = make_fetch_fn(
        {
            "https://reuters.com/yes": "Acme shipped product X today.",
            "https://apnews.com/yes2": "Acme has shipped product X to all customers.",
            "https://bloomberg.com/no": "Acme delayed product X indefinitely.",
        }
    )
    judge = StubJudge(
        {
            "shipped product X today": _entails("Acme shipped product X today"),
            "shipped product X to all": _entails("Acme has shipped product X to all customers"),
            "delayed product X": _contradicts("Acme delayed product X indefinitely"),
        }
    )

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.CORRECT  # 2 entail > 1 contradict
    assert o.requires_human_review is True  # but conflict is surfaced
    assert o.confidence == "high"  # 2 independent supporting domains


@pytest.mark.asyncio
async def test_independent_domains_required_for_high_confidence():
    """Two URLs on the SAME domain count once — confidence stays medium."""
    pred = _make_prediction()
    search = make_search_fn(
        {"outcome result": ["https://reuters.com/a", "https://reuters.com/b"]}
    )
    fetch = make_fetch_fn(
        {
            "https://reuters.com/a": "Acme shipped product X today (story A).",
            "https://reuters.com/b": "Acme shipped product X today (story B).",
        }
    )
    judge = StubJudge({"Acme shipped product X today": _entails("Acme shipped product X today")})

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.CORRECT
    assert o.confidence == "medium"  # only 1 distinct domain


# --------------------------------------------------------------------------- #
# Evidence integrity                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_sources_found():
    pred = _make_prediction()
    search = make_search_fn({})  # returns nothing
    fetch = make_fetch_fn({})
    judge = StubJudge({})

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.INSUFFICIENT_EVIDENCE
    assert o.resolution_reason == REASON_NO_SOURCES
    assert o.evidence_snapshots == []


@pytest.mark.asyncio
async def test_all_fetches_fail():
    pred = _make_prediction()
    search = make_search_fn({"outcome result": ["https://reuters.com/a"]})
    fetch = make_fetch_fn({}, errors={"https://reuters.com/a": "timeout"})
    judge = StubJudge({})

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.INSUFFICIENT_EVIDENCE
    assert o.resolution_reason == REASON_ALL_FETCHES_FAILED
    snap = o.evidence_snapshots[0]
    assert snap.fetch_ok is False
    assert snap.fetch_error


@pytest.mark.asyncio
async def test_hallucinated_quote_is_rejected():
    """A non-NEUTRAL verdict whose quote was NOT verified must not resolve."""
    pred = _make_prediction()
    search = make_search_fn({"outcome result": ["https://reuters.com/a"]})
    fetch = make_fetch_fn({"https://reuters.com/a": "Totally unrelated page content."})
    # ENTAILS but quote_verified=False -> stance is "neutral" by construction.
    bad = EntailmentJudgment(
        label=Entailment.ENTAILS, quote="quote not in text", quote_verified=False
    )
    judge = StubJudge({"unrelated": bad})

    outcomes = await _engine(judge, search, fetch).resolve([pred])

    o = outcomes[0]
    assert o.resolved_as is Status.INSUFFICIENT_EVIDENCE
    assert o.evidence_snapshots[0].stance == "neutral"
    assert o.evidence_snapshots[0].quote == ""


@pytest.mark.asyncio
async def test_to_result_projects_all_audit_fields():
    pred = _make_prediction()
    search = make_search_fn({"outcome result": ["https://reuters.com/a"]})
    fetch = make_fetch_fn({"https://reuters.com/a": "Acme shipped product X today."})
    judge = StubJudge({"Acme shipped product X": _entails("Acme shipped product X today")})

    outcomes = await _engine(judge, search, fetch).resolve([pred])
    result = outcomes[0].to_result()

    assert result.prediction_id == "pred-1"
    assert result.new_status is Status.CORRECT
    assert result.previous_status == Status.PENDING.value
    assert result.resolution_claim
    assert result.resolution_reason == REASON_RESOLVED
    assert result.evidence_snapshots
    assert result.evidence_urls == ["https://reuters.com/a"]


# --------------------------------------------------------------------------- #
# Engine accounting / filtering                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_only_past_deadline_pending_resolved():
    future = Prediction(
        id="future",
        category=Category.PRODUCT_LAUNCH,
        statement="This has not reached its deadline yet at all",
        confidence=0.6,
        deadline=datetime.now(timezone.utc) + timedelta(days=5),
        status=Status.PENDING,
    )
    no_deadline = Prediction(
        id="nodl",
        category=Category.PRODUCT_LAUNCH,
        statement="This prediction has no deadline set whatsoever",
        confidence=0.6,
        status=Status.PENDING,
    )
    already = Prediction(
        id="done",
        category=Category.PRODUCT_LAUNCH,
        statement="This one was already resolved correctly before now",
        confidence=0.6,
        deadline=datetime.now(timezone.utc) - timedelta(days=2),
        status=Status.CORRECT,
    )
    judge = StubJudge({})
    search = make_search_fn({})
    fetch = make_fetch_fn({})

    outcomes = await _engine(judge, search, fetch).resolve([future, no_deadline, already])
    assert outcomes == []


@pytest.mark.asyncio
async def test_per_prediction_exception_becomes_insufficient_not_expired():
    pred = _make_prediction()

    async def boom_search(query, max_results=5):
        raise RuntimeError("search exploded")

    # Search failures are swallowed (-> no sources). Force a harder failure by
    # making the fetch stub raise *after* a URL is found via a working search.
    search = make_search_fn({"outcome result": ["https://reuters.com/a"]})

    async def boom_fetch(url, max_chars=4000):
        raise RuntimeError("fetch exploded")

    judge = StubJudge({})
    outcomes = await _engine(judge, search, boom_fetch).resolve([pred])

    o = outcomes[0]
    # A fetch error is captured on the snapshot, not raised; all fetches failed.
    assert o.resolved_as is Status.INSUFFICIENT_EVIDENCE
    assert o.resolved_as is not Status.EXPIRED
