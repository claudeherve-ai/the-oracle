"""Tests for the per-prediction audit trail (D14).

Reproducibility is trust: when an :class:`~oracle.audit.AuditSink` is wired into
the engine, every ``generate()`` call must emit a structured, replayable record
of the prompts, model, sources, evidence spans, and confidence *before and
after* verification — and emit **nothing** (zero overhead) when no sink is set.
"""

import json
import logging

import pytest

from oracle.audit import (
    AuditRecord,
    AuditSink,
    EvidenceSpan,
    InMemoryAuditSink,
    LoggingAuditSink,
    PredictionAudit,
    new_audit_logger,
)
from oracle.llm import MockProvider
from oracle.prediction.engine import PredictionEngine

from tests.test_prediction_engine import (
    make_signal,
    make_valid_prediction_response,
)


# ---------------------------------------------------------------------------
# Stub verifier that returns quoted evidence spans
# ---------------------------------------------------------------------------

class _EvidenceVerifier:
    """Returns deterministic results carrying quoted :class:`SourceEvidence`."""

    def __init__(self, adjusted: float = 0.42, verdict: str = "corroborated"):
        self.called = False
        self._adjusted = adjusted
        self._verdict = verdict

    async def verify(self, predictions, *, deep_check: bool = False):
        from oracle.prediction.verifier import SourceEvidence, VerificationResult

        self.called = True
        results = []
        for p in predictions:
            results.append(
                VerificationResult(
                    prediction_id=p.id,
                    statement=p.statement,
                    original_confidence=p.confidence,
                    adjusted_confidence=self._adjusted,
                    summary="2 independent sources support",
                    verdict=self._verdict,
                    evidence=[
                        SourceEvidence(
                            url="https://example.com/a",
                            title="Source A",
                            snippet="...",
                            supports=True,
                            stance="supports",
                            quote="The company confirmed the launch for Q4.",
                            relevance=0.9,
                            credibility=0.8,
                        ),
                        SourceEvidence(
                            url="https://example.com/b",
                            title="Source B",
                            snippet="...",
                            supports=False,
                            stance="contradicts",
                            quote="Analysts doubt the timeline will hold.",
                            relevance=0.7,
                            credibility=0.6,
                        ),
                    ],
                )
            )
        return results


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_sink_emits_no_record_and_unchanged_behaviour():
    """Without a sink, generation is byte-for-byte unchanged and silent."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(2))

    engine = PredictionEngine(provider)  # no audit_sink
    result = await engine.generate([make_signal("x")], question="Will X ship?")

    assert len(result) == 2
    # Nothing to assert on a sink because there is none — the point is no crash
    # and no behavioural change. Confidences come straight from the model.
    assert all(0.0 < p.confidence < 1.0 for p in result)


@pytest.mark.asyncio
async def test_sink_captures_record_off_mode():
    """A record is captured even with verification off; pre == post confidence."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(3))
    sink = InMemoryAuditSink()

    engine = PredictionEngine(provider, audit_sink=sink)
    result = await engine.generate([make_signal("x")], question="Will X ship?")

    assert len(sink) == 1
    rec = sink.last
    assert isinstance(rec, AuditRecord)
    assert rec.question == "Will X ship?"
    assert rec.verification_mode == "off"
    assert rec.model == provider.model_name
    assert rec.raw_response  # captured the raw LLM output
    assert rec.system_prompt and rec.user_prompt
    assert len(rec.predictions) == len(result) == 3

    for pa in rec.predictions:
        assert isinstance(pa, PredictionAudit)
        # Verification was off → confidence not moved, no spans, no verdict.
        assert pa.confidence_pre == pa.confidence_post
        assert pa.confidence_delta == 0.0
        assert pa.evidence_spans == []
        assert pa.verdict is None
        assert pa.statement


@pytest.mark.asyncio
async def test_sink_captures_pre_and_post_confidence_live():
    """Live verification records the confidence move and the quoted spans."""
    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(2))
    sink = InMemoryAuditSink()
    verifier = _EvidenceVerifier(adjusted=0.42, verdict="corroborated")

    engine = PredictionEngine(
        provider,
        verifier=verifier,
        verification_mode="live",
        audit_sink=sink,
    )
    result = await engine.generate([make_signal("x")], question="Will X ship?")

    assert verifier.called is True
    assert len(sink) == 1
    rec = sink.last
    assert rec.verification_mode == "live"
    assert len(rec.predictions) == 2

    for pa, p in zip(rec.predictions, result):
        # Post confidence equals the (clamped) verifier output on the prediction.
        assert pa.confidence_post == pytest.approx(p.confidence)
        assert pa.confidence_post == pytest.approx(0.42)
        # Pre confidence is whatever the model proposed, before the verifier.
        assert pa.confidence_pre != pa.confidence_post
        assert pa.confidence_delta == round(pa.confidence_post - pa.confidence_pre, 4)
        assert pa.verdict == "corroborated"
        # Quoted spans captured verbatim with stance + credibility.
        assert len(pa.evidence_spans) == 2
        stances = {s.stance for s in pa.evidence_spans}
        assert stances == {"supports", "contradicts"}
        assert all(s.quote for s in pa.evidence_spans)
        assert all(s.url.startswith("https://") for s in pa.evidence_spans)


@pytest.mark.asyncio
async def test_audit_failure_is_non_fatal():
    """A misbehaving sink must never break generation."""

    class _BoomSink:
        def record(self, record):  # noqa: ANN001
            raise RuntimeError("sink exploded")

    provider = MockProvider()
    provider.set_response(make_valid_prediction_response(1))

    engine = PredictionEngine(provider, audit_sink=_BoomSink())
    # Should not raise despite the sink blowing up.
    result = await engine.generate([make_signal("x")], question="Will X ship?")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Record / sink units
# ---------------------------------------------------------------------------

def test_evidence_span_to_dict_rounds_credibility():
    span = EvidenceSpan(url="u", quote="q", stance="supports", credibility=0.123456)
    d = span.to_dict()
    assert d == {"url": "u", "quote": "q", "stance": "supports", "credibility": 0.1235}


def test_prediction_audit_confidence_delta():
    pa = PredictionAudit(
        prediction_id="p1",
        statement="s",
        category="product_launch",
        status="active",
        confidence_pre=0.70,
        confidence_post=0.42,
    )
    assert pa.confidence_delta == -0.28
    d = pa.to_dict()
    assert d["confidence_delta"] == -0.28
    assert d["evidence_spans"] == []


def test_audit_record_to_json_roundtrips():
    rec = AuditRecord(
        model="mock",
        question="Will X ship?",
        system_prompt="sys",
        user_prompt="usr",
        raw_response='{"predictions": []}',
        verification_mode="live",
        grounding_present=True,
        grounding_chars=128,
        predictions=[
            PredictionAudit(
                prediction_id="p1",
                statement="s",
                category="product_launch",
                status="active",
                confidence_pre=0.5,
                confidence_post=0.6,
                verdict="corroborated",
                sources=["https://example.com"],
                evidence_spans=[EvidenceSpan(url="https://example.com", quote="q")],
            )
        ],
    )
    line = rec.to_json()
    assert "\n" not in line  # single JSON line for log pipelines
    parsed = json.loads(line)
    assert parsed["model"] == "mock"
    assert parsed["prediction_count"] == 1
    assert parsed["predictions"][0]["evidence_spans"][0]["quote"] == "q"
    assert parsed["grounding_chars"] == 128


def test_in_memory_sink_last_and_len():
    sink = InMemoryAuditSink()
    assert sink.last is None
    assert len(sink) == 0
    rec = AuditRecord(model="m")
    sink.record(rec)
    assert len(sink) == 1
    assert sink.last is rec


def test_logging_sink_emits_single_json_line(caplog):
    sink = LoggingAuditSink()
    rec = AuditRecord(model="m", question="Q?")
    with caplog.at_level(logging.INFO, logger="oracle.audit"):
        sink.record(rec)
    # Exactly one audit log line, and its payload is valid JSON.
    audit_lines = [r for r in caplog.records if r.name == "oracle.audit"]
    assert len(audit_lines) == 1
    payload = audit_lines[0].getMessage().split("audit ", 1)[1]
    parsed = json.loads(payload)
    assert parsed["model"] == "m"
    assert parsed["question"] == "Q?"


def test_audit_sink_protocol_runtime_checkable():
    assert isinstance(InMemoryAuditSink(), AuditSink)
    assert isinstance(LoggingAuditSink(), AuditSink)

    class _NotASink:
        pass

    assert not isinstance(_NotASink(), AuditSink)


def test_new_audit_logger_name():
    assert new_audit_logger().name == "oracle.audit"
