"""Natural-language-inference (entailment) judge.

This is the reliability keystone. It replaces keyword-overlap "support
guessing" with a real entailment decision: given a ``claim`` and a span of
``evidence_text``, decide whether the evidence — *on its own* — ENTAILS the
claim, CONTRADICTS it, or is NEUTRAL. Crucially, the judge must QUOTE the exact
verbatim span of the evidence it relied on, and that quote is then validated
**programmatically**: if the quoted span does not appear (near-verbatim) in the
evidence text, the verdict is downgraded to NEUTRAL.

That quote-validation step is the trust boundary. A model cannot manufacture
support (or contradiction) it cannot quote, which kills hallucinated citations:
keyword overlap fundamentally cannot detect contradiction, and it cannot prove
that a source actually says what we claim it says. This can.

The judge is LLM-backed but provider-injected, so it is fully mockable and runs
with zero network access in tests (inject a ``MockProvider``).
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from oracle.llm import LLMProvider

logger = logging.getLogger(__name__)

# Minimum length (normalized chars) a quote must have to be considered a real
# span rather than a stray word the model could match by accident.
_MIN_QUOTE_CHARS = 6

# Fraction of the quote that must match a contiguous block of the evidence for a
# "near-verbatim" acceptance (handles trivial punctuation/whitespace drift).
_NEAR_VERBATIM_THRESHOLD = 0.85


class Entailment(str, Enum):
    """The three NLI verdicts."""

    ENTAILS = "ENTAILS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


@dataclass
class EntailmentJudgment:
    """A single entailment decision with its supporting quote.

    ``supports``/``contradicts`` are only ever True when the quote was
    *verified* against the evidence text — an unverified non-NEUTRAL verdict is
    always downgraded to NEUTRAL before this object is returned.
    """

    label: Entailment
    quote: str = ""
    reason: str = ""
    quote_verified: bool = False

    @property
    def supports(self) -> bool:
        return self.label is Entailment.ENTAILS and self.quote_verified

    @property
    def contradicts(self) -> bool:
        return self.label is Entailment.CONTRADICTS and self.quote_verified

    @property
    def stance(self) -> str:
        if self.supports:
            return "supports"
        if self.contradicts:
            return "contradicts"
        return "neutral"


@runtime_checkable
class EntailmentJudge(Protocol):
    """Interface for an entailment judge (mockable injection point)."""

    async def judge(
        self, claim: str, evidence_text: str, *, source_url: str = ""
    ) -> EntailmentJudgment:
        ...


# ---------------------------------------------------------------------------
# Quote validation — the trust boundary
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _strip_punct(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text or "")


def quote_in_evidence(
    quote: str, evidence_text: str, *, threshold: float = _NEAR_VERBATIM_THRESHOLD
) -> bool:
    """Return True iff ``quote`` appears (near-)verbatim in ``evidence_text``.

    Verification is deliberately strict but tolerant of cosmetic drift:
    1. exact substring match after whitespace normalization, then
    2. exact substring match after also stripping punctuation, then
    3. near-verbatim: the longest contiguous matching block covers at least
       ``threshold`` of the quote.

    Anything that requires reordering, paraphrase, or inference fails — which is
    exactly what we want, because the model must not infer from titles/URLs.
    """

    q = _normalize(quote)
    if len(q) < _MIN_QUOTE_CHARS:
        return False

    e = _normalize(evidence_text)
    if not e:
        return False
    if q in e:
        return True

    qp = _normalize(_strip_punct(quote))
    ep = _normalize(_strip_punct(evidence_text))
    if qp and qp in ep:
        return True

    needle = qp or q
    haystack = ep or e
    if not needle or not haystack:
        return False
    matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
    block = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    return (block.size / len(needle)) >= threshold


# ---------------------------------------------------------------------------
# LLM-backed judge
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a strict natural-language-inference (entailment) judge. "
    "Given a CLAIM and a piece of EVIDENCE text, decide whether the EVIDENCE, "
    "ON ITS OWN, ENTAILS the claim (clearly supports or confirms it), "
    "CONTRADICTS it (clearly refutes it), or is NEUTRAL (insufficient, "
    "unrelated, or only weakly related).\n\n"
    "Hard rules:\n"
    "1. Judge ONLY from the EVIDENCE text. Never use outside knowledge.\n"
    "2. You MUST copy, verbatim, the exact span of the EVIDENCE you relied on "
    "into the \"quote\" field. Do not paraphrase it. Do not quote the CLAIM.\n"
    "3. If you cannot find a verbatim span that justifies ENTAILS or "
    "CONTRADICTS, you MUST answer NEUTRAL with an empty quote.\n"
    "4. Do NOT infer support from a title, headline, or URL alone — only the "
    "evidence body counts.\n\n"
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"label": "ENTAILS|CONTRADICTS|NEUTRAL", '
    '"quote": "<verbatim span from EVIDENCE, or empty>", '
    '"reason": "<one short sentence>"}'
)


def _parse_judge_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _coerce_label(raw: object) -> Entailment:
    s = str(raw or "").strip().upper()
    if "ENTAIL" in s or s in {"SUPPORT", "SUPPORTS", "SUPPORTED", "TRUE", "YES"}:
        return Entailment.ENTAILS
    if "CONTRADICT" in s or s in {"REFUTE", "REFUTES", "REFUTED", "FALSE", "NO"}:
        return Entailment.CONTRADICTS
    return Entailment.NEUTRAL


class LLMEntailmentJudge:
    """Entailment judge backed by an injected :class:`LLMProvider`.

    Deterministic by construction: temperature 0, strict JSON, and a hard
    programmatic quote check that the LLM cannot bypass.
    """

    def __init__(self, llm: LLMProvider, *, max_evidence_chars: int = 4000):
        self._llm = llm
        self._max_evidence_chars = max_evidence_chars

    async def judge(
        self, claim: str, evidence_text: str, *, source_url: str = ""
    ) -> EntailmentJudgment:
        claim = (claim or "").strip()
        evidence_text = (evidence_text or "").strip()
        if not claim or not evidence_text:
            return EntailmentJudgment(Entailment.NEUTRAL, reason="empty claim or evidence")

        user_prompt = (
            f"CLAIM:\n{claim}\n\n"
            f"EVIDENCE:\n{evidence_text[: self._max_evidence_chars]}"
        )
        try:
            resp = await self._llm.complete(
                _SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=400
            )
        except Exception as exc:  # network/provider failure → safe NEUTRAL
            logger.debug("NLI judge LLM error: %s", exc)
            return EntailmentJudgment(Entailment.NEUTRAL, reason=f"judge error: {exc}")

        data = _parse_judge_json(getattr(resp, "content", ""))
        if not data:
            return EntailmentJudgment(Entailment.NEUTRAL, reason="unparseable judge output")

        label = _coerce_label(data.get("label"))
        quote = str(data.get("quote") or "").strip()
        reason = str(data.get("reason") or "").strip()

        verified = bool(quote) and quote_in_evidence(quote, evidence_text)

        # Trust boundary: a non-NEUTRAL verdict without a verified quote is a
        # potential hallucination — downgrade it to NEUTRAL.
        if label in (Entailment.ENTAILS, Entailment.CONTRADICTS) and not verified:
            logger.debug(
                "Downgrading %s -> NEUTRAL (quote not found in evidence): %r",
                label.value,
                quote[:80],
            )
            return EntailmentJudgment(
                Entailment.NEUTRAL,
                quote=quote,
                reason=reason or "quote not found in evidence",
                quote_verified=False,
            )

        return EntailmentJudgment(
            label=label, quote=quote, reason=reason, quote_verified=verified
        )


__all__ = [
    "Entailment",
    "EntailmentJudgment",
    "EntailmentJudge",
    "LLMEntailmentJudge",
    "quote_in_evidence",
]
