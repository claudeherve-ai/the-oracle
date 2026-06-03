"""Resolution Engine — auto-resolve predictions against real-world data.

This is the calibration keystone. Calibration math is only as trustworthy as the
labels it is fed: a single mislabelled outcome silently poisons every reliability
curve derived from it. So resolution here is deliberately conservative and fully
auditable:

1. Gather candidate source URLs (outcome searches, news, ticker lookups).
2. **Fetch the actual page text** of each source — never judge from a URL string.
3. Run a real natural-language-inference judge (:class:`LLMEntailmentJudge`) that
   must QUOTE the verbatim span it relied on; unquotable verdicts are downgraded
   to NEUTRAL, which kills hallucinated resolutions.
4. The claim that is judged is **deadline-baked** — it asserts the event has
   *actually happened* (not merely been planned/expected), so a late event does
   not produce a false ``CORRECT``.
5. Snapshot every source (text hash, verified quote, surrounding context) onto
   the result so the call can be audited months later.

Conflicting verified evidence (some sources entail, some contradict) never gets
silently coerced into a label: it resolves to ``INSUFFICIENT_EVIDENCE`` and is
flagged ``requires_human_review``. Anything we cannot stand behind resolves to
``INSUFFICIENT_EVIDENCE`` (an honest abstention excluded from calibration) — the
engine never emits ``EXPIRED`` to mean "couldn't determine".

The engine is fully injectable (``judge``/``search_fn``/``fetch_fn``) so it runs
with zero network access in tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from oracle.llm import LLMProvider
from oracle.models.prediction import (
    EvidenceSnapshot,
    Prediction,
    ResolutionResult,
    Status,
)
from oracle.tools import web_fetch, web_search
from oracle.tools.nli import EntailmentJudge, LLMEntailmentJudge

logger = logging.getLogger(__name__)

# Module-level aliases so the defaults are patchable and so tests can inject
# stubs without touching the network.
_web_search = web_search
_web_fetch = web_fetch

# Tokens that look like tickers but are common words — never search them.
_TICKER_STOPWORDS = frozenset(
    {
        "THE", "A", "I", "AT", "IN", "BY", "Q", "IS", "AND", "OR", "BE", "TO",
        "IT", "WILL", "FOR", "OF", "ON", "AN", "AS", "IF", "SO", "NO", "US",
        "AI", "ML", "API", "CEO", "CTO", "IPO", "GDP", "USA", "EU", "UK",
    }
)

# Query-string params that are pure tracking noise and must be stripped when
# canonicalizing a URL for de-duplication.
_TRACKING_PARAMS = frozenset(
    {
        "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "yclid", "_hsenc",
        "_hsmi", "mc_cid", "mc_eid", "igshid", "ref", "ref_src", "ref_url",
        "spm", "cmpid", "scid",
    }
)

# Audit reason codes stored on ``ResolutionResult.resolution_reason``. They keep
# the semantic signal the old EXPIRED status used to carry, without polluting the
# calibration label.
REASON_RESOLVED = "resolved"
REASON_NO_SOURCES = "no_sources_found"
REASON_ALL_FETCHES_FAILED = "all_fetches_failed"
REASON_ALL_NEUTRAL = "all_neutral"
REASON_CONFLICTING = "conflicting_verified_evidence"
REASON_EXCEPTION = "resolver_exception"

# How much context to keep on either side of a verified quote in the snapshot.
_SNIPPET_PAD = 400


SearchFn = Callable[..., Awaitable[Sequence]]
FetchFn = Callable[..., Awaitable[str]]


@dataclass
class ResolutionOutcome:
    """Internal resolution result, convertible to a :class:`ResolutionResult`."""

    prediction_id: str
    statement: str
    deadline: Optional[datetime]
    resolved_as: Status
    confidence: str = "low"  # low, medium, high
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    resolution_claim: str = ""
    resolution_reason: str = ""
    requires_human_review: bool = False
    evidence_snapshots: List[EvidenceSnapshot] = field(default_factory=list)
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_result(self) -> ResolutionResult:
        """Project this outcome onto the public, persisted ResolutionResult."""
        return ResolutionResult(
            prediction_id=self.prediction_id,
            statement=self.statement,
            previous_status=Status.PENDING.value,
            new_status=self.resolved_as,
            resolution=self.reasoning,
            resolution_claim=self.resolution_claim,
            resolution_reason=self.resolution_reason,
            confidence=self.confidence,
            reasoning=self.reasoning,
            evidence_urls=self.evidence[:10],
            evidence_snapshots=self.evidence_snapshots,
            requires_human_review=self.requires_human_review,
            resolved_at=self.resolved_at,
        )


def _canonicalize_url(url: str) -> str:
    """Normalize a URL for de-duplication.

    Lowercases scheme+host, drops the fragment, strips tracking query params,
    and normalizes a bare trailing slash. Best-effort: returns the input on any
    parse failure rather than raising.
    """
    if not url:
        return ""
    raw = url.strip()
    if raw.startswith("//"):
        raw = "https:" + raw
    try:
        parts = urlsplit(raw)
        scheme = (parts.scheme or "https").lower()
        netloc = parts.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parts.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")
        kept = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=False)
            if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        query = urlencode(sorted(kept))
        return urlunsplit((scheme, netloc, path, query, ""))
    except Exception:  # pragma: no cover - defensive
        return raw


def _domain_of(url: str) -> str:
    try:
        netloc = urlsplit(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:  # pragma: no cover - defensive
        return url


def _build_claim(statement: str, deadline: Optional[datetime]) -> str:
    """Construct the deadline-baked claim that is actually judged.

    Baking the deadline and an "actually happened (not merely planned/expected)"
    framing into the claim is what stops a late or still-anticipated event from
    being scored CORRECT.
    """
    statement = (statement or "").strip()
    when = deadline.strftime("%Y-%m-%d") if deadline else "an unspecified date"
    return (
        f"As of {when}, the following has ACTUALLY HAPPENED (it has already "
        f"occurred and is confirmed — not merely planned, expected, predicted, "
        f"announced, or in progress): {statement}"
    )


def _make_snippet(text: str, quote: str) -> str:
    """Return a context window around ``quote`` within ``text``."""
    if not text:
        return ""
    if not quote:
        return text[: _SNIPPET_PAD * 2]
    idx = text.lower().find(quote.strip().lower())
    if idx < 0:
        return text[: _SNIPPET_PAD * 2]
    start = max(0, idx - _SNIPPET_PAD)
    end = min(len(text), idx + len(quote) + _SNIPPET_PAD)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


class ResolutionEngine:
    """Automated, auditable prediction resolution.

    For each past-deadline prediction the engine fetches real page text, judges
    it with a quote-grounded NLI judge against a deadline-baked claim, snapshots
    the evidence, and maps verified verdicts to a calibration-safe status.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        judge: Optional[EntailmentJudge] = None,
        search_fn: Optional[SearchFn] = None,
        fetch_fn: Optional[FetchFn] = None,
        max_sources: int = 5,
        max_fetch_chars: int = 4000,
        concurrency: int = 4,
    ):
        self._llm = llm
        self._judge: EntailmentJudge = judge or LLMEntailmentJudge(
            llm, max_evidence_chars=max_fetch_chars
        )
        self._search: SearchFn = search_fn or _web_search
        self._fetch: FetchFn = fetch_fn or _web_fetch
        self._max_sources = max(1, max_sources)
        self._max_fetch_chars = max(500, max_fetch_chars)
        self._sem = asyncio.Semaphore(max(1, concurrency))

    async def resolve(self, predictions: List[Prediction]) -> List[ResolutionOutcome]:
        """Resolve all past-deadline pending predictions."""
        now = datetime.now(timezone.utc)
        pending = [
            p for p in predictions
            if p.status == Status.PENDING and p.deadline and p.deadline < now
        ]

        if not pending:
            logger.info("No predictions past deadline to resolve")
            return []

        logger.info("Resolving %d past-deadline predictions...", len(pending))
        results = await asyncio.gather(
            *[self._resolve_one(p) for p in pending],
            return_exceptions=True,
        )

        outcomes: List[ResolutionOutcome] = []
        for pred, result in zip(pending, results):
            if isinstance(result, Exception):
                logger.warning("Resolution failed for %s: %s", pred.id, result)
                outcomes.append(
                    ResolutionOutcome(
                        prediction_id=pred.id,
                        statement=pred.statement,
                        deadline=pred.deadline,
                        resolved_as=Status.INSUFFICIENT_EVIDENCE,
                        confidence="low",
                        resolution_claim=_build_claim(pred.statement, pred.deadline),
                        resolution_reason=REASON_EXCEPTION,
                        reasoning=f"Resolution error: {result}",
                    )
                )
            else:
                outcomes.append(result)

        resolved_count = sum(
            1 for o in outcomes if o.resolved_as in (Status.CORRECT, Status.INCORRECT)
        )
        review_count = sum(1 for o in outcomes if o.requires_human_review)
        logger.info(
            "Resolved %d/%d predictions (%d flagged for human review)",
            resolved_count,
            len(pending),
            review_count,
        )
        return outcomes

    # -- candidate gathering --------------------------------------------------

    async def _gather_candidate_urls(self, statement: str) -> List[str]:
        """Collect, de-duplicate, and cap candidate source URLs."""
        seen_canonical: set[str] = set()
        ordered: List[str] = []

        def _add(url: str) -> None:
            if not url:
                return
            canon = _canonicalize_url(url)
            if canon and canon not in seen_canonical:
                seen_canonical.add(canon)
                ordered.append(url)

        queries = [
            f"{statement} outcome result",
            f"{statement} outcome result news",
        ]
        for ticker in self._extract_tickers(statement)[:2]:
            queries.append(f"{ticker} stock price")

        for query in queries:
            if len(ordered) >= self._max_sources:
                break
            try:
                results = await self._search(query, max_results=self._max_sources)
            except Exception as exc:
                logger.debug("Resolution search failed for %r: %s", query[:50], exc)
                continue
            for r in results or []:
                _add(getattr(r, "url", "") or "")

        return ordered[: self._max_sources]

    @staticmethod
    def _extract_tickers(statement: str) -> List[str]:
        tickers = re.findall(r"\$?[A-Z]{1,5}\b", (statement or "").upper())
        out: List[str] = []
        for t in tickers:
            t = t.lstrip("$")
            if t and t not in _TICKER_STOPWORDS and t not in out:
                out.append(t)
        return out

    # -- per-source fetch + judge --------------------------------------------

    async def _snapshot_and_judge(self, url: str, claim: str) -> EvidenceSnapshot:
        """Fetch a single URL, snapshot it, and judge it against the claim."""
        snap = EvidenceSnapshot(url=url, canonical_url=_canonicalize_url(url))
        async with self._sem:
            try:
                text = await self._fetch(url, max_chars=self._max_fetch_chars)
            except Exception as exc:
                snap.fetch_ok = False
                snap.fetch_error = str(exc)
                return snap

            text = text or ""
            if not text.strip():
                snap.fetch_ok = False
                snap.fetch_error = "empty content"
                return snap

            snap.content_chars = len(text)
            snap.content_truncated = len(text) >= self._max_fetch_chars
            snap.content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            try:
                judgment = await self._judge.judge(claim, text, source_url=url)
            except Exception as exc:
                logger.debug("NLI judge failed for %s: %s", url[:60], exc)
                snap.stance = "neutral"
                return snap

        snap.stance = judgment.stance
        if judgment.quote_verified and judgment.quote:
            snap.quote = judgment.quote
            snap.snippet = _make_snippet(text, judgment.quote)
        return snap

    async def _resolve_one(self, pred: Prediction) -> ResolutionOutcome:
        """Resolve a single prediction from real, quoted page text."""
        statement = pred.statement
        claim = _build_claim(statement, pred.deadline)
        logger.debug("Resolving: %s", statement[:80])

        urls = await self._gather_candidate_urls(statement)
        if not urls:
            return ResolutionOutcome(
                prediction_id=pred.id,
                statement=statement,
                deadline=pred.deadline,
                resolved_as=Status.INSUFFICIENT_EVIDENCE,
                confidence="low",
                resolution_claim=claim,
                resolution_reason=REASON_NO_SOURCES,
                reasoning="No candidate sources were found for this prediction.",
            )

        snapshots = await asyncio.gather(
            *[self._snapshot_and_judge(u, claim) for u in urls],
            return_exceptions=True,
        )
        clean: List[EvidenceSnapshot] = []
        for u, s in zip(urls, snapshots):
            if isinstance(s, Exception):
                logger.debug("Snapshot task error for %s: %s", u[:60], s)
                clean.append(
                    EvidenceSnapshot(
                        url=u,
                        canonical_url=_canonicalize_url(u),
                        fetch_ok=False,
                        fetch_error=str(s),
                    )
                )
            else:
                clean.append(s)

        return self._aggregate(pred, claim, urls, clean)

    # -- aggregation ----------------------------------------------------------

    def _aggregate(
        self,
        pred: Prediction,
        claim: str,
        urls: List[str],
        snapshots: List[EvidenceSnapshot],
    ) -> ResolutionOutcome:
        """Map verified per-source verdicts onto a calibration-safe status.

        Independence matters: agreement is counted by *distinct domain*, so ten
        syndicated copies of one wire story count once. A genuine conflict (both
        sides have verified evidence) is never silently coerced — it abstains and
        is flagged for human review.
        """
        support_domains = {
            _domain_of(s.url) for s in snapshots if s.stance == "supports"
        }
        contradict_domains = {
            _domain_of(s.url) for s in snapshots if s.stance == "contradicts"
        }
        e = len(support_domains)
        c = len(contradict_domains)

        any_fetch_ok = any(s.fetch_ok for s in snapshots)
        requires_review = e > 0 and c > 0

        if e > 0 and c > 0:
            if e == c:
                status = Status.INSUFFICIENT_EVIDENCE
                reason = REASON_CONFLICTING
                winning = 0
            elif e > c:
                status = Status.CORRECT
                reason = REASON_RESOLVED
                winning = e
            else:
                status = Status.INCORRECT
                reason = REASON_RESOLVED
                winning = c
        elif e > 0:
            status = Status.CORRECT
            reason = REASON_RESOLVED
            winning = e
        elif c > 0:
            status = Status.INCORRECT
            reason = REASON_RESOLVED
            winning = c
        else:
            status = Status.INSUFFICIENT_EVIDENCE
            winning = 0
            reason = REASON_ALL_NEUTRAL if any_fetch_ok else REASON_ALL_FETCHES_FAILED

        if status in (Status.CORRECT, Status.INCORRECT):
            confidence = "high" if winning >= 2 else "medium"
        else:
            confidence = "low"

        reasoning = self._explain(status, reason, e, c, snapshots)

        return ResolutionOutcome(
            prediction_id=pred.id,
            statement=pred.statement,
            deadline=pred.deadline,
            resolved_as=status,
            confidence=confidence,
            evidence=urls,
            reasoning=reasoning,
            resolution_claim=claim,
            resolution_reason=reason,
            requires_human_review=requires_review,
            evidence_snapshots=snapshots,
        )

    @staticmethod
    def _explain(
        status: Status,
        reason: str,
        e: int,
        c: int,
        snapshots: List[EvidenceSnapshot],
    ) -> str:
        fetched = sum(1 for s in snapshots if s.fetch_ok)
        base = (
            f"Judged {fetched}/{len(snapshots)} fetched source(s): "
            f"{e} independent domain(s) entailed the claim, "
            f"{c} contradicted it."
        )
        if reason == REASON_CONFLICTING:
            return base + " Verified evidence conflicts — abstaining and flagging for human review."
        if reason == REASON_ALL_NEUTRAL:
            return base + " No source produced a verified quote either way — insufficient evidence."
        if reason == REASON_ALL_FETCHES_FAILED:
            return "No source could be fetched and read — insufficient evidence."
        if status is Status.CORRECT:
            return base + " The prediction is judged CORRECT."
        if status is Status.INCORRECT:
            return base + " The prediction is judged INCORRECT."
        return base


__all__ = ["ResolutionEngine", "ResolutionOutcome"]
