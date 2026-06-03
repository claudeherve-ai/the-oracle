"""Structured-output validation at the LLM boundary (A3).

Replaces fragile string-splitting JSON extraction with **Pydantic-validated**
parsing. Every candidate prediction is validated against a strict schema;
anything that fails is *rejected with a reason* (surfaced + logged), never
silently dropped. This is the contract that turns "the model said something
that vaguely looked like JSON" into "the model produced a value that conforms
to our schema, or we know exactly why it didn't."

Two complementary use-cases:

* **Mock / test mode** — the LLM returns plain-string JSON. ``parse_prediction_batch``
  robustly extracts the JSON (markdown fences, surrounding prose) and validates
  it. Invalid items become :class:`Rejection` records, not silent ``None``\\ s.
* **Production mode** — ``prediction_response_format()`` yields an OpenAI
  ``response_format`` payload (JSON-schema structured outputs) so the model is
  *constrained* to emit a conforming object in the first place. The same
  Pydantic models validate the result, giving one source of truth for the
  shape of a prediction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("oracle.tools.structured")

__all__ = [
    "PredictionDraft",
    "PredictionBatch",
    "Rejection",
    "ParsedBatch",
    "extract_json_block",
    "parse_prediction_batch",
    "prediction_json_schema",
    "prediction_response_format",
]


# ---------------------------------------------------------------------------
# Schema — the raw, LLM-emitted shape of a prediction (pre domain-logic)
# ---------------------------------------------------------------------------


class PredictionDraft(BaseModel):
    """A single prediction exactly as an LLM is asked to emit it.

    Deliberately permissive on *semantics* (category/deadline are free strings
    that the engine maps with sensible fallbacks) but strict on *structure*: a
    draft with no usable statement, a non-numeric confidence, or junk types is
    rejected at the boundary rather than silently coerced downstream.
    """

    model_config = ConfigDict(extra="ignore")

    statement: str = Field(
        min_length=10,
        max_length=1000,
        description="Specific, falsifiable, time-bound prediction.",
    )
    category: str = Field(
        default="tech_trend",
        description="One of the Oracle categories; mapped with fallback by the engine.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Calibrated probability the statement resolves TRUE (0..1).",
    )
    reasoning: str = Field(default="", description="Why this prediction, grounded in signals.")
    deadline: str = Field(default="", description="When it resolves (ISO date or relative phrase).")
    sources: List[str] = Field(
        default_factory=list,
        description="URLs / source identifiers grounding the prediction.",
    )

    @field_validator("category", "reasoning", "deadline", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, (list, tuple)):
            return [str(s).strip() for s in v if str(s).strip()]
        return [str(v).strip()] if str(v).strip() else []


class PredictionBatch(BaseModel):
    """The top-level object the LLM returns: ``{"predictions": [...]}``."""

    model_config = ConfigDict(extra="ignore")

    predictions: List[PredictionDraft] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result types — nothing is silently dropped
# ---------------------------------------------------------------------------


@dataclass
class Rejection:
    """A candidate prediction that failed validation, with a human reason."""

    index: int
    reason: str
    raw: Any = None


@dataclass
class ParsedBatch:
    """Outcome of parsing an LLM response.

    ``valid`` holds the drafts that passed; ``rejected`` records every item that
    failed and *why*; ``parse_error`` is set when the payload was not even valid
    JSON / not the expected container. Callers should surface rejections so a
    silently-malformed model never masquerades as "no predictions".
    """

    valid: List[PredictionDraft] = field(default_factory=list)
    rejected: List[Rejection] = field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.parse_error is None

    @property
    def rejection_count(self) -> int:
        return len(self.rejected)


# ---------------------------------------------------------------------------
# Robust JSON extraction (mock / test mode)
# ---------------------------------------------------------------------------


def extract_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object/array from free-form model text.

    Handles markdown code fences (```json ... ``` or bare ``` ... ```) and
    JSON embedded in surrounding prose. Returns ``None`` when no plausible JSON
    payload can be located.
    """

    if not text:
        return None

    content = text.strip()

    # 1) Markdown fences — prefer an explicitly tagged ```json block.
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.lower().startswith("json"):
                candidate = stripped[4:].strip()
                if candidate:
                    return candidate
            if stripped.startswith("{") or stripped.startswith("["):
                return stripped

    # 2) Already a bare JSON document.
    if content.startswith("{") or content.startswith("["):
        return content

    # 3) JSON embedded in prose — slice between the first opening and the last
    #    matching closing brace/bracket.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = content.find(open_ch)
        end = content.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return content[start : end + 1]

    return None


def _normalise_to_items(data: Any) -> Optional[List[Any]]:
    """Reduce a decoded JSON value to the list of prediction items."""

    if isinstance(data, dict):
        items = data.get("predictions", [])
        return items if isinstance(items, list) else None
    if isinstance(data, list):
        return data
    return None


def _format_validation_error(exc: ValidationError) -> str:
    """Condense a Pydantic error into a short, log-friendly reason."""

    parts: List[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "validation failed"


def parse_prediction_batch(payload: Union[str, dict, list]) -> ParsedBatch:
    """Parse + validate an LLM prediction payload into a :class:`ParsedBatch`.

    Accepts a raw response string (mock/test mode), or an already-decoded dict /
    list (production structured-output mode). Never raises — structural failures
    become ``parse_error`` and per-item failures become :class:`Rejection`\\ s.
    """

    # 1) Decode strings; pass through already-decoded structures.
    if isinstance(payload, str):
        json_str = extract_json_block(payload)
        if json_str is None:
            return ParsedBatch(parse_error="no JSON payload found in response")
        try:
            data: Any = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return ParsedBatch(parse_error=f"invalid JSON: {exc}")
    else:
        data = payload

    # 2) Reduce to a list of candidate items.
    items = _normalise_to_items(data)
    if items is None:
        return ParsedBatch(parse_error="payload is not a prediction batch (expected list or {'predictions': [...]})")

    # 3) Validate each item; record rejections rather than dropping them.
    batch = ParsedBatch()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            batch.rejected.append(Rejection(index=idx, reason="item is not an object", raw=item))
            continue
        try:
            batch.valid.append(PredictionDraft.model_validate(item))
        except ValidationError as exc:
            batch.rejected.append(
                Rejection(index=idx, reason=_format_validation_error(exc), raw=item)
            )

    if batch.rejected:
        logger.warning(
            "Structured parse rejected %d/%d prediction items: %s",
            len(batch.rejected),
            len(items),
            "; ".join(f"[{r.index}] {r.reason}" for r in batch.rejected),
        )

    return batch


# ---------------------------------------------------------------------------
# Production structured-output schema (OpenAI function-calling / response_format)
# ---------------------------------------------------------------------------


def prediction_json_schema() -> dict:
    """Return the JSON Schema describing a :class:`PredictionBatch`."""

    return PredictionBatch.model_json_schema()


def prediction_response_format(name: str = "prediction_batch") -> dict:
    """Return an OpenAI ``response_format`` payload for structured outputs.

    Pass this as ``response_format=prediction_response_format()`` so a capable
    model is *constrained* to emit a conforming object, then validate the result
    with :func:`parse_prediction_batch` for one source of truth on the shape.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": prediction_json_schema(),
        },
    }
