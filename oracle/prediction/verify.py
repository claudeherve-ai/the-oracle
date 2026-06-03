"""NON-EVIDENTIARY self-consistency sanity flag for The Oracle.

.. warning::
    This module is **demoted** (roadmap item A4). It is *not* a source of
    evidence and *must never* drive confidence calibration, resolution, or any
    ``supported`` / ``contradicted`` label that feeds the calibration math.

    The canonical, evidence-driven verification path is
    :class:`oracle.prediction.verifier.VerificationEngine`, which uses real
    multi-source retrieval, NLI entailment, and quote verification. That engine
    is the ONLY thing allowed to adjust a prediction's confidence.

What this module does (and only this): a single secondary LLM pass that looks
at the *internal logical consistency* of a freshly generated prediction set —
e.g. two predictions that contradict each other, statements too vague to be
resolvable, or impossible/expired deadlines — and emits advisory
``self_consistency_warning`` strings. It reads NO external sources and produces
NO confidence numbers, verdicts, or resolution labels.

Output shape::

    {
      "self_consistency_warnings": [
        {"statement": "...", "self_consistency_warning": "..."}
      ],
      "summary": "advisory-only assessment of internal consistency"
    }
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, Dict, List

from oracle.llm import LLMProvider

logger = logging.getLogger(__name__)

#: Prompt for the advisory internal-consistency pass. It deliberately asks for
#: NO confidence numbers and NO supported/contradicted verdicts — only flags
#: about whether the set hangs together logically.
SELF_CONSISTENCY_PROMPT = """You are an internal-consistency reviewer. You are NOT a fact-checker
and you have NO access to external sources. Do not verify truth. Do not assign
confidence scores. Do not say whether a claim is true or false.

Only inspect this set of predictions for INTERNAL problems:
1. Two predictions that logically contradict each other.
2. A statement too vague or unfalsifiable to ever be resolved objectively.
3. An impossible, missing, or already-past deadline.

Return valid JSON ONLY in this exact shape:
{
  "self_consistency_warnings": [
    {"statement": "the original statement", "self_consistency_warning": "what is internally off"}
  ],
  "summary": "one-line advisory note about internal consistency"
}

If everything is internally consistent, return an empty warnings list."""


async def self_consistency_check(
    llm: LLMProvider,
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run an advisory-only internal-consistency pass over predictions.

    This is NON-EVIDENTIARY: it never returns confidence scores, verdicts, or
    resolution labels, and its output must not be used to adjust calibration.
    Callers may surface warnings to a human reviewer, nothing more.
    """
    if not predictions:
        return {"self_consistency_warnings": [], "summary": "No predictions to review"}

    user_prompt = json.dumps({
        "predictions": [
            {
                "statement": p.get("statement"),
                "category": p.get("category"),
                "deadline": str(p.get("deadline", ""))[:20],
            }
            for p in predictions[:5]
        ],
    })

    try:
        response = await llm.complete(
            system_prompt=SELF_CONSISTENCY_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=2048,
        )
        result = json.loads(_extract_json(response.content))
        warnings_list = result.get("self_consistency_warnings", [])
        logger.info("Self-consistency review: %d advisory warning(s)", len(warnings_list))
        return {
            "self_consistency_warnings": warnings_list,
            "summary": result.get("summary", ""),
        }
    except Exception as e:  # advisory pass must never be fatal
        logger.warning("Self-consistency review failed (non-fatal, advisory only): %s", e)
        return {
            "self_consistency_warnings": [],
            "summary": f"Self-consistency review unavailable: {e}",
        }


async def verify_predictions(
    llm: LLMProvider,
    predictions: List[Dict[str, Any]],
    web_sources: str = "",  # noqa: ARG001 - kept for back-compat signature
) -> Dict[str, Any]:
    """Deprecated back-compat shim. Use :class:`VerificationEngine` instead.

    Historically this performed LLM self-grading and adjusted confidence. That
    behaviour was removed (A4) because LLM self-grading is not evidence. This
    shim now delegates to the non-evidentiary :func:`self_consistency_check`
    and never adjusts confidence.
    """
    warnings.warn(
        "verify_predictions() is deprecated and non-evidentiary; use "
        "oracle.prediction.verifier.VerificationEngine for confidence "
        "adjustment, or self_consistency_check() for advisory warnings.",
        DeprecationWarning,
        stacklevel=2,
    )
    return await self_consistency_check(llm, predictions)


def _extract_json(content: str) -> str:
    text = content.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                return p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                return p
    return text
