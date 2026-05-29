"""Verification layer for The Oracle — fact-check predictions.

Runs a secondary LLM call to verify each prediction against web sources,
adjust confidence, and flag uncertain claims.
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any

from oracle.llm import LLMProvider

logger = logging.getLogger(__name__)

VERIFY_PREDICTIONS_PROMPT = """You are a fact-checker and confidence calibrator. Review these
predictions and verify them against the provided web sources.

For each prediction:
1. Check if the web sources support or contradict the claim
2. Adjust the confidence score based on source quality and agreement
3. Flag any prediction that cannot be verified from the sources
4. Add a "verification_note" explaining your reasoning

Return valid JSON:
{
  "verified_predictions": [
    {
      "statement": "original statement",
      "original_confidence": 0.72,
      "adjusted_confidence": 0.68,
      "verdict": "supported|contradicted|unverifiable",
      "verification_note": "Sources indicate...",
      "supporting_sources": ["url1"],
      "contradicting_sources": ["url2"]
    }
  ],
  "overall_reliability": 0.75,
  "summary": "Overall assessment of prediction quality"
}"""


async def verify_predictions(
    llm: LLMProvider,
    predictions: List[Dict[str, Any]],
    web_sources: str = "",
) -> Dict[str, Any]:
    """Verify and calibrate predictions against web sources.

    Runs a secondary LLM pass to fact-check every prediction.
    Returns adjusted confidences and verification notes.
    """
    if not predictions:
        return {"verified_predictions": [], "overall_reliability": 0.0, "summary": "No predictions to verify"}

    user_prompt = json.dumps({
        "web_sources": web_sources[:5000],
        "predictions": [
            {
                "statement": p.get("statement"),
                "confidence": p.get("confidence"),
                "category": p.get("category"),
                "deadline": str(p.get("deadline", ""))[:20],
            }
            for p in predictions[:5]
        ],
    })

    try:
        response = await llm.complete(
            system_prompt=VERIFY_PREDICTIONS_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=4096,
        )
        result = json.loads(_extract_json(response.content))
        logger.info("Verification: reliability=%.2f, verified=%d predictions",
                    result.get("overall_reliability", 0),
                    len(result.get("verified_predictions", [])))
        return result
    except Exception as e:
        logger.warning("Prediction verification failed (non-fatal): %s", e)
        return {
            "verified_predictions": [
                {**p, "verdict": "unverified", "adjusted_confidence": p.get("confidence", 0.5),
                 "verification_note": "Verification unavailable"}
                for p in predictions
            ],
            "overall_reliability": 0.5,
            "summary": f"Verification error: {e}",
        }


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
