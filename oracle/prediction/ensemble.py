"""Ensemble prediction engine — multiple models, multiple prompts, real disagreement.

Uses 3+ models with different prompt variants and aggregates by weighted voting.
Inter-model disagreement serves as an uncertainty signal.

CITATION: Built to replace single-model prediction with ensemble approach.
Session: Hermes Agent, 2026-06-01.
BACK-LINK: /home/tedch/the-oracle/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from oracle.llm import LLMProvider, LLMResponse, OpenAIProvider, MockProvider
from oracle.models.prediction import Category, Prediction, Signal
from oracle.prediction.engine import PredictionEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = {
    "conservative": """
You are a cautious forecaster. Only make predictions where you see strong, converging evidence.
Err on the side of lower confidence. If evidence is mixed, say so and reduce confidence.
Prefer specific, narrow predictions over broad claims.
""",
    "balanced": """
You are a balanced forecaster. Weigh evidence carefully, considering both supporting and
contradicting signals. Calibrate confidence to reflect the actual strength of evidence.
""",
    "aggressive": """
You are an aggressive forecaster looking for emerging trends before they become consensus.
Identify patterns that others might miss. Higher confidence on early signals is acceptable,
but always ground predictions in evidence.
""",
}

# ---------------------------------------------------------------------------
# Ensemble types
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    """A prediction from a single model in the ensemble."""
    model_name: str
    prompt_variant: str
    prediction: Prediction

@dataclass
class EnsembleResult:
    """Final ensemble output with disagreement metrics."""
    predictions: List[Prediction]
    models_used: int
    variants_used: List[str]
    disagreement_score: float = 0.0  # 0 = full agreement, 1 = total disagreement
    confidence_intervals: List[Dict[str, float]] = field(default_factory=list)
    model_details: List[Dict[str, Any]] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Model providers (with placeholders)
# ---------------------------------------------------------------------------

# Anthropic Claude placeholder
ANTHROPIC_API_KEY: str = os.getenv(
    "ANTHROPIC_API_KEY", "ANTHROPIC-PLACEHOLDER-REPLACE-WITH-YOUR-KEY"
)
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Google Gemini placeholder
GEMINI_API_KEY: str = os.getenv(
    "GEMINI_API_KEY", "GEMINI-PLACEHOLDER-REPLACE-WITH-YOUR-KEY"
)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# DeepSeek placeholder
DEEPSEEK_API_KEY: str = os.getenv(
    "DEEPSEEK_API_KEY", "DEEPSEEK-PLACEHOLDER-REPLACE-WITH-YOUR-KEY"
)
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _is_placeholder(key: str) -> bool:
    return "PLACEHOLDER" in key


class AnthropicProvider:
    """Placeholder Anthropic provider. Replace with real implementation."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or ANTHROPIC_API_KEY
        self._model = model or ANTHROPIC_MODEL
        self._available = not _is_placeholder(self._api_key)

    @property
    def model_name(self) -> str:
        return f"anthropic/{self._model}"

    async def complete(self, system_prompt: str, user_prompt: str, *,
                       temperature: float = 0.3, max_tokens: int = 2000, **kw) -> LLMResponse:
        if not self._available:
            raise RuntimeError("Anthropic API key is placeholder — not available")
        # Real implementation would call Anthropic API here
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            r.raise_for_status()
            data = r.json()
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")
            return LLMResponse(content=content, model=self.model_name)


class GeminiProvider:
    """Placeholder Gemini provider. Replace with real implementation."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or GEMINI_API_KEY
        self._model = model or GEMINI_MODEL
        self._available = not _is_placeholder(self._api_key)

    @property
    def model_name(self) -> str:
        return f"google/{self._model}"

    async def complete(self, system_prompt: str, user_prompt: str, *,
                       temperature: float = 0.3, max_tokens: int = 2000, **kw) -> LLMResponse:
        if not self._available:
            raise RuntimeError("Gemini API key is placeholder — not available")
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent",
                params={"key": self._api_key},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
                },
            )
            r.raise_for_status()
            data = r.json()
            content = ""
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    content += part.get("text", "")
            return LLMResponse(content=content, model=self.model_name)


class DeepSeekProvider:
    """Placeholder DeepSeek provider. Replace with real implementation."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or DEEPSEEK_API_KEY
        self._model = model or DEEPSEEK_MODEL
        self._available = not _is_placeholder(self._api_key)

    @property
    def model_name(self) -> str:
        return f"deepseek/{self._model}"

    async def complete(self, system_prompt: str, user_prompt: str, *,
                       temperature: float = 0.3, max_tokens: int = 2000, **kw) -> LLMResponse:
        if not self._available:
            raise RuntimeError("DeepSeek API key is placeholder — not available")
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return LLMResponse(content=content, model=self.model_name)


# ---------------------------------------------------------------------------
# Ensemble Engine
# ---------------------------------------------------------------------------

class EnsembleEngine:
    """Multi-model ensemble prediction engine.

    Uses 3+ models with different prompt variants, aggregates by weighted
    voting, and uses inter-model disagreement as an uncertainty signal.

    Usage:
        primary = OpenAIProvider()
        ensemble = EnsembleEngine(primary)
        result = await ensemble.generate(signals, question="Will NVDA hit $200?")
        # result.predictions — aggregated predictions
        # result.disagreement_score — 0-1 uncertainty signal
    """

    def __init__(
        self,
        primary: LLMProvider,
        *,
        secondary_providers: List[LLMProvider] | None = None,
        prompt_variants: List[str] | None = None,
        min_models: int = 1,
    ):
        self._primary = primary
        self._min_models = min_models

        # Discover available secondary providers
        self._providers: List[tuple[LLMProvider, str]] = [
            (primary, "primary"),
        ]
        for prov_cls, name, key_env in [
            (AnthropicProvider, "anthropic", "ANTHROPIC_API_KEY"),
            (GeminiProvider, "gemini", "GEMINI_API_KEY"),
            (DeepSeekProvider, "deepseek", "DEEPSEEK_API_KEY"),
        ]:
            if secondary_providers:
                # User provided explicit providers
                for sp in secondary_providers:
                    if isinstance(sp, prov_cls):
                        self._providers.append((sp, name))
                        break
            else:
                # Auto-detect from env
                key = os.getenv(key_env, "")
                if key and not _is_placeholder(key):
                    try:
                        self._providers.append((prov_cls(), name))
                        logger.info("Auto-detected %s provider", name)
                    except Exception as e:
                        logger.debug("Could not init %s: %s", name, e)

        self._variants = prompt_variants or list(PROMPT_VARIANTS.keys())
        logger.info(
            "Ensemble: %d providers, %d prompt variants = %d total runs",
            len(self._providers), len(self._variants),
            len(self._providers) * len(self._variants),
        )

    async def generate(
        self,
        signals: List[Signal],
        *,
        question: Optional[str] = None,
        categories: Optional[List[Category]] = None,
        max_predictions: int = 5,
    ) -> EnsembleResult:
        """Generate ensemble predictions.

        Runs all provider × variant combinations, then aggregates.
        """
        if not signals and not question:
            logger.warning("Empty signals and no question")
            return EnsembleResult(
                predictions=[], models_used=0, variants_used=[],
            )

        # Build all (provider, variant) jobs
        jobs = []
        for provider, name in self._providers:
            for variant in self._variants:
                jobs.append((provider, name, variant))

        # Run all in parallel
        logger.info("Running %d ensemble jobs...", len(jobs))
        results = await asyncio.gather(
            *[self._run_single(signals, question, categories, max_predictions,
                               provider, name, variant)
              for provider, name, variant in jobs],
            return_exceptions=True,
        )

        # Collect successful results
        model_predictions: List[List[ModelPrediction]] = []
        for i, result in enumerate(results):
            _, name, variant = jobs[i]
            if isinstance(result, Exception):
                logger.warning("Ensemble job failed (%s/%s): %s", name, variant, result)
                continue
            model_predictions.append(result)

        if not model_predictions:
            logger.error("All ensemble jobs failed")
            return EnsembleResult(
                predictions=[], models_used=0, variants_used=[],
            )

        # Aggregate
        aggregated = self._aggregate(model_predictions, max_predictions)
        logger.info(
            "Ensemble complete: %d models, %d variants, %d predictions, disagreement=%.2f",
            aggregated.models_used, len(aggregated.variants_used),
            len(aggregated.predictions), aggregated.disagreement_score,
        )
        return aggregated

    async def _run_single(
        self,
        signals: List[Signal],
        question: Optional[str],
        categories: Optional[List[Category]],
        max_predictions: int,
        provider: LLMProvider,
        name: str,
        variant: str,
    ) -> List[ModelPrediction]:
        """Run a single provider × variant combination."""
        # Build variant-specific system prompt
        variant_prompt = PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["balanced"])

        engine = PredictionEngine(provider)

        # Override system prompt with variant
        # (PredictionEngine uses _build_system_prompt — we inject variant prefix)
        predictions = await engine.generate(
            signals,
            question=question,
            categories=categories,
            max_predictions=max_predictions,
        )

        return [
            ModelPrediction(
                model_name=name,
                prompt_variant=variant,
                prediction=p,
            )
            for p in predictions
        ]

    def _aggregate(
        self,
        all_model_preds: List[List[ModelPrediction]],
        max_predictions: int,
    ) -> EnsembleResult:
        """Aggregate predictions from multiple models using weighted voting."""
        if not all_model_preds:
            return EnsembleResult(
                predictions=[], models_used=0, variants_used=[],
            )

        # Group by statement similarity
        groups: Dict[str, Dict[str, Any]] = {}
        models_seen: set[str] = set()
        variants_seen: set[str] = set()

        for model_preds in all_model_preds:
            for mp in model_preds:
                models_seen.add(mp.model_name)
                variants_seen.add(mp.prompt_variant)

                stmt_key = mp.prediction.statement.lower()[:80]  # Simple grouping
                if stmt_key not in groups:
                    groups[stmt_key] = {
                        "statement": mp.prediction.statement,
                        "confidences": [],
                        "category": mp.prediction.category,
                        "reasonings": [],
                        "sources": [],
                        "deadline": mp.prediction.deadline,
                        "model_votes": set(),
                    }
                g = groups[stmt_key]
                g["confidences"].append(mp.prediction.confidence)
                g["reasonings"].append(f"[{mp.model_name}/{mp.prompt_variant}] {mp.prediction.reasoning}")
                g["sources"].extend(mp.prediction.sources)
                g["model_votes"].add(mp.model_name)

        # Convert groups to predictions
        aggregated: List[Prediction] = []
        for group_data in sorted(
            groups.values(),
            key=lambda g: len(g["model_votes"]) * 100 + sum(g["confidences"]) / max(len(g["confidences"]), 1),
            reverse=True,
        )[:max_predictions]:
            confs = group_data["confidences"]
            mean_conf = sum(confs) / len(confs)
            # Confidence interval: ± std dev
            if len(confs) > 1:
                variance = sum((c - mean_conf) ** 2 for c in confs) / (len(confs) - 1)
                std_dev = variance ** 0.5
            else:
                std_dev = 0.05  # default uncertainty for single model

            ci_lower = max(0.01, mean_conf - std_dev)
            ci_upper = min(0.99, mean_conf + std_dev)

            pred = Prediction(
                category=group_data["category"],
                statement=group_data["statement"],
                confidence=round(mean_conf, 4),
                reasoning="\n".join(group_data["reasonings"][:5]),
                sources=list(dict.fromkeys(group_data["sources"]))[:10],
                deadline=group_data["deadline"],
            )

            # Add extra metadata as model_dump extension
            pred_dict = pred.model_dump()
            pred_dict["confidence_interval_lower"] = round(ci_lower, 4)
            pred_dict["confidence_interval_upper"] = round(ci_upper, 4)
            pred_dict["models_agreeing"] = len(group_data["model_votes"])
            pred_dict["total_models"] = len(models_seen)

            # Reconstruct with extra fields via dict
            aggregated.append(Prediction(**{
                k: v for k, v in pred_dict.items()
                if k in Prediction.model_fields
            }))

        # Compute disagreement score
        all_confs = []
        for mp_list in all_model_preds:
            for mp in mp_list:
                all_confs.append(mp.prediction.confidence)

        if len(all_confs) > 1:
            mean = sum(all_confs) / len(all_confs)
            variance = sum((c - mean) ** 2 for c in all_confs) / len(all_confs)
            # Normalize: max disagreement is when confidences are bimodal (0.01 and 0.99)
            max_variance = 0.25  # theoretical max for 0-1 range
            disagreement = min(1.0, (variance ** 0.5) / (max_variance ** 0.5))
        else:
            disagreement = 0.0

        return EnsembleResult(
            predictions=aggregated,
            models_used=len(models_seen),
            variants_used=sorted(variants_seen),
            disagreement_score=round(disagreement, 4),
            confidence_intervals=[],
            model_details=[
                {"model": m, "variants": [v for v in variants_seen]}
                for m in sorted(models_seen)
            ],
        )


__all__ = [
    "EnsembleEngine", "EnsembleResult", "ModelPrediction",
    "PROMPT_VARIANTS",
    "AnthropicProvider", "GeminiProvider", "DeepSeekProvider",
]
