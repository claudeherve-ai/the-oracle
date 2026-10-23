"""API dependencies."""

import os
from functools import lru_cache
from oracle.storage.database import get_session
from oracle.storage.repository import PredictionRepository, SignalRepository
from oracle.llm import OpenAIProvider, LLMProvider, MockProvider
from oracle.prediction.engine import PredictionEngine


@lru_cache()
def get_llm() -> LLMProvider:
    """Return the configured LLM; never silently fall back to a mock.

    Predictions from a fake model would look real, so a missing endpoint fails
    loudly. Set ORACLE_LLM=mock explicitly for demos/tests.
    """
    if os.getenv("ORACLE_LLM", "").lower() == "mock":
        return MockProvider()
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set. Configure the Foundry endpoint "
            "(see .env.example) before requesting predictions."
        )
    return OpenAIProvider()


def build_engine(llm: LLMProvider | None = None) -> PredictionEngine:
    """Production engine: live verification and abstention on by default."""
    return PredictionEngine(
        llm or get_llm(),
        verification_mode=os.getenv("ORACLE_VERIFICATION", "live"),
        abstain_on_weak_evidence=os.getenv("ORACLE_ABSTAIN", "true").lower() != "false",
        abstain_threshold=float(os.getenv("ORACLE_ABSTAIN_THRESHOLD", "0.35")),
    )


def get_prediction_repo():
    session = get_session()
    try:
        yield PredictionRepository(session)
    finally:
        session.close()


def get_signal_repo():
    session = get_session()
    try:
        yield SignalRepository(session)
    finally:
        session.close()
