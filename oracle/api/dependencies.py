"""API dependencies."""

import os
from functools import lru_cache
from oracle.storage.database import get_session
from oracle.storage.repository import PredictionRepository, SignalRepository
from oracle.llm import OpenAIProvider, LLMProvider, MockProvider


@lru_cache()
def get_llm() -> LLMProvider:
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return OpenAIProvider()
    return MockProvider()


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
