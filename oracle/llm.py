"""LLM provider protocol — testable abstraction for all LLM calls.

All modules in The Oracle accept an LLMProvider for generation.
Uses Azure Foundry (Azure OpenAI) with gpt-5.4 by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Structured response from an LLM provider."""
    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @property
    def model_name(self) -> str: ...


class OpenAIProvider:
    """Azure Foundry (Azure OpenAI) LLM provider with gpt-5.4."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-12-01-preview",
    ):
        import os
        self._model = os.getenv("AZURE_OPENAI_DEPLOYMENT", model)
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._client is None:
            import os
            from openai import AsyncAzureOpenAI

            endpoint = self._azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise RuntimeError(
                    "AZURE_OPENAI_ENDPOINT is not configured. Set the "
                    "AZURE_OPENAI_ENDPOINT environment variable or pass "
                    "azure_endpoint=... to OpenAIProvider(). No default endpoint "
                    "is provided so the engine fails loudly rather than calling "
                    "an unexpected resource."
                )
            self._client = AsyncAzureOpenAI(
                api_key=self._api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=endpoint,
                api_version=self._api_version or os.getenv(
                    "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
                ),
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        usage_dict = {}
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return LLMResponse(
            content=content,
            model=response.model or self._model,
            usage=usage_dict,
        )


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, model: str = "mock-model"):
        self._model = model
        self._responses: List[str] = []
        self._index = 0
        self.calls: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model

    def set_response(self, *responses: str) -> None:
        self._responses = list(responses)
        self._index = 0

    def add_response(self, response: str) -> None:
        self._responses.append(response)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self._index < len(self._responses):
            content = self._responses[self._index]
            self._index += 1
        else:
            content = self._responses[-1] if self._responses else ""
        return LLMResponse(
            content=content,
            model=self._model,
            usage={"total_tokens": len(content.split())},
        )


__all__ = ["LLMResponse", "LLMProvider", "OpenAIProvider", "MockProvider"]
