"""LLM providers. All pipeline stages call ``provider.run(stage, system_prompt, user_prompt, payload)``
and receive a JSON-compatible dict."""

from __future__ import annotations

from ..config import Settings
from .base import LLMError, LLMProvider


def make_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "mock":
        from .mock import MockProvider

        return MockProvider()
    if settings.provider == "openai":
        from .openai_compat import OpenAICompatProvider

        if not settings.api_key:
            raise LLMError("AVATAR_LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAICompatProvider(api_key=settings.api_key, model=settings.model, base_url=settings.base_url)
    raise LLMError(f"unknown provider: {settings.provider}")


__all__ = ["LLMProvider", "LLMError", "make_provider"]
