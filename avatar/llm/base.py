from __future__ import annotations

from typing import Any, Protocol


class LLMError(Exception):
    """Raised when a provider cannot produce a usable JSON response. The message never includes user content."""


class LLMProvider(Protocol):
    name: str

    def run(self, stage: str, system_prompt: str, user_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute one pipeline stage and return the parsed JSON object."""
        ...
