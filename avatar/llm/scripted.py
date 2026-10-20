"""Scripted provider for tests: returns queued responses (optionally per stage) and records calls."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from .base import LLMError

Responder = Callable[[dict[str, Any]], dict[str, Any]]


class ScriptedProvider:
    name = "scripted"

    def __init__(self) -> None:
        self._queues: dict[str, deque] = {}
        self.calls: list[dict[str, Any]] = []

    def queue(self, stage: str, response: dict[str, Any] | Responder) -> "ScriptedProvider":
        self._queues.setdefault(stage, deque()).append(response)
        return self

    def run(self, stage: str, system_prompt: str, user_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"stage": stage, "system_prompt": system_prompt, "user_prompt": user_prompt, "payload": payload})
        q = self._queues.get(stage)
        if not q:
            raise LLMError(f"no scripted response for stage {stage}")
        item = q.popleft()
        return item(payload) if callable(item) else item
