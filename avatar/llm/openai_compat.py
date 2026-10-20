"""Provider for any OpenAI-compatible chat-completions endpoint (OpenAI, Azure-compatible proxies, local servers).

Privacy notes: the key comes from the environment only; request/response bodies are never logged; before using a
hosted provider, check its data-retention / training-opt-out settings (see README)."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ..logging_utils import log_error, log_event
from .base import LLMError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json(text: str) -> dict[str, Any]:
    text = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 120.0, temperature: float = 0.2):
        self._api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)

    def run(self, stage: str, system_prompt: str, user_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        last_exc: Exception | None = None
        for attempt in range(2):
            body = {"model": self.model, "messages": messages, "temperature": self.temperature, "response_format": {"type": "json_object"}}
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json=body,
                )
                if resp.status_code >= 400:
                    log_event("llm.http_error", stage=stage, status=resp.status_code, attempt=attempt)
                    raise LLMError(f"provider returned HTTP {resp.status_code}")
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                log_event("llm.ok", stage=stage, attempt=attempt, prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
                return _parse_json(content)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_exc = exc
                log_error("llm.bad_json", exc, stage=stage, attempt=attempt)
                messages = messages + [{"role": "user", "content": "Your previous reply was not a valid JSON object. Reply with only the JSON object."}]
            except httpx.HTTPError as exc:
                log_error("llm.transport_error", exc, stage=stage, attempt=attempt)
                raise LLMError("transport error talking to provider") from exc
        raise LLMError("provider did not return valid JSON") from last_exc
