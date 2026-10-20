"""Agent bridge provider: the Cursor project agent is the model.

Each stage request is written as JSON to ``<bridge_dir>/analysis-inbox/<id>.json`` and the provider waits for
``<bridge_dir>/analysis-outbox/<id>.json``. The agent (or a processor it leaves running) reads the request, acts as
the model for that stage, and writes the JSON object the stage prompt asks for. ``scripts/agent_inbox.py`` wraps
listing, showing and answering requests.

Request file::

    {"id": "<stage>-<sha1[:12]>", "stage": str, "created_at": iso8601,
     "system": str,            # system prompt (common rules)
     "user": str,              # stage instructions + input records as JSON
     "payload": {...}}         # the structured input, also embedded in "user"

Answer file: either the bare stage JSON object, or ``{"id": ..., "response": {...}}``.

The request id is a hash of the stage and the full user prompt, so an identical request reuses an existing answer
(the outbox doubles as a cache, and an answer that arrives after a timeout is picked up by a retry). Pending work
for a processor is therefore: inbox files that have no outbox file of the same name.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging_utils import log_event
from .base import LLMError

INBOX = "analysis-inbox"
OUTBOX = "analysis-outbox"


class AgentUnavailable(LLMError):
    """The agent did not answer within the timeout. The request stays in the inbox."""


def request_id(stage: str, user_prompt: str) -> str:
    digest = hashlib.sha1((stage + "\n" + user_prompt).encode("utf-8")).hexdigest()[:12]
    return f"{stage}-{digest}"


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_answer(path: Path) -> dict[str, Any] | None:
    """Parse an answer file; None if it is not (yet) a complete JSON object."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "response" in obj and isinstance(obj["response"], dict) and set(obj) <= {"id", "response", "stage", "answered_at", "note"}:
        return obj["response"]
    return obj


def pending_requests(bridge_dir: str | Path) -> list[Path]:
    base = Path(bridge_dir)
    inbox, outbox = base / INBOX, base / OUTBOX
    if not inbox.exists():
        return []
    out = []
    for p in sorted(inbox.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if not (outbox / p.name).exists():
            out.append(p)
    return out


class AgentBridgeProvider:
    name = "agent"

    def __init__(self, bridge_dir: str, timeout: float = 120.0, poll_interval: float = 0.5):
        self.base = Path(bridge_dir)
        self.inbox = self.base / INBOX
        self.outbox = self.base / OUTBOX
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.current_request_id: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"bridge_dir": str(self.base), "inbox": str(self.inbox), "outbox": str(self.outbox), "timeout_s": self.timeout,
                "pending_requests": len(pending_requests(self.base))}

    def run(self, stage: str, system_prompt: str, user_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        rid = request_id(stage, user_prompt)
        req_path, ans_path = self.inbox / f"{rid}.json", self.outbox / f"{rid}.json"
        if not ans_path.exists() and not req_path.exists():
            write_json_atomic(req_path, {
                "id": rid, "stage": stage, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "system": system_prompt, "user": user_prompt, "payload": payload,
            })
            log_event("bridge.request_written", stage=stage, request=rid)
        self.current_request_id = rid
        deadline = time.monotonic() + self.timeout
        invalid_since: float | None = None
        try:
            while True:
                if ans_path.exists():
                    answer = read_answer(ans_path)
                    if answer is not None:
                        req_path.unlink(missing_ok=True)
                        log_event("bridge.answer_received", stage=stage, request=rid)
                        return answer
                    # A partially written or malformed answer: give the writer a few seconds, then reject it.
                    invalid_since = invalid_since or time.monotonic()
                    if time.monotonic() - invalid_since > 5:
                        bad = ans_path.with_suffix(".invalid.json")
                        os.replace(ans_path, bad)
                        log_event("bridge.answer_invalid", stage=stage, request=rid)
                        raise LLMError(f"the agent's answer for {rid} was not a JSON object (moved aside as {bad.name})")
                if time.monotonic() > deadline:
                    log_event("bridge.timeout", stage=stage, request=rid, timeout_s=self.timeout)
                    raise AgentUnavailable(
                        f"The Avatar agent has not answered request {rid} within {int(self.timeout)} s. "
                        "The request stays in the analysis inbox; retry once the agent has answered."
                    )
                time.sleep(self.poll_interval)
        finally:
            self.current_request_id = None
