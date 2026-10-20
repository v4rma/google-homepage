"""Background job queue so the UI never blocks on the model.

One worker thread runs pipeline calls in order; the API returns a job record immediately and the UI polls it.
With the agent bridge this means at most one request is outstanding in the inbox at a time."""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .logging_utils import log_error, log_event
from .textutil import new_id

ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobQueue:
    def __init__(self, provider: Any = None, keep: int = 200):
        self.provider = provider
        self.keep = keep
        self._jobs: dict[str, dict] = {}
        self._fns: dict[str, Callable[[], Any]] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._worker, name="avatar-jobs", daemon=True)
        self._thread.start()

    def submit(self, kind: str, label: str, fn: Callable[[], Any], episode_id: str | None = None, assumption_id: str | None = None) -> dict:
        job = {"id": new_id("job"), "kind": kind, "label": label, "episode_id": episode_id, "assumption_id": assumption_id,
               "status": "queued", "created_at": _now(), "started_at": None, "finished_at": None, "result": None, "error": None, "error_type": None}
        with self._lock:
            self._jobs[job["id"]] = job
            self._fns[job["id"]] = fn
            self._trim()
        self._q.put(job["id"])
        log_event("job.submitted", job=job["id"], kind=kind, episode_id=episode_id)
        return self._view(job)

    def get(self, job_id: str, wait: float = 0.0) -> dict | None:
        deadline = time.monotonic() + wait
        with self._changed:
            while True:
                job = self._jobs.get(job_id)
                if job is None or job["status"] not in ACTIVE:
                    return self._view(job) if job else None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._view(job)
                self._changed.wait(timeout=min(remaining, 1.0))

    def list(self, active_only: bool = False) -> list[dict]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if not active_only or j["status"] in ACTIVE]
        return [self._view(j, with_result=False) for j in jobs]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j["status"] in ACTIVE)

    def _view(self, job: dict, with_result: bool = True) -> dict:
        v = {k: v for k, v in job.items() if with_result or k != "result"}
        if job["status"] == "running" and self.provider is not None:
            v["waiting_on"] = getattr(self.provider, "current_request_id", None)
        return v

    def _trim(self) -> None:
        finished = [j for j in self._jobs.values() if j["status"] not in ACTIVE]
        for j in finished[: max(0, len(finished) - self.keep)]:
            self._jobs.pop(j["id"], None)
            self._fns.pop(j["id"], None)

    def _worker(self) -> None:
        while True:
            job_id = self._q.get()
            with self._lock:
                job = self._jobs.get(job_id)
                fn = self._fns.pop(job_id, None)
                if job is None or fn is None:
                    continue
                job["status"], job["started_at"] = "running", _now()
            try:
                result = fn()
                with self._changed:
                    job.update(status="done", result=result, finished_at=_now())
                    self._changed.notify_all()
                log_event("job.done", job=job_id, kind=job["kind"])
            except Exception as exc:  # noqa: BLE001 - the job record carries the error
                with self._changed:
                    job.update(status="failed", error=str(exc), error_type=type(exc).__name__, finished_at=_now())
                    self._changed.notify_all()
                log_error("job.failed", exc, job=job_id, kind=job["kind"])
