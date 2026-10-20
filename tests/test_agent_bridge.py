"""The agent bridge: requests written to the inbox, answers read from the outbox, timeouts, caching, jobs."""

import json
import threading
import time

import pytest

from avatar.config import default_provider
from avatar.jobs import JobQueue
from avatar.llm.agent_bridge import INBOX, OUTBOX, AgentBridgeProvider, AgentUnavailable, pending_requests, read_answer, request_id
from avatar.llm.base import LLMError
from avatar.pipeline import Pipeline


def answer_later(bridge_dir, delay, body=None, raw=None):
    """Simulate the agent: wait for the pending request, then write an answer."""
    def worker():
        deadline = time.time() + 5
        while time.time() < deadline:
            pend = pending_requests(bridge_dir)
            if pend:
                time.sleep(delay)
                req = json.loads(pend[0].read_text())
                out = bridge_dir / OUTBOX / pend[0].name
                out.write_text(raw if raw is not None else json.dumps(body if body is not None else {"id": req["id"], "response": {"echo": req["stage"]}}))
                return
            time.sleep(0.05)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def test_default_provider_selection():
    assert default_provider(None) == "agent"
    assert default_provider("sk-x") == "openai"


def test_request_written_and_answer_consumed(tmp_path):
    p = AgentBridgeProvider(str(tmp_path), timeout=5, poll_interval=0.05)
    answer_later(tmp_path, 0.2)
    result = p.run("extract", "SYS", "USER prompt", {"episode": {"id": "ep-1"}})
    assert result == {"echo": "extract"}
    assert pending_requests(tmp_path) == []  # request removed once consumed
    assert list((tmp_path / OUTBOX).glob("*.json"))  # answer kept as cache
    assert p.current_request_id is None


def test_request_file_shape(tmp_path):
    p = AgentBridgeProvider(str(tmp_path), timeout=0.3, poll_interval=0.05)
    with pytest.raises(AgentUnavailable):
        p.run("critique", "SYS", "USER", {"assumption": {"id": "as-1"}})
    pend = pending_requests(tmp_path)
    assert len(pend) == 1
    req = json.loads(pend[0].read_text())
    assert req["id"] == request_id("critique", "USER") == pend[0].stem
    assert set(req) == {"id", "stage", "created_at", "system", "user", "payload"}
    assert req["payload"] == {"assumption": {"id": "as-1"}}


def test_timeout_keeps_request_and_retry_picks_up_late_answer(tmp_path):
    p = AgentBridgeProvider(str(tmp_path), timeout=0.3, poll_interval=0.05)
    with pytest.raises(AgentUnavailable) as exc:
        p.run("hypothesize", "S", "U", {})
    assert "inbox" in str(exc.value) and "U" not in str(exc.value).replace("Avatar", "")
    rid = pending_requests(tmp_path)[0].stem
    (tmp_path / OUTBOX / f"{rid}.json").write_text(json.dumps({"candidates": [], "cannot_tell": True}))
    assert p.run("hypothesize", "S", "U", {}) == {"candidates": [], "cannot_tell": True}
    assert pending_requests(tmp_path) == []


def test_identical_request_reuses_cached_answer_without_new_inbox_file(tmp_path):
    p = AgentBridgeProvider(str(tmp_path), timeout=2, poll_interval=0.05)
    answer_later(tmp_path, 0.1, body={"events": []})
    assert p.run("extract", "S", "U", {}) == {"events": []}
    assert p.run("extract", "S", "U", {}) == {"events": []}
    assert list((tmp_path / INBOX).glob("*.json")) == []


def test_bare_object_and_fenced_answers_are_accepted(tmp_path):
    p = AgentBridgeProvider(str(tmp_path), timeout=2, poll_interval=0.05)
    answer_later(tmp_path, 0.1, raw='```json\n{"verdict": "keep"}\n```')
    assert p.run("critique", "S", "U1", {}) == {"verdict": "keep"}
    assert read_answer(tmp_path / OUTBOX / f"{request_id('critique', 'U1')}.json") == {"verdict": "keep"}


def test_invalid_answer_is_moved_aside(tmp_path):
    p = AgentBridgeProvider(str(tmp_path), timeout=10, poll_interval=0.05)
    rid = request_id("extract", "U2")
    (tmp_path / OUTBOX / f"{rid}.json").write_text("this is not json")
    start = time.time()
    with pytest.raises(LLMError):
        p.run("extract", "S", "U2", {})
    assert 5 <= time.time() - start < 9
    assert (tmp_path / OUTBOX / f"{rid}.invalid.json").exists()


def test_pipeline_through_bridge(tmp_path, store, episode):
    p = AgentBridgeProvider(str(tmp_path), timeout=3, poll_interval=0.05)
    pipeline = Pipeline(store, p)
    answer_later(tmp_path, 0.1, body={"events": ["A friend texted."], "quotations": ["Want to join our Tuesday run? It's every week."], "expectations": [], "emotions": [], "actions": [], "missing": []})
    obs = pipeline.extract(episode["id"])
    assert any(o["kind"] == "quotation" and o["verified"] for o in obs)
    assert store.list_runs()[-1]["ok"] is True


def test_job_queue_runs_in_order_and_reports_failures():
    q = JobQueue()
    order = []
    j1 = q.submit("a", "first", lambda: (time.sleep(0.1), order.append(1), "r1")[-1])
    j2 = q.submit("b", "second", lambda: (order.append(2), 1 / 0)[-1])
    assert j1["status"] in ("queued", "running")
    done1 = q.get(j1["id"], wait=5)
    done2 = q.get(j2["id"], wait=5)
    assert done1["status"] == "done" and done1["result"] == "r1"
    assert done2["status"] == "failed" and done2["error_type"] == "ZeroDivisionError"
    assert order == [1, 2]
    assert q.list(active_only=True) == [] and len(q.list()) == 2 and "result" not in q.list()[0]
    assert q.get("job-missing") is None


def test_job_exposes_waiting_request(tmp_path):
    p = AgentBridgeProvider(str(tmp_path), timeout=1.5, poll_interval=0.05)
    q = JobQueue(p)
    j = q.submit("extract", "x", lambda: p.run("extract", "S", "U3", {}))
    time.sleep(0.3)
    running = q.get(j["id"])
    assert running["status"] == "running" and running["waiting_on"] == request_id("extract", "U3")
    final = q.get(j["id"], wait=5)
    assert final["status"] == "failed" and final["error_type"] == "AgentUnavailable"
