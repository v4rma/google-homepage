import pytest
from fastapi.testclient import TestClient

from avatar.api import create_app
from avatar.config import Settings
from avatar.llm.mock import MockProvider
from avatar.storage import Store


@pytest.fixture
def client():
    settings = Settings(provider="mock", api_key=None, model="", base_url="", db_path=":memory:", log_level="WARNING")
    app = create_app(store=Store(":memory:"), provider=MockProvider(), settings=settings)
    return TestClient(app)


def run_job(client, resp):
    """Model-backed endpoints return a job (202); poll it to completion and return the result."""
    assert resp.status_code == 202, resp.text
    job = resp.json()
    while job["status"] in ("queued", "running"):
        job = client.get(f"/api/jobs/{job['id']}?wait=5").json()
    assert job["status"] == "done", job
    return job["result"]


EP = {"title": "Weekly run", "situation": 'A friend texted: "Want to join our Tuesday run? It\'s every week."', "immediate_thought": "Now I have to do this every week.",
      "emotion_body": "Irritation. My jaw went tight.", "action_outcome": "Left it unanswered.", "comparable_occasion": "I run alone whenever I like and enjoy it."}


def test_episode_crud_and_immutability(client):
    r = client.post("/api/episodes", json=EP)
    assert r.status_code == 201
    ep = r.json()
    assert client.get(f"/api/episodes/{ep['id']}").json()["observations"] == []
    assert client.patch(f"/api/episodes/{ep['id']}/reserved", json={"reserved": True}).json()["reserved"] is True
    assert client.get("/api/episodes/nope").status_code == 404
    assert client.post("/api/episodes", json={"situation": "x"}).status_code == 422
    assert client.delete(f"/api/episodes/{ep['id']}").status_code == 204


def test_full_flow_through_api(client):
    assert client.post("/api/demo/seed").json()["created"]
    eps = client.get("/api/episodes").json()
    assert len(eps) == 13
    for ep in [e for e in eps if not e["reserved"]][:3]:
        r = run_job(client, client.post(f"/api/episodes/{ep['id']}/process"))
        assert "hypothesize" in r
    assert client.post(f"/api/episodes/{[e for e in eps if e['reserved']][0]['id']}/process").status_code == 400
    assumptions = client.get("/api/assumptions").json()
    assert assumptions and all("supporting" in a and "revisions" in a for a in assumptions)
    detail = client.get(f"/api/assumptions/{assumptions[0]['id']}").json()
    assert "interpretations" in detail
    # interview
    q = run_job(client, client.post("/api/interview/next", json={}))
    assert q["turn"] and q["stop"] is False
    assert run_job(client, client.post("/api/interview/next", json={}))["turn"]["id"] == q["turn"]["id"]
    ans = run_job(client, client.post(f"/api/interview/{q['turn']['id']}/answer", json={"answer": "I don't know"}))
    assert ans["turn"]["dont_know"] is True
    assert client.post(f"/api/interview/{q['turn']['id']}/answer", json={"answer": "again"}).status_code == 400
    # corrections / rejection / delete
    aid = assumptions[0]["id"]
    assert client.post(f"/api/assumptions/{aid}/correct", json={"text": "  "}).status_code == 400
    c = run_job(client, client.post(f"/api/assumptions/{aid}/correct", json={"text": "If someone else fixes a schedule for me, then I lose my time."}))
    assert c["assumption"]["status"] == "revised" and len(c["assumption"]["user_corrections"]) == 1
    rj = client.post(f"/api/assumptions/{aid}/reject", json={"text": "not it"})
    assert client.get(f"/api/assumptions/{aid}").json()["status"] == "rejected"  # effective before the model answers
    assert run_job(client, rj)["assumption"]["status"] == "rejected"
    assert len(client.get(f"/api/assumptions/{aid}").json()["user_corrections"]) == 2
    assert client.patch(f"/api/assumptions/{aid}/status", json={"status": "bogus"}).status_code == 400
    assert client.delete(f"/api/assumptions/{aid}").status_code == 204
    assert client.get(f"/api/assumptions/{aid}").status_code == 404
    # holdout
    held = [e for e in eps if e["reserved"]][0]
    r = run_job(client, client.post(f"/api/holdout/{held['id']}/run"))
    assert r["predictions"]
    pid = r["predictions"][0]["id"]
    assert client.patch(f"/api/predictions/{pid}/rating", json={"rating": "accurate"}).json()["user_rating"] == "accurate"
    assert client.post(f"/api/holdout/{eps[0]['id']}/run").status_code == 400
    status = client.get("/api/status").json()
    assert status["provider"] == "mock" and status["counts"]["episodes"] == 13 and status["jobs_active"] == 0
    assert len(client.get("/api/jobs").json()) >= 8 and client.get("/api/jobs?active=1").json() == []
    assert client.get("/api/jobs/job-nope").status_code == 404


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Avatar" in r.text
    assert client.get("/static/app.js").status_code == 200
