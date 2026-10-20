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
        r = client.post(f"/api/episodes/{ep['id']}/process")
        assert r.status_code == 200, r.text
    assumptions = client.get("/api/assumptions").json()
    assert assumptions and all("supporting" in a and "revisions" in a for a in assumptions)
    detail = client.get(f"/api/assumptions/{assumptions[0]['id']}").json()
    assert "interpretations" in detail
    # interview
    q = client.post("/api/interview/next", json={}).json()
    assert q["turn"] and q["stop"] is False
    assert client.post("/api/interview/next", json={}).json()["turn"]["id"] == q["turn"]["id"]
    ans = client.post(f"/api/interview/{q['turn']['id']}/answer", json={"answer": "I don't know"}).json()
    assert ans["turn"]["dont_know"] is True
    # corrections / rejection / delete
    aid = assumptions[0]["id"]
    c = client.post(f"/api/assumptions/{aid}/correct", json={"text": "If someone else fixes a schedule for me, then I lose my time."}).json()
    assert c["assumption"]["status"] == "revised" and c["assumption"]["user_corrections"]
    assert client.post(f"/api/assumptions/{aid}/reject", json={"text": ""}).json()["assumption"]["status"] == "rejected"
    assert client.patch(f"/api/assumptions/{aid}/status", json={"status": "bogus"}).status_code == 400
    assert client.delete(f"/api/assumptions/{aid}").status_code == 204
    assert client.get(f"/api/assumptions/{aid}").status_code == 404
    # holdout
    held = [e for e in eps if e["reserved"]][0]
    r = client.post(f"/api/holdout/{held['id']}/run")
    assert r.status_code == 200 and r.json()["predictions"]
    pid = r.json()["predictions"][0]["id"]
    assert client.patch(f"/api/predictions/{pid}/rating", json={"rating": "accurate"}).json()["user_rating"] == "accurate"
    assert client.post(f"/api/holdout/{eps[0]['id']}/run").status_code == 400
    status = client.get("/api/status").json()
    assert status["provider"] == "mock" and status["counts"]["episodes"] == 13


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Avatar" in r.text
    assert client.get("/static/app.js").status_code == 200
