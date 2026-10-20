import pytest

from avatar.storage import ImmutableEpisodeError, Store


def test_episode_roundtrip(store, episode):
    ep = store.get_episode(episode["id"])
    assert ep["situation"].startswith("A friend texted")
    assert ep["reserved"] is False
    assert store.list_episodes() == [ep]


def test_original_account_is_immutable(store, episode):
    with pytest.raises(ImmutableEpisodeError):
        store.update_episode_content(episode["id"], situation="rewritten")
    with pytest.raises(Exception):
        store.conn.execute("UPDATE episodes SET immediate_thought=? WHERE id=?", ("edited", episode["id"]))
    assert store.get_episode(episode["id"])["immediate_thought"] == "Now I have to do this every week."


def test_reserved_flag_can_change_but_content_cannot(store, episode):
    ep = store.set_reserved(episode["id"], True)
    assert ep["reserved"] is True
    assert store.list_episodes(include_reserved=False) == []


def test_interview_material_is_separate_from_original(store, episode):
    turn = store.add_turn(episode["id"], "What did you expect?", discriminates_between=["as-x"])
    store.answer_turn(turn["id"], "That my weeks would fill up.", dont_know=False)
    ep = store.get_episode(episode["id"])
    assert "fill up" not in Store.episode_text(ep)
    assert any("fill up" in s for s in store.user_sources(episode["id"]))
    assert store.pending_turn() is None


def test_evidence_quotation_verification(store, episode):
    a = store.add_assumption("If I commit, then I lose my time.")
    ok = store.add_evidence(a["id"], episode["id"], "Want to join our Tuesday run? It's every week.", "supporting")
    assert ok["verified"] is True
    curly = store.add_evidence(a["id"], episode["id"], "“now I have to do this every week”", "supporting")
    assert curly["verified"] is True
    fake = store.add_evidence(a["id"], episode["id"], "I always hated commitments", "supporting")
    assert fake["verified"] is False
    full = store.get_assumption(a["id"])
    assert full["supporting_episode_ids"] == [episode["id"]]
    assert len(full["supporting"]) == 3
    assert store.add_evidence(a["id"], "ep-missing", "anything", "supporting") is None
    assert store.add_evidence(a["id"], episode["id"], "Want to join our Tuesday run? It's every week.", "supporting") is None  # dedupe


def test_evidence_from_answer_verifies_against_answer(store, episode):
    a = store.add_assumption("If I commit, then I lose my time.")
    turn = store.add_turn(episode["id"], "Q?")
    store.answer_turn(turn["id"], "It is the fixed schedule that gets me.", dont_know=False)
    ev = store.add_evidence(a["id"], episode["id"], "the fixed schedule that gets me", "supporting", turn_id=turn["id"])
    assert ev["verified"] is True and ev["turn_id"] == turn["id"]


def test_observations_and_interpretations_are_separate_tables(store, episode):
    store.add_observation(episode["id"], "quotation", "Want to join our Tuesday run?", source="extract", verified=True)
    store.add_interpretation("hypothesize", {"rationale": "maybe autonomy"}, episode_id=episode["id"])
    assert [o["kind"] for o in store.list_observations(episode["id"])] == ["quotation"]
    assert store.list_interpretations(episode_id=episode["id"])[0]["content"] == {"rationale": "maybe autonomy"}
    tables = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"observations", "interpretations", "assumption_evidence"} <= tables


def test_revision_history_and_status_transitions(store):
    a = store.add_assumption("If X then Y", situations=["s1"])
    assert a["status"] == "proposed" and a["version"] == 1
    a = store.revise_assumption(a["id"], {"proposition": "If X' then Y", "status": "revised"}, trigger="user_correction", reason="narrowed")
    assert a["version"] == 2 and a["status"] == "revised"
    assert a["revisions"][0]["previous"]["proposition"] == "If X then Y"
    assert a["revisions"][0]["trigger"] == "user_correction"
    a = store.revise_assumption(a["id"], {"status": "bogus"}, trigger="x")
    assert a["version"] == 2  # invalid status ignored, no revision recorded
    a = store.set_status(a["id"], "rejected", trigger="user_rejection")
    assert a["status"] == "rejected" and a["version"] == 3


def test_corrections_are_preserved(store):
    a = store.add_assumption("If X then Y")
    a = store.add_correction(a["id"], "Only when someone else sets it")
    assert a["user_corrections"][0]["text"] == "Only when someone else sets it"
    assert a["user_corrections"][0]["version"] == 1


def test_merge_lists_dedupes(store):
    a = store.add_assumption("If X then Y", weakeners=["w1"])
    a = store.merge_lists(a["id"], "weakeners", ["w1", "w2"], trigger="critique")
    assert a["weakeners"] == ["w1", "w2"] and a["version"] == 2
    a = store.merge_lists(a["id"], "weakeners", ["w2"], trigger="critique")
    assert a["version"] == 2


def test_delete_cascades(store, episode):
    a = store.add_assumption("If X then Y")
    store.add_evidence(a["id"], episode["id"], "every week", "supporting")
    store.add_interpretation("critique", {"x": 1}, assumption_id=a["id"])
    store.delete_assumption(a["id"])
    assert store.conn.execute("SELECT COUNT(*) FROM assumption_evidence").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0] == 0
    b = store.add_assumption("If P then Q")
    store.add_evidence(b["id"], episode["id"], "every week", "supporting")
    store.delete_episode(episode["id"])
    assert store.get_assumption(b["id"])["supporting"] == []


def test_predictions_and_runs(store, episode):
    store.set_reserved(episode["id"], True)
    p = store.add_prediction(episode["id"], "batch1", None, predicted_thought="t", confidence=0.5, match_score=0.7, comparison="ok")
    assert p["match_score"] == 0.7 and p["assumption_id"] is None
    assert store.rate_prediction(p["id"], "accurate")["user_rating"] == "accurate"
    rid = store.start_run("extract", episode["id"])
    store.finish_run(rid, ok=False, error_type="LLMError")
    run = store.list_runs()[0]
    assert run["ok"] is False and run["error_type"] == "LLMError"
    assert "situation" not in run


def test_persistence_on_disk(tmp_path):
    path = str(tmp_path / "a.db")
    s = Store(path)
    s.add_episode("s", "t", "e", "a")
    s.close()
    assert Store(path).counts()["episodes"] == 1
