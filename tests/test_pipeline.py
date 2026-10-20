"""Pipeline tests with a scripted (fully controlled) LLM and with the rule-based mock."""

import logging

import pytest

from avatar.demo_data import seed, synthetic_answer
from avatar.llm.base import LLMError
from avatar.logging_utils import configure_logging
from avatar.pipeline.common import GUARD_NOTICE
from avatar.retrieval import bm25_rank, retrieve_related_episodes


# ---------------------------------------------------------------- extract
def test_extract_marks_verbatim_quotes_and_paraphrases(pipeline, scripted, episode):
    scripted.queue("extract", {
        "events": ["A friend texted an invitation."],
        "quotations": ["Want to join our Tuesday run? It's every week.", "I never want to run with anyone"],
        "expectations": ["Now I have to do this every week."],
        "emotions": ["Irritation."], "actions": [], "missing": ["No outcome given."],
    })
    obs = pipeline.extract(episode["id"])
    by = {(o["kind"], o["content"]): o["verified"] for o in obs}
    assert by[("quotation", "Want to join our Tuesday run? It's every week.")] is True
    assert by[("quotation", "I never want to run with anyone")] is False
    assert by[("event", "A friend texted an invitation.")] is False
    assert by[("missing", "No outcome given.")] is True
    assert scripted.calls[0]["stage"] == "extract"
    assert "interview_answers" not in scripted.calls[0]["payload"]["episode"]


# ---------------------------------------------------------------- hypothesize
def test_hypothesize_creates_candidates_with_verified_provenance(pipeline, scripted, store, episode):
    other = store.add_episode("Manager said: \"everyone joins the 9am standup every day.\"", "There goes my morning.", "Irritation", "Said fine, arrived late.")
    scripted.queue("hypothesize", lambda p: {
        "candidates": [
            {"proposition": "If I agree to a recurring commitment, then I lose control over my time.", "kind": "if_then",
             "situations": ["recurring requests"], "supporting": [
                 {"episode_id": episode["id"], "quotation": "Now I have to do this every week."},
                 {"episode_id": other["id"], "quotation": "There goes my morning."},
                 {"episode_id": other["id"], "quotation": "I hate my manager"},
                 {"episode_id": "ep-not-in-context", "quotation": "There goes my morning."}],
             "contradicting": [], "alternative_explanations": ["tiredness"], "prediction_new_situation": "irritation", "weakeners": ["a calm commitment"],
             "relates_to_existing": None, "rationale": "pattern"},
            {"proposition": "The reaction is explained by tiredness that day.", "kind": "situational", "situations": [], "supporting": [], "contradicting": [],
             "alternative_explanations": [], "prediction_new_situation": "", "weakeners": [], "relates_to_existing": None, "rationale": "situational"},
            {"proposition": "", "kind": "if_then"},
        ],
        "cannot_tell": False, "note": "",
    })
    r = pipeline.hypothesize(episode["id"])
    assert len(r["assumptions"]) == 2
    a = r["assumptions"][0]
    assert a["status"] == "proposed"
    assert sorted(a["supporting_episode_ids"]) == sorted([episode["id"], other["id"]])
    assert [e["verified"] for e in a["supporting"]] == [True, True, False]  # fabricated quote stored but flagged
    assert r["related_episode_ids"] == [other["id"]]
    payload = scripted.calls[0]["payload"]
    assert payload["max_candidates"] == 3 and payload["related_episodes"][0]["id"] == other["id"]
    assert store.list_interpretations(assumption_id=a["id"])[0]["content"]["rationale"] == "pattern"


def test_hypothesize_attaches_to_existing_instead_of_duplicating(pipeline, scripted, store, episode):
    existing = store.add_assumption("If I agree to a recurring commitment, then I lose control over my time.")
    scripted.queue("hypothesize", lambda p: {"candidates": [{"proposition": "same idea", "kind": "if_then", "situations": ["new situation"],
                                                             "supporting": [{"episode_id": episode["id"], "quotation": "My jaw went tight."}],
                                                             "relates_to_existing": existing["id"], "rationale": "dup"}], "cannot_tell": False})
    r = pipeline.hypothesize(episode["id"])
    assert [a["id"] for a in r["assumptions"]] == [existing["id"]]
    a = store.get_assumption(existing["id"])
    assert a["proposition"].startswith("If I agree") and a["situations"] == ["new situation"]
    assert a["supporting_episode_ids"] == [episode["id"]]
    assert scripted.calls[0]["payload"]["existing_assumptions"][0]["id"] == existing["id"]


def test_hypothesize_refuses_reserved_and_cannot_tell(pipeline, scripted, store, episode):
    store.set_reserved(episode["id"], True)
    with pytest.raises(ValueError):
        pipeline.hypothesize(episode["id"])
    store.set_reserved(episode["id"], False)
    scripted.queue("hypothesize", {"candidates": [], "cannot_tell": True, "note": "ordinary"})
    r = pipeline.hypothesize(episode["id"])
    assert r["cannot_tell"] is True and r["assumptions"] == []


def test_reserved_episodes_never_reach_the_model(pipeline, scripted, store, episode):
    held = store.add_episode("Reserved: a weekly rota request every week.", "ugh every week", "irritation", "declined", reserved=True)
    assert retrieve_related_episodes(store, episode["id"]) == []
    scripted.queue("hypothesize", {"candidates": [], "cannot_tell": True})
    pipeline.hypothesize(episode["id"])
    assert held["id"] not in str(scripted.calls[0]["payload"])


# ---------------------------------------------------------------- critique
def test_critique_records_interpretation_and_moves_status(pipeline, scripted, store, episode):
    a = store.add_assumption("If I commit, then I lose my time.", origin_episode_id=episode["id"])
    store.add_evidence(a["id"], episode["id"], "every week", "supporting")
    scripted.queue("critique", {"contradictions": [{"episode_id": episode["id"], "quotation": "I run on my own whenever I feel like it and enjoy it.", "note": "chosen freely"}],
                                "unsupported_leaps": ["one episode"], "alternative_explanations": ["dislikes running"], "weakeners": ["calm commitment"],
                                "resolving_questions": [{"question": "When was it different?", "episode_id": episode["id"], "discriminates_between": [a["id"], "as-nope"]}], "verdict": "needs_examination"})
    r = pipeline.critique(a["id"])
    a2 = r["assumption"]
    assert a2["status"] == "under examination"
    assert a2["alternative_explanations"] == ["dislikes running"] and a2["weakeners"] == ["calm commitment"]
    assert a2["contradicting"][0]["verified"] is True
    assert r["critique"]["resolving_questions"][0]["discriminates_between"] == [a["id"]]
    interp = [i for i in store.list_interpretations(assumption_id=a["id"]) if i["stage"] == "critique"]
    assert interp and interp[0]["content"]["verdict"] == "needs_examination"
    # the critique's claims are interpretations: they never became supporting evidence
    assert a2["supporting_episode_ids"] == [episode["id"]] and len(a2["supporting"]) == 1


# ---------------------------------------------------------------- interview
def _two_candidates(store, episode):
    a = store.add_assumption("If I commit to a schedule, then I lose my time.", origin_episode_id=episode["id"])
    b = store.add_assumption("If others watch me, then I will be judged.", origin_episode_id=episode["id"])
    store.add_evidence(a["id"], episode["id"], "every week", "supporting")
    return a, b


def test_interview_one_question_at_a_time(pipeline, scripted, store, episode):
    a, b = _two_candidates(store, episode)
    scripted.queue("interview_question", {"question": "When you chose it yourself but others watched, how was it?", "episode_id": episode["id"],
                                          "discriminates_between": [a["id"], b["id"], "as-ghost"], "rationale": "separates autonomy from evaluation", "stop": False})
    r = pipeline.next_question()
    turn = r["turn"]
    assert turn["discriminates_between"] == [a["id"], b["id"]]
    again = pipeline.next_question()
    assert again["turn"]["id"] == turn["id"] and "already waiting" in again["reason"]
    assert len([c for c in scripted.calls if c["stage"] == "interview_question"]) == 1


def test_interview_stop_and_no_candidates(pipeline, scripted, store, episode):
    assert pipeline.next_question()["stop"] is True
    _two_candidates(store, episode)
    scripted.queue("interview_question", {"question": "", "stop": True, "stop_reason": "speculation"})
    r = pipeline.next_question()
    assert r["stop"] is True and r["reason"] == "speculation" and store.pending_turn() is None


def test_dont_know_is_accepted_without_llm_call(pipeline, scripted, store, episode):
    _two_candidates(store, episode)
    turn = store.add_turn(episode["id"], "What was at stake?")
    r = pipeline.answer(turn["id"], "I don't know, honestly.")
    assert r["turn"]["dont_know"] is True and r["updates"] == []
    assert all(c["stage"] != "interview_update" for c in scripted.calls)
    assert store.list_observations(episode["id"])[0]["source"] == "interview_answer"


def test_answer_updates_candidates(pipeline, scripted, store, episode):
    a, b = _two_candidates(store, episode)
    turn = store.add_turn(episode["id"], "Chosen but watched: how was it?", discriminates_between=[a["id"], b["id"]])
    scripted.queue("interview_update", {"updates": [
        {"assumption_id": a["id"], "action": "strengthen", "reason": "answer favours autonomy", "supporting": [{"episode_id": episode["id"], "quotation": "It felt fine"}]},
        {"assumption_id": b["id"], "action": "reject", "reason": "chosen-but-watched felt fine", "contradicting": [{"episode_id": episode["id"], "quotation": "even with everyone watching"}]},
        {"assumption_id": "as-ghost", "action": "revise"},
    ], "new_candidates": [{"proposition": "If the activity is dull, then I avoid it.", "kind": "if_then", "supporting": [], "rationale": "new"}], "stop_interview": False})
    r = pipeline.answer(turn["id"], "It felt fine, even with everyone watching.")
    assert [(u["assumption_id"], u["action"]) for u in r["updates"]] == [(a["id"], "strengthen"), (b["id"], "reject")]
    a2, b2 = store.get_assumption(a["id"]), store.get_assumption(b["id"])
    assert a2["status"] == "under examination" and len(a2["supporting"]) == 2 and a2["supporting"][1]["verified"] and a2["supporting"][1]["turn_id"] == turn["id"]
    assert b2["status"] == "rejected" and b2["revisions"][-1]["trigger"] == "interview_answer"
    assert len(r["new_assumptions"]) == 1 and r["new_assumptions"][0]["status"] == "proposed"
    with pytest.raises(ValueError):
        pipeline.answer(turn["id"], "again")


def test_revise_action_keeps_history(pipeline, scripted, store, episode):
    a, _ = _two_candidates(store, episode)
    turn = store.add_turn(episode["id"], "Q", discriminates_between=[a["id"]])
    scripted.queue("interview_update", {"updates": [{"assumption_id": a["id"], "action": "revise", "new_proposition": "If someone else sets the schedule, then I lose my time.",
                                                     "new_weakeners": ["self-set schedules also irritate"], "reason": "narrowed"}], "new_candidates": []})
    pipeline.answer(turn["id"], "Only when it is not my schedule.")
    a2 = store.get_assumption(a["id"])
    assert a2["status"] == "revised" and a2["proposition"].startswith("If someone else") and a2["weakeners"] == ["self-set schedules also irritate"]
    assert a2["revisions"][-1]["previous"]["proposition"].startswith("If I commit")


def test_resistance_language_is_removed(pipeline, scripted, store, episode, caplog):
    a, _ = _two_candidates(store, episode)
    turn = store.add_turn(episode["id"], "Q", discriminates_between=[a["id"]])
    scripted.queue("interview_update", {"updates": [{"assumption_id": a["id"], "action": "keep", "reason": "The user's denial shows resistance to the insight."}], "new_candidates": []})
    r = pipeline.answer(turn["id"], "No, that is not it.")
    assert r["updates"][0]["reason"] == GUARD_NOTICE


# ---------------------------------------------------------------- user corrections and rejection
def test_user_correction_is_authoritative_and_revises(pipeline, scripted, store, episode):
    a, _ = _two_candidates(store, episode)
    scripted.queue("revise_correction", {"proposition": "If someone else fixes a recurring schedule for me, then I lose control over my time.", "kind": "if_then",
                                         "situations": ["schedules set by others"], "weakeners": [], "alternative_explanations": [], "prediction_new_situation": "", "reason": "re-worded"})
    r = pipeline.correct(a["id"], "It is only when someone else sets the schedule.")
    a2 = r["assumption"]
    assert a2["status"] == "revised" and a2["version"] == 2
    assert a2["user_corrections"][0]["text"] == "It is only when someone else sets the schedule."
    assert a2["proposition"].startswith("If someone else") and a2["situations"] == ["schedules set by others"]
    assert a2["revisions"][-1]["trigger"] == "user_correction"
    assert scripted.calls[-1]["payload"]["action"] == "correct"


def test_user_correction_survives_llm_failure(pipeline, scripted, store, episode):
    a, _ = _two_candidates(store, episode)
    r = pipeline.correct(a["id"], "Not quite.")  # no scripted response -> LLMError inside
    assert r["assumption"]["status"] == "revised" and r["assumption"]["user_corrections"][0]["text"] == "Not quite."


def test_user_rejection_sticks_and_offers_alternative(pipeline, scripted, store, episode):
    a, _ = _two_candidates(store, episode)
    scripted.queue("revise_correction", {"proposition": a["proposition"], "kind": "if_then", "situations": [], "weakeners": [], "alternative_explanations": [], "prediction_new_situation": "",
                                         "reason": "The user is clearly resistant and in denial.",
                                         "propose_alternative": {"proposition": "If the activity is dull, then I avoid it.", "kind": "if_then", "supporting": [{"episode_id": episode["id"], "quotation": "every week"}], "rationale": "alt"}})
    r = pipeline.reject(a["id"], "That is not it.")
    assert r["assumption"]["status"] == "rejected"
    assert r["assumption"]["revisions"][-1]["trigger"] == "user_rejection"
    assert r["reason"] == GUARD_NOTICE
    assert r["alternative"]["status"] == "proposed" and r["alternative"]["supporting_episode_ids"] == [episode["id"]]
    # rejected candidates no longer take part in the interview
    assert all(c["id"] != a["id"] for c in pipeline.active_assumptions())


def test_user_rejection_sticks_without_llm(pipeline, scripted, store, episode):
    a, _ = _two_candidates(store, episode)
    r = pipeline.reject(a["id"])
    assert r["assumption"]["status"] == "rejected" and r["alternative"] is None


# ---------------------------------------------------------------- holdout
def test_holdout_sends_situation_only_and_compares(pipeline, scripted, store, episode):
    a, b = _two_candidates(store, episode)
    held = store.add_episode("Coordinator asked: \"a fixed slot every Saturday?\"", "There goes every Saturday.", "Irritation, tight shoulders.", "Said I'd drop in instead.", reserved=True)
    scripted.queue("holdout_predict", {"predictions": [
        {"assumption_id": a["id"], "applies": True, "predicted_thought": "Now I have to", "predicted_emotion": "irritation", "predicted_action": "postpone", "confidence": 0.7, "rationale": "fits"},
        {"assumption_id": b["id"], "applies": False, "predicted_thought": "nothing", "predicted_emotion": "neutral", "predicted_action": "agree", "confidence": 0.3, "rationale": "n/a"},
        {"assumption_id": "as-ghost", "predicted_thought": "x"}],
        "baseline": {"predicted_thought": "consider it", "predicted_emotion": "mixed", "predicted_action": "respond", "rationale": "generic"}})
    scripted.queue("holdout_compare", {"comparisons": [{"assumption_id": a["id"], "match_score": 0.9, "note": "close"}, {"assumption_id": b["id"], "match_score": 0.1, "note": "off"}, {"assumption_id": None, "match_score": 0.4, "note": "vague"}]})
    r = pipeline.holdout(held["id"])
    predict_payload = scripted.calls[-2]["payload"]
    assert set(predict_payload.keys()) == {"situation", "candidates"}
    assert "There goes every Saturday" not in str(predict_payload)
    assert scripted.calls[-1]["payload"]["recorded"]["immediate_thought"] == "There goes every Saturday."
    scores = {p["assumption_id"]: p["match_score"] for p in r["predictions"]}
    assert scores == {a["id"]: 0.9, b["id"]: 0.1}
    assert r["baseline"]["match_score"] == 0.4
    assert "does not apply" in [p for p in r["predictions"] if p["assumption_id"] == b["id"]][0]["rationale"]
    assert len(store.list_predictions(held["id"])) == 3
    with pytest.raises(ValueError):
        pipeline.holdout(episode["id"])  # not reserved


# ---------------------------------------------------------------- failures, logging, retrieval
def test_stage_failure_is_recorded_without_content(pipeline, scripted, store, episode, caplog):
    configure_logging("INFO")
    with caplog.at_level(logging.INFO, logger="avatar"), pytest.raises(LLMError):
        pipeline.extract(episode["id"])
    run = store.list_runs()[-1]
    assert run["ok"] is False and run["error_type"] == "LLMError"
    assert "Tuesday" not in caplog.text and "jaw" not in caplog.text


def test_logging_never_contains_episode_text(mock_pipeline, store, caplog):
    configure_logging("DEBUG")
    seed(store)
    with caplog.at_level(logging.DEBUG, logger="avatar"):
        for ep in store.list_episodes(include_reserved=False)[:4]:
            mock_pipeline.process_episode(ep["id"])
        q = mock_pipeline.next_question()
        mock_pipeline.answer(q["turn"]["id"], synthetic_answer(q["turn"]["question"]))
    text = caplog.text
    for ep in store.list_episodes():
        for field in ("situation", "immediate_thought", "emotion_body"):
            assert ep[field][:25] not in text
    assert "wrecked" not in text and "side project" not in text
    assert "pipeline.stage_ok" in text


def test_bm25_ranking_prefers_overlap():
    docs = {"a": "weekly run commitment schedule", "b": "a walk in the park with a heron", "c": "sign up for a weekly class schedule"}
    ranked = [d for d, _ in bm25_rank("weekly schedule", docs)]
    assert ranked[:2] == ["a", "c"] or ranked[:2] == ["c", "a"]
    assert "b" not in ranked


# ---------------------------------------------------------------- end-to-end with the rule-based mock
def test_mock_end_to_end_reaches_a_supported_revisable_assumption(mock_pipeline, store):
    seed(store)
    for ep in store.list_episodes(include_reserved=False):
        mock_pipeline.process_episode(ep["id"])
    kinds = {a["kind"] for a in store.list_assumptions()}
    assert "situational" in kinds  # situational explanation was among the candidates
    for _ in range(6):
        q = mock_pipeline.next_question()
        if q["stop"]:
            break
        mock_pipeline.answer(q["turn"]["id"], synthetic_answer(q["turn"]["question"]))
    active = [a for a in store.list_assumptions() if a["status"] != "rejected"]
    best = max(active, key=lambda a: len(a["supporting_episode_ids"]))
    assert len(best["supporting_episode_ids"]) >= 5
    assert all(e["verified"] for e in best["supporting"])
    assert any(a["status"] == "rejected" for a in store.list_assumptions())
    r = mock_pipeline.correct(best["id"], "If someone else fixes a recurring schedule for me, then I lose control over my time.")
    assert r["assumption"]["status"] == "revised" and r["assumption"]["proposition"].startswith("If someone else")
    held = [e for e in store.list_episodes() if e["reserved"]]
    results = {e["id"]: mock_pipeline.holdout(e["id"]) for e in held}
    garden = results["ep-demo11"]
    best_pred = [p for p in garden["predictions"] if p["assumption_id"] == best["id"]][0]
    assert best_pred["match_score"] > garden["baseline"]["match_score"]
