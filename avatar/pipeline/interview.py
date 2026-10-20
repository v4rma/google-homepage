"""Stage 4: Interview and update. One question at a time, grounded in an episode, designed to discriminate between
candidates; the answer (or the user's own correction/rejection) drives revision."""

from __future__ import annotations

from ..llm.base import LLMError
from ..logging_utils import log_error
from ..textutil import looks_like_dont_know


def _suggested_questions(pipeline, candidates: list[dict]) -> list[dict]:
    out = []
    for a in candidates:
        crits = [i for i in pipeline.store.list_interpretations(assumption_id=a["id"]) if i["stage"] == "critique"]
        if crits:
            out.extend(crits[-1]["content"].get("resolving_questions", [])[:2])
    return out[:6]


def next_question(pipeline, focus_episode_id: str | None = None) -> dict:
    store = pipeline.store
    pending = store.pending_turn()
    if pending:
        return {"turn": pending, "stop": False, "reason": "A question is already waiting for an answer."}
    candidates = pipeline.active_assumptions()
    if not candidates:
        return {"turn": None, "stop": True, "reason": "There are no candidate assumptions to examine yet."}
    episodes = pipeline.episodes_for_assumptions(candidates, extra_ids=[focus_episode_id] if focus_episode_id else [])
    if not episodes:
        episodes = store.list_episodes(include_reserved=False)[:6]
    answered = [t for t in store.list_turns() if t.get("answer")]
    payload = {
        "candidates": [pipeline.assumption_record(a) for a in candidates],
        "episodes": [pipeline.episode_record(e) for e in episodes],
        "previous_turns": [{"episode_id": t["episode_id"], "question": t["question"], "answer": t["answer"], "dont_know": t["dont_know"]} for t in answered],
        "suggested_questions": _suggested_questions(pipeline, candidates),
        "focus_episode_id": focus_episode_id,
    }
    result = pipeline.call("interview_question", payload, episode_id=focus_episode_id)
    if result.get("stop") or not str(result.get("question") or "").strip():
        store.add_interpretation("interview_question", {"stop": True, "stop_reason": result.get("stop_reason", "")}, episode_id=focus_episode_id)
        return {"turn": None, "stop": True, "reason": str(result.get("stop_reason") or "The model judged that further questions would be speculation.")}
    allowed = {e["id"] for e in episodes}
    ep_id = result.get("episode_id") if result.get("episode_id") in allowed else (focus_episode_id if focus_episode_id in allowed else episodes[0]["id"])
    valid = {a["id"] for a in candidates}
    disc = [i for i in (result.get("discriminates_between") or []) if i in valid]
    turn = store.add_turn(ep_id, str(result["question"]).strip(), discriminates_between=disc, rationale=pipeline.guard_reason(str(result.get("rationale") or "")))
    return {"turn": turn, "stop": False, "reason": ""}


def answer_question(pipeline, turn_id: str, answer: str) -> dict:
    store = pipeline.store
    turn = store.get_turn(turn_id)
    if not turn:
        raise KeyError("turn not found")
    if turn.get("answer"):
        raise ValueError("this question has already been answered")
    answer = (answer or "").strip()
    dont_know = looks_like_dont_know(answer)
    turn = store.answer_turn(turn_id, answer, dont_know)
    store.add_observation(turn["episode_id"], "answer", answer or "(no answer)", source="interview_answer", verified=True, turn_id=turn_id)
    if dont_know:
        store.add_interpretation("interview_update", {"dont_know": True, "updates": []}, episode_id=turn["episode_id"])
        return {"turn": turn, "updates": [], "new_assumptions": [], "stop": False, "note": "Recorded. 'I don't know' is a valid answer; nothing was changed."}

    candidates = pipeline.active_assumptions()
    ep = store.get_episode(turn["episode_id"])
    payload = {
        "turn": {"episode_id": turn["episode_id"], "question": turn["question"], "answer": answer, "dont_know": False, "discriminates_between": turn["discriminates_between"]},
        "candidates": [pipeline.assumption_record(a) for a in candidates],
        "episode": pipeline.episode_record(ep) if ep else None,
    }
    result = pipeline.call("interview_update", payload, episode_id=turn["episode_id"])
    allowed = {turn["episode_id"]} | {e["id"] for e in pipeline.episodes_for_assumptions(candidates)}
    valid = {a["id"]: a for a in candidates}
    applied = []
    for upd in result.get("updates") or []:
        if not isinstance(upd, dict) or upd.get("assumption_id") not in valid:
            continue
        aid, action = upd["assumption_id"], str(upd.get("action") or "keep")
        reason = pipeline.guard_reason(str(upd.get("reason") or ""))
        current = valid[aid]
        if action == "strengthen":
            pipeline.add_evidence_items(aid, upd.get("supporting") or [], "supporting", allowed, turn_id=turn_id)
            if current["status"] == "proposed":
                store.set_status(aid, "under examination", trigger="interview_answer", reason=reason)
        elif action == "revise":
            changes = {}
            if str(upd.get("new_proposition") or "").strip():
                changes["proposition"] = str(upd["new_proposition"]).strip()
            if upd.get("new_situations"):
                changes["situations"] = [str(s) for s in upd["new_situations"]]
            if upd.get("new_weakeners"):
                changes["weakeners"] = list(current["weakeners"]) + [str(w) for w in upd["new_weakeners"] if str(w) not in current["weakeners"]]
            changes["status"] = "revised"
            store.revise_assumption(aid, changes, trigger="interview_answer", reason=reason)
            pipeline.add_evidence_items(aid, upd.get("contradicting") or [], "contradicting", allowed, turn_id=turn_id)
            pipeline.add_evidence_items(aid, upd.get("supporting") or [], "supporting", allowed, turn_id=turn_id)
        elif action == "reject":
            pipeline.add_evidence_items(aid, upd.get("contradicting") or [], "contradicting", allowed, turn_id=turn_id)
            store.set_status(aid, "rejected", trigger="interview_answer", reason=reason)
        applied.append({"assumption_id": aid, "action": action, "reason": reason})
    new_assumptions = []
    for cand in result.get("new_candidates") or []:
        a = pipeline.ingest_candidate(cand, origin_episode_id=turn["episode_id"], allowed_episode_ids=allowed, stage="interview_update")
        if a:
            new_assumptions.append(a)
    stop = bool(result.get("stop_interview"))
    store.add_interpretation(
        "interview_update",
        {"updates": applied, "new_assumption_ids": [a["id"] for a in new_assumptions], "stop_interview": stop, "stop_reason": result.get("stop_reason", "")},
        episode_id=turn["episode_id"],
    )
    return {"turn": turn, "updates": applied, "new_assumptions": new_assumptions, "stop": stop, "note": str(result.get("stop_reason") or "")}


def _revision_payload(pipeline, a: dict, action: str, text: str) -> dict:
    episodes = pipeline.episodes_for_assumptions([a])
    return {"assumption": pipeline.assumption_record(a), "action": action, "correction": text, "episodes": [pipeline.episode_record(e) for e in episodes]}


def apply_user_correction(pipeline, assumption_id: str, text: str) -> dict:
    """The user's correction is recorded first and is authoritative; the model is then asked to re-word to match."""
    store = pipeline.store
    text = (text or "").strip()
    if not text:
        raise ValueError("correction text is required")
    a = store.add_correction(assumption_id, text)
    if not a:
        raise KeyError("assumption not found")
    try:
        result = pipeline.call("revise_correction", _revision_payload(pipeline, a, "correct", text), episode_id=a.get("origin_episode_id"))
    except LLMError as exc:
        log_error("correction.llm_failed_recorded_anyway", exc, assumption_id=assumption_id)
        result = {}
    reason = pipeline.guard_reason(str(result.get("reason") or "User correction recorded."))
    changes = {
        "proposition": str(result.get("proposition") or a["proposition"]).strip(),
        "kind": result.get("kind") or a["kind"],
        "situations": [str(s) for s in result.get("situations") or a["situations"]],
        "weakeners": [str(s) for s in result.get("weakeners") or a["weakeners"]],
        "alternative_explanations": [str(s) for s in result.get("alternative_explanations") or a["alternative_explanations"]],
        "prediction_new_situation": str(result.get("prediction_new_situation") or a["prediction_new_situation"]),
        "status": "revised",
    }
    updated = store.revise_assumption(assumption_id, changes, trigger="user_correction", reason=reason)
    store.add_interpretation("revise_correction", {"action": "correct", "reason": reason}, assumption_id=assumption_id)
    return {"assumption": updated, "reason": reason}


def apply_user_rejection(pipeline, assumption_id: str, text: str = "") -> dict:
    """Rejection always sticks, regardless of what the model says; the model may only offer an alternative."""
    store = pipeline.store
    a = store.get_assumption(assumption_id)
    if not a:
        raise KeyError("assumption not found")
    if text.strip():
        a = store.add_correction(assumption_id, text.strip())
    rejected = store.set_status(assumption_id, "rejected", trigger="user_rejection", reason="The user rejected this interpretation.")
    alternative = None
    reason = "Rejected at the user's request."
    try:
        result = pipeline.call("revise_correction", _revision_payload(pipeline, a, "reject", text), episode_id=a.get("origin_episode_id"))
        reason = pipeline.guard_reason(str(result.get("reason") or reason))
        cand = result.get("propose_alternative")
        if isinstance(cand, dict) and cand.get("proposition"):
            allowed = {e["id"] for e in pipeline.episodes_for_assumptions([a])}
            cand.pop("relates_to_existing", None)
            alternative = pipeline.ingest_candidate(cand, origin_episode_id=a.get("origin_episode_id"), allowed_episode_ids=allowed, stage="revise_correction")
    except LLMError as exc:
        log_error("rejection.llm_failed_rejection_kept", exc, assumption_id=assumption_id)
    store.add_interpretation("revise_correction", {"action": "reject", "reason": reason, "alternative_id": alternative["id"] if alternative else None}, assumption_id=assumption_id)
    return {"assumption": rejected, "alternative": alternative, "reason": reason}
