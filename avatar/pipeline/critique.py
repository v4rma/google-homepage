"""Stage 3: Critique. Contradictions, unsupported leaps, alternatives and resolving questions for one candidate."""

from __future__ import annotations


def run_critique(pipeline, assumption_id: str) -> dict:
    store = pipeline.store
    a = store.get_assumption(assumption_id)
    if not a:
        raise KeyError("assumption not found")
    others = [o for o in pipeline.active_assumptions() if o["id"] != assumption_id]
    episodes = pipeline.episodes_for_assumptions([a] + others)
    payload = {
        "assumption": pipeline.assumption_record(a),
        "other_candidates": [pipeline.assumption_record(o, brief=True) for o in others],
        "episodes": [pipeline.episode_record(e) for e in episodes],
    }
    result = pipeline.call("critique", payload, episode_id=a.get("origin_episode_id"))
    allowed = {e["id"] for e in episodes}
    pipeline.add_evidence_items(assumption_id, result.get("contradictions") or [], "contradicting", allowed)
    store.merge_lists(assumption_id, "alternative_explanations", [str(x) for x in result.get("alternative_explanations") or []], trigger="critique")
    store.merge_lists(assumption_id, "weakeners", [str(x) for x in result.get("weakeners") or []], trigger="critique")
    valid_ids = {assumption_id} | {o["id"] for o in others}
    questions = []
    for q in result.get("resolving_questions") or []:
        if isinstance(q, dict) and q.get("question"):
            questions.append({
                "question": str(q["question"]),
                "episode_id": q.get("episode_id") if q.get("episode_id") in allowed else None,
                "discriminates_between": [i for i in (q.get("discriminates_between") or []) if i in valid_ids],
            })
    critique = {
        "contradictions": result.get("contradictions") or [],
        "unsupported_leaps": [str(x) for x in result.get("unsupported_leaps") or []],
        "alternative_explanations": [str(x) for x in result.get("alternative_explanations") or []],
        "weakeners": [str(x) for x in result.get("weakeners") or []],
        "resolving_questions": questions,
        "verdict": str(result.get("verdict") or "needs_examination"),
    }
    store.add_interpretation("critique", critique, assumption_id=assumption_id, episode_id=a.get("origin_episode_id"))
    if a["status"] == "proposed":
        store.set_status(assumption_id, "under examination", trigger="critique", reason="critique produced; awaiting interview")
    return {"assumption": store.get_assumption(assumption_id), "critique": critique}
