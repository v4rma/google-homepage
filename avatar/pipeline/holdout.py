"""Holdout / prediction test. The model sees only the situation of a reserved episode; each candidate assumption
predicts the response; the comparison with what was recorded is shown side by side."""

from __future__ import annotations

from ..textutil import new_id


def run_holdout(pipeline, episode_id: str) -> dict:
    store = pipeline.store
    ep = store.get_episode(episode_id)
    if not ep:
        raise KeyError("episode not found")
    if not ep["reserved"]:
        raise ValueError("only reserved (holdout) episodes can be used for prediction tests")
    candidates = pipeline.active_assumptions()
    if not candidates:
        raise ValueError("no active candidate assumptions to test")
    predict_payload = {
        "situation": ep["situation"],
        "candidates": [{"id": a["id"], "proposition": a["proposition"], "kind": a["kind"], "situations": a["situations"]} for a in candidates],
    }
    predicted = pipeline.call("holdout_predict", predict_payload, episode_id=episode_id)
    valid = {a["id"] for a in candidates}
    preds = [p for p in (predicted.get("predictions") or []) if isinstance(p, dict) and p.get("assumption_id") in valid]
    baseline = predicted.get("baseline") if isinstance(predicted.get("baseline"), dict) else None
    rows = list(preds)
    if baseline:
        rows.append({**baseline, "assumption_id": None, "applies": None, "confidence": None})
    compare_payload = {
        "recorded": {"immediate_thought": ep["immediate_thought"], "emotion_body": ep["emotion_body"], "action_outcome": ep["action_outcome"]},
        "predictions": [{k: r.get(k) for k in ("assumption_id", "predicted_thought", "predicted_emotion", "predicted_action")} for r in rows],
    }
    compared = pipeline.call("holdout_compare", compare_payload, episode_id=episode_id)
    scores = {}
    for c in compared.get("comparisons") or []:
        if isinstance(c, dict):
            scores[c.get("assumption_id")] = c
    batch = new_id("hb")
    saved = []
    for r in rows:
        c = scores.get(r.get("assumption_id"), {})
        try:
            score = float(c.get("match_score")) if c.get("match_score") is not None else None
        except (TypeError, ValueError):
            score = None
        saved.append(store.add_prediction(
            episode_id, batch, r.get("assumption_id"),
            predicted_thought=str(r.get("predicted_thought") or ""), predicted_emotion=str(r.get("predicted_emotion") or ""),
            predicted_action=str(r.get("predicted_action") or ""), confidence=r.get("confidence"),
            rationale=str(r.get("rationale") or "") + ("" if r.get("applies") in (None, True) else " (assumption says it does not apply here)"),
            match_score=score, comparison=str(c.get("note") or ""),
        ))
    return {
        "batch_id": batch,
        "episode": ep,
        "predictions": [p for p in saved if p["assumption_id"]],
        "baseline": next((p for p in saved if not p["assumption_id"]), None),
        "recorded": compare_payload["recorded"],
    }
