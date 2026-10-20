"""Stage 2: Hypothesize. Up to N candidate explanations, including situational ones, grounded in retrieved episodes."""

from __future__ import annotations


def run_hypothesize(pipeline, episode_id: str) -> dict:
    store = pipeline.store
    ep = store.get_episode(episode_id)
    if not ep:
        raise KeyError("episode not found")
    if ep["reserved"]:
        raise ValueError("reserved (holdout) episodes are not analysed; un-reserve it first")
    related = pipeline.related_episodes(episode_id)
    existing = store.list_assumptions()
    payload = {
        "episode": pipeline.episode_record(ep),
        "related_episodes": [pipeline.episode_record(r) for r in related],
        "existing_assumptions": [pipeline.assumption_record(a, brief=True) for a in existing],
        "max_candidates": pipeline.settings.max_candidates,
    }
    result = pipeline.call("hypothesize", payload, episode_id=episode_id)
    allowed = {ep["id"]} | {r["id"] for r in related}
    assumptions = []
    for cand in (result.get("candidates") or [])[: pipeline.settings.max_candidates]:
        a = pipeline.ingest_candidate(cand, origin_episode_id=episode_id, allowed_episode_ids=allowed, stage="hypothesize")
        if a:
            assumptions.append(a)
    store.add_interpretation(
        "hypothesize",
        {"cannot_tell": bool(result.get("cannot_tell")), "note": result.get("note", ""), "candidate_count": len(assumptions), "related_episode_ids": sorted(allowed - {ep["id"]})},
        episode_id=episode_id,
    )
    return {"assumptions": assumptions, "cannot_tell": bool(result.get("cannot_tell")), "note": result.get("note", ""), "related_episode_ids": [r["id"] for r in related]}
