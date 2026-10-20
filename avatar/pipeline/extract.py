"""Stage 1: Extract. Observations only; quotations are verified against the user's original account."""

from __future__ import annotations

from ..storage import Store
from ..textutil import verify_quotation

OBSERVATION_KINDS = ("events", "quotations", "expectations", "emotions", "actions", "missing")


def run_extract(pipeline, episode_id: str) -> list[dict]:
    store: Store = pipeline.store
    ep = store.get_episode(episode_id)
    if not ep:
        raise KeyError("episode not found")
    result = pipeline.call("extract", {"episode": pipeline.episode_record(ep, include_answers=False)}, episode_id=episode_id)
    store.clear_observations(episode_id, source="extract")
    sources = [ep[f] for f in ("situation", "immediate_thought", "emotion_body", "action_outcome", "comparable_occasion") if ep.get(f)]
    out = []
    for kind in OBSERVATION_KINDS:
        for item in result.get(kind) or []:
            content = str(item).strip()
            if not content:
                continue
            # "missing" entries describe absence; everything else is checked for being verbatim user text.
            verified = kind == "missing" or verify_quotation(content, sources)
            out.append(store.add_observation(episode_id, kind.rstrip("s") if kind != "missing" else "missing", content, source="extract", verified=verified))
    return out
