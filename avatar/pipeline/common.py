"""Pipeline core: stage dispatch, record builders shared by stages, candidate ingestion, and the guard against
re-interpreting the user's corrections."""

from __future__ import annotations

import re
from typing import Any

from ..config import Settings
from ..llm.base import LLMProvider
from ..logging_utils import log_error, log_event
from ..prompts import build_prompts
from ..retrieval import retrieve_related_episodes
from ..storage import ACTIVE_STATUSES, Store
from ..textutil import strip_quotes

# If the model describes the user's correction/rejection in these terms, the explanation is discarded.
RESISTANCE_TERMS = re.compile(r"\b(resist\w*|defensiv\w*|denial|in denial|avoid\w* the (question|issue)|lack of insight|unwilling to (see|accept))\b", re.I)
GUARD_NOTICE = "[model explanation removed: it characterised the user's response instead of revising the assumption]"


class Pipeline:
    def __init__(self, store: Store, provider: LLMProvider, settings: Settings | None = None):
        self.store = store
        self.provider = provider
        self.settings = settings or Settings(provider=getattr(provider, "name", "?"), api_key=None, model="", base_url="", db_path=":memory:", log_level="INFO")

    # ------------------------------------------------------------------ dispatch
    def call(self, stage: str, payload: dict[str, Any], episode_id: str | None = None) -> dict[str, Any]:
        run_id = self.store.start_run(stage, episode_id)
        system_prompt, user_prompt = build_prompts(stage, payload)
        try:
            result = self.provider.run(stage, system_prompt, user_prompt, payload)
        except Exception as exc:
            self.store.finish_run(run_id, ok=False, error_type=type(exc).__name__)
            log_error("pipeline.stage_failed", exc, stage=stage, episode_id=episode_id, run=run_id)
            raise
        self.store.finish_run(run_id, ok=True)
        log_event("pipeline.stage_ok", stage=stage, episode_id=episode_id, run=run_id, prompt_chars=len(user_prompt))
        return result if isinstance(result, dict) else {}

    # ------------------------------------------------------------------ record builders (what the model sees)
    def episode_record(self, ep: dict, include_answers: bool = True, situation_only: bool = False) -> dict:
        rec = {"id": ep["id"], "title": ep.get("title", ""), "situation": ep["situation"]}
        if situation_only:
            return rec
        rec.update(
            immediate_thought=ep["immediate_thought"],
            emotion_body=ep["emotion_body"],
            action_outcome=ep["action_outcome"],
            comparable_occasion=ep.get("comparable_occasion", ""),
        )
        obs = [o for o in self.store.list_observations(ep["id"]) if o["source"] == "extract"]
        if obs:
            rec["observations"] = [{"kind": o["kind"], "content": o["content"], "verified": o["verified"]} for o in obs]
        if include_answers:
            answers = [{"question": t["question"], "answer": t["answer"], "dont_know": t["dont_know"]} for t in self.store.list_turns(ep["id"]) if t.get("answer")]
            if answers:
                rec["interview_answers"] = answers
        return rec

    def assumption_record(self, a: dict, brief: bool = False) -> dict:
        rec = {"id": a["id"], "proposition": a["proposition"], "kind": a["kind"], "status": a["status"], "situations": a["situations"]}
        rec["user_corrections"] = [c["text"] for c in a.get("user_corrections", [])]
        rec["supporting_episode_ids"] = a.get("supporting_episode_ids", [])
        if brief:
            return rec
        rec.update(
            alternative_explanations=a["alternative_explanations"],
            prediction_new_situation=a["prediction_new_situation"],
            weakeners=a["weakeners"],
            supporting=[{"episode_id": e["episode_id"], "quotation": e["quotation"], "verified": e["verified"]} for e in a.get("supporting", [])],
            contradicting=[{"episode_id": e["episode_id"], "quotation": e["quotation"], "note": e["note"]} for e in a.get("contradicting", [])],
        )
        return rec

    def active_assumptions(self) -> list[dict]:
        return self.store.list_assumptions(statuses=ACTIVE_STATUSES)

    def related_episodes(self, episode_id: str) -> list[dict]:
        return retrieve_related_episodes(self.store, episode_id, k=self.settings.retrieval_k)

    def episodes_for_assumptions(self, assumptions: list[dict], extra_ids: list[str] = ()) -> list[dict]:
        ids: list[str] = []
        for a in assumptions:
            for e in a.get("supporting", []) + a.get("contradicting", []):
                ids.append(e["episode_id"])
            if a.get("origin_episode_id"):
                ids.append(a["origin_episode_id"])
        ids.extend(extra_ids)
        out, seen = [], set()
        for i in ids:
            if i in seen:
                continue
            seen.add(i)
            ep = self.store.get_episode(i)
            if ep and not ep["reserved"]:
                out.append(ep)
        return out

    # ------------------------------------------------------------------ guards and ingestion
    @staticmethod
    def guard_reason(text: str | None) -> str:
        text = text or ""
        if RESISTANCE_TERMS.search(text):
            log_event("guard.resistance_language_removed")
            return GUARD_NOTICE
        return text

    def add_evidence_items(self, assumption_id: str, items: list[dict], role: str, allowed_episode_ids: set[str], turn_id: str | None = None) -> list[dict]:
        added = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            ep_id, quote = item.get("episode_id"), strip_quotes(str(item.get("quotation") or ""))
            if ep_id not in allowed_episode_ids or not quote:
                log_event("evidence.skipped", assumption_id=assumption_id, reason="unknown_episode_or_empty")
                continue
            ev = self.store.add_evidence(assumption_id, ep_id, quote, role, note=str(item.get("note") or ""), turn_id=turn_id)
            if ev:
                added.append(ev)
                if not ev["verified"]:
                    log_event("evidence.unverified_quotation", assumption_id=assumption_id, episode_id=ep_id)
        return added

    def ingest_candidate(self, cand: dict, origin_episode_id: str | None, allowed_episode_ids: set[str], stage: str) -> dict | None:
        """Create a new assumption (or attach to an existing one) from a model-proposed candidate."""
        if not isinstance(cand, dict) or not str(cand.get("proposition") or "").strip():
            return None
        existing_id = cand.get("relates_to_existing")
        existing = self.store.get_assumption(existing_id) if existing_id else None
        if existing and existing["status"] != "rejected":
            assumption = existing
            self.store.merge_lists(assumption["id"], "situations", cand.get("situations") or [], trigger=stage, reason="merged from a new episode")
            self.store.merge_lists(assumption["id"], "alternative_explanations", cand.get("alternative_explanations") or [], trigger=stage)
            self.store.merge_lists(assumption["id"], "weakeners", cand.get("weakeners") or [], trigger=stage)
        else:
            assumption = self.store.add_assumption(
                proposition=str(cand["proposition"]).strip(),
                kind=str(cand.get("kind") or "if_then"),
                situations=[str(s) for s in cand.get("situations") or []],
                alternative_explanations=[str(s) for s in cand.get("alternative_explanations") or []],
                prediction_new_situation=str(cand.get("prediction_new_situation") or ""),
                weakeners=[str(s) for s in cand.get("weakeners") or []],
                origin_episode_id=origin_episode_id,
            )
        self.add_evidence_items(assumption["id"], cand.get("supporting") or [], "supporting", allowed_episode_ids)
        self.add_evidence_items(assumption["id"], cand.get("contradicting") or [], "contradicting", allowed_episode_ids)
        self.store.add_interpretation(stage, {"rationale": cand.get("rationale", ""), "candidate": {k: v for k, v in cand.items() if k != "rationale"}}, assumption_id=assumption["id"], episode_id=origin_episode_id)
        return self.store.get_assumption(assumption["id"])

    # ------------------------------------------------------------------ stage entry points
    def extract(self, episode_id: str) -> list[dict]:
        from .extract import run_extract

        return run_extract(self, episode_id)

    def hypothesize(self, episode_id: str) -> dict:
        from .hypothesize import run_hypothesize

        return run_hypothesize(self, episode_id)

    def critique(self, assumption_id: str) -> dict:
        from .critique import run_critique

        return run_critique(self, assumption_id)

    def next_question(self, focus_episode_id: str | None = None) -> dict:
        from .interview import next_question

        return next_question(self, focus_episode_id)

    def answer(self, turn_id: str, answer: str) -> dict:
        from .interview import answer_question

        return answer_question(self, turn_id, answer)

    def correct(self, assumption_id: str, text: str) -> dict:
        from .interview import apply_user_correction

        return apply_user_correction(self, assumption_id, text)

    def reject(self, assumption_id: str, text: str = "") -> dict:
        from .interview import apply_user_rejection

        return apply_user_rejection(self, assumption_id, text)

    def holdout(self, episode_id: str) -> dict:
        from .holdout import run_holdout

        return run_holdout(self, episode_id)

    def process_episode(self, episode_id: str, critique_new: bool = True) -> dict:
        """Extract -> Hypothesize -> Critique each touched candidate. Convenience for the UI and scripts."""
        observations = self.extract(episode_id)
        hyp = self.hypothesize(episode_id)
        critiques = []
        if critique_new:
            for a in hyp["assumptions"]:
                if a["status"] != "rejected":
                    critiques.append(self.critique(a["id"]))
        return {"observations": observations, "hypothesize": hyp, "critiques": critiques}
