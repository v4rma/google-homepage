"""Rule-based stand-in for the LLM.

This is NOT a language model. It is a deterministic, keyword-driven imitation of the stage outputs so that the
pipeline, storage, UI and milestone script can be exercised end to end without a provider key. It only
"understands" three themes (autonomy, evaluation, fatigue) and will produce shallow results on anything else.
Its outputs are shaped exactly like the JSON the real prompts request, so swapping in the real provider changes
nothing else.
"""

from __future__ import annotations

import re
from typing import Any

from ..textutil import normalize, quoted_spans, sentences, tokenize

NEGATIVE = ("irritat", "annoy", "resent", "trapped", "tight", "tense", "nervous", "anxious", "dread", "worried", "angry", "heavy", "stuck", "jaw", "clench", "sink")
POSITIVE = ("calm", "fine", "glad", "enjoy", "relaxed", "happy", "easy", "pleased", "light", "didn't mind", "did not mind", "okay", "no problem", "warm", "content", "curious")

THEMES: dict[str, dict[str, Any]] = {
    "autonomy": {
        "keywords": ("commit", "every day", "every single day", "daily", "weekly", "every week", "every sunday", "every saturday", "monthly",
                     "schedule", "sign up", "signed up", "have to", "must", "promise", "oblig", "routine", "recurring", "regular", "pledge",
                     "lose control", "my time", "rota", "fixed slot", "contract", "twelve months", "12-month", "stuck", "trapped", "same slot"),
        "proposition": "If I agree to a recurring commitment, then I will lose control over my time.",
        "kind": "if_then",
        "situations": ["Being asked to commit to a regular or recurring activity", "Signing up for something with a fixed schedule set by someone else"],
        "alternatives": ["Tiredness or low energy on that particular day", "Dislike of the specific activity rather than of commitment itself", "The request itself was unreasonable"],
        "prediction": "When asked to commit to a weekly activity, the first thought will be about losing control of time, followed by irritation and postponing the answer.",
        "weakeners": ["An occasion where a recurring commitment was accepted calmly and kept without irritation", "Irritation that appears equally for one-off requests"],
        "predicted": ("Now I have to do this.", "Irritation; tightness in the chest or jaw", "Postpone the answer or decline"),
        "prop_markers": ("commit", "recurring", "control", "freedom", "schedule", "my time"),
    },
    "evaluation": {
        "keywords": ("judg", "evaluat", "people watched", "watched me", "watching me", "audience", "present", "review", "grade", "feedback", "perform", "critic", "what they think", "in front of"),
        "proposition": "If my performance is visible to others, then I will be judged as inadequate.",
        "kind": "if_then",
        "situations": ["Doing something while others watch, review or grade it"],
        "alternatives": ["The specific evaluator was harsh", "Lack of preparation on that occasion"],
        "prediction": "When work will be reviewed by others, the first thought will be about being found lacking, with nervousness and either over-preparation or avoidance.",
        "weakeners": ["Being evaluated on freely chosen work without distress"],
        "predicted": ("They will see I am not good enough.", "Nervousness; racing heart", "Over-prepare or avoid the situation"),
        "prop_markers": ("judg", "visible", "watch", "inadequate", "evaluat", "perform"),
    },
    "fatigue": {
        "keywords": ("tired", "exhaust", "slept", "sleep", "night shift", "drained", "worn out", "no energy"),
        "proposition": "The irritation on that occasion is explained by exhaustion at the time rather than by a general belief.",
        "kind": "situational",
        "situations": ["Requests that arrive after poor sleep or a long shift"],
        "alternatives": ["A general belief about commitment that exhaustion merely amplified"],
        "prediction": "The same request on a rested day will produce a milder reaction.",
        "weakeners": ["The same irritation appearing on well-rested days"],
        "predicted": ("I just cannot deal with this right now.", "Flat, drained", "Put it off until later"),
        "prop_markers": ("exhaust", "tired", "situational", "sleep"),
    },
}

QUESTIONS = {
    ("autonomy", "evaluation"): "Think of a time when you chose an activity yourself but other people watched or reviewed it. What went through your mind then, compared with {ep}?",
    ("autonomy", "fatigue"): "On the occasion in {ep}, how rested were you? Has the same kind of request come on a day you felt rested, and what happened then?",
    ("evaluation", "fatigue"): "In {ep}, was it the thought of being watched or the tiredness that came first? What would have changed if you had slept well?",
}


def _theme_score(text: str, theme: str) -> int:
    t = normalize(text)
    return sum(t.count(k) for k in THEMES[theme]["keywords"])


def _main_text(ep: dict) -> str:
    """Account without the comparable occasion, so a contrasting occasion does not count as the episode itself."""
    return "\n".join(str(ep.get(f) or "") for f in ("situation", "immediate_thought", "emotion_body", "action_outcome"))


def _has(text: str, markers: tuple[str, ...]) -> bool:
    t = normalize(text)
    return any(m in t for m in markers)


def _episode_text(ep: dict) -> str:
    return "\n".join(str(ep.get(f) or "") for f in ("situation", "immediate_thought", "emotion_body", "action_outcome", "comparable_occasion"))


def _quote_from(ep: dict) -> str:
    spans = quoted_spans(_episode_text(ep))
    if spans:
        return spans[0]
    return str(ep.get("immediate_thought") or ep.get("situation") or "")[:120]


def _theme_of_proposition(prop: str) -> str | None:
    for name, th in THEMES.items():
        if _has(prop, th["prop_markers"]):
            return name
    return None


def _label(ep: dict | None) -> str:
    return f"the episode {ep.get('id')}" if ep else "that episode"


class MockProvider:
    name = "mock"

    def run(self, stage: str, system_prompt: str, user_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_{stage}", None)
        if handler is None:
            raise ValueError(f"mock provider has no handler for stage {stage}")
        return handler(payload)

    # ------------------------------------------------------------------ extract
    def _extract(self, p: dict) -> dict:
        ep = p["episode"]
        missing = []
        if not ep.get("comparable_occasion"):
            missing.append("No comparable occasion that went differently was given.")
        if not quoted_spans(_episode_text(ep)):
            missing.append("The words the other person actually used are not quoted.")
        if len(tokenize(ep.get("emotion_body", ""))) < 2:
            missing.append("Bodily sensation is not described.")
        return {
            "events": sentences(ep.get("situation", ""))[:4],
            "quotations": quoted_spans(_episode_text(ep)),
            "expectations": [ep["immediate_thought"]] if ep.get("immediate_thought") else [],
            "emotions": sentences(ep.get("emotion_body", ""))[:3],
            "actions": sentences(ep.get("action_outcome", ""))[:3],
            "missing": missing,
        }

    # ------------------------------------------------------------------ hypothesize
    def _hypothesize(self, p: dict) -> dict:
        ep = p["episode"]
        related = p.get("related_episodes", [])
        existing = p.get("existing_assumptions", [])
        scores = {name: _theme_score(_main_text(ep), name) * 2 + sum(_theme_score(_main_text(r), name) for r in related) for name in THEMES}
        ranked = [n for n, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0]
        if all(_theme_score(_main_text(ep), t) == 0 for t in THEMES):
            return {"candidates": [], "cannot_tell": True, "note": "Nothing in this account points to a recurring assumption; it reads as an ordinary episode."}
        candidates = []
        for theme in ranked[: p.get("max_candidates", 3)]:
            th = THEMES[theme]
            supporting, contradicting = [], []
            for e in [ep] + related:
                txt = _main_text(e)
                if _theme_score(txt, theme) == 0:
                    continue
                if _has(txt, POSITIVE) and not _has(e.get("immediate_thought", "") + e.get("emotion_body", ""), NEGATIVE):
                    contradicting.append({"episode_id": e["id"], "quotation": _quote_from(e), "note": "Same kind of situation, but the reaction was calm."})
                elif _has(txt, NEGATIVE):
                    supporting.append({"episode_id": e["id"], "quotation": _quote_from(e)})
            relates = next((a["id"] for a in existing if _theme_of_proposition(a["proposition"]) == theme and a.get("status") != "rejected"), None)
            candidates.append({
                "proposition": th["proposition"],
                "kind": th["kind"],
                "situations": list(th["situations"]),
                "supporting": supporting,
                "contradicting": contradicting,
                "alternative_explanations": list(th["alternatives"]),
                "prediction_new_situation": th["prediction"],
                "weakeners": list(th["weakeners"]),
                "relates_to_existing": relates,
                "rationale": f"Keyword pattern '{theme}' recurs across {len(supporting)} episode(s) with a negative reaction.",
            })
        return {"candidates": candidates, "cannot_tell": not candidates, "note": ""}

    # ------------------------------------------------------------------ critique
    def _critique(self, p: dict) -> dict:
        a = p["assumption"]
        theme = _theme_of_proposition(a["proposition"])
        others = [o for o in p.get("other_candidates", []) if o["id"] != a["id"]]
        episodes = p.get("episodes", [])
        contradictions = []
        for e in episodes:
            txt = _main_text(e)
            if theme and _theme_score(txt, theme) and _has(txt, POSITIVE) and not _has(e.get("immediate_thought", "") + e.get("emotion_body", ""), NEGATIVE):
                contradictions.append({"episode_id": e["id"], "quotation": _quote_from(e), "note": "A comparable situation without the predicted reaction."})
        n_support = len({s["episode_id"] for s in a.get("supporting", []) if s.get("verified")})
        leaps = []
        if n_support < 3:
            leaps.append(f"Only {n_support} verified supporting episode(s); the pattern may not be recurring.")
        if contradictions:
            leaps.append("The proposition does not yet say why the calm occasions differ from the irritated ones.")
        questions = []
        for o in others:
            ot = _theme_of_proposition(o["proposition"])
            if theme and ot and ot != theme:
                key = tuple(sorted((theme, ot)))
                ep = next((e for e in episodes if _theme_score(_main_text(e), ot)), episodes[0] if episodes else None)
                questions.append({
                    "question": QUESTIONS.get(key, "When has a similar situation felt different?").format(ep=_label(ep)),
                    "episode_id": ep["id"] if ep else None,
                    "discriminates_between": [a["id"], o["id"]],
                })
        if not questions and episodes:
            questions.append({"question": f"In {_label(episodes[0])}, what did you expect would happen next, and what would that have meant to you?", "episode_id": episodes[0]["id"], "discriminates_between": [a["id"]]})
        verdict = "likely_situational" if a.get("kind") == "situational" else ("needs_examination" if contradictions or leaps else "keep")
        return {
            "contradictions": contradictions,
            "unsupported_leaps": leaps,
            "alternative_explanations": list(THEMES[theme]["alternatives"]) if theme else ["A situational cause specific to that day"],
            "weakeners": list(THEMES[theme]["weakeners"]) if theme else [],
            "resolving_questions": questions,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------ interview
    def _interview_question(self, p: dict) -> dict:
        cands = p.get("candidates", [])
        episodes = p.get("episodes", [])
        turns = p.get("previous_turns", [])
        asked = [normalize(t.get("question", ""))[:40] for t in turns]
        stop = {"question": "", "episode_id": None, "discriminates_between": [], "rationale": "", "stop": True}
        if not cands:
            return {**stop, "stop_reason": "No candidate assumptions to examine."}
        if len(turns) >= 6:
            return {**stop, "stop_reason": "Further questioning would be speculation."}
        ranked = sorted(cands, key=lambda c: -len(c.get("supporting_episode_ids", [])))
        focus = p.get("focus_episode_id")
        focus_ep = next((e for e in episodes if e["id"] == focus), None) if focus else None
        # Discriminating questions first, one per pair of competing themes, never repeating a question already asked.
        for i, a in enumerate(ranked):
            ta = _theme_of_proposition(a["proposition"])
            for b in ranked[i + 1:]:
                tb = _theme_of_proposition(b["proposition"])
                if not ta or not tb or ta == tb:
                    continue
                key = tuple(sorted((ta, tb)))
                ep = focus_ep or next((e for e in episodes if _theme_score(_main_text(e), tb)), episodes[0] if episodes else None)
                q = QUESTIONS.get(key, "When has a similar situation felt different?").format(ep=_label(ep))
                if normalize(q)[:40] in asked:
                    continue
                return {"question": q, "episode_id": ep["id"] if ep else None, "discriminates_between": [a["id"], b["id"]],
                        "rationale": f"Separates '{ta}' from '{tb}': the two predict different reactions in this case.", "stop": False, "stop_reason": ""}
        ep = focus_ep or (episodes[0] if episodes else None)
        q = f"In {_label(ep)}, what did you expect would happen next, and what would that outcome have meant to you?"
        if normalize(q)[:40] in asked:
            return {**stop, "stop_reason": "The remaining uncertainty cannot be resolved by asking; it needs new episodes."}
        return {"question": q, "episode_id": ep["id"] if ep else None, "discriminates_between": [ranked[0]["id"]],
                "rationale": "Downward arrow: checks whether the expected consequence matches the proposition.", "stop": False, "stop_reason": ""}

    def _interview_update(self, p: dict) -> dict:
        turn = p["turn"]
        answer = normalize(turn.get("answer", ""))
        cands = p.get("candidates", [])
        first = sentences(turn.get("answer", ""))[:1]
        quote = first[0] if first else turn.get("answer", "")
        updates = []
        if turn.get("dont_know"):
            for c in cands:
                updates.append({"assumption_id": c["id"], "action": "keep", "reason": "The user could not say; nothing changes.", "supporting": [], "contradicting": []})
            return {"updates": updates, "new_candidates": [], "stop_interview": False, "stop_reason": ""}
        positive = _has(answer, POSITIVE)
        eval_negative = _has(answer, ("nervous", "judg", "worried", "watched", "anxious", "heart"))
        rested_too = _has(answer, ("slept well", "when rested", "rested too", "same thing happens", "even when rested", "when i am rested", "when i'm rested"))
        tired = _has(answer, THEMES["fatigue"]["keywords"]) and not rested_too
        denial = _has(answer, ("that's not it", "not about", "doesn't fit", "does not fit", "that is not it", "wrong"))
        for c in cands:
            theme = _theme_of_proposition(c["proposition"])
            n_support = len(c.get("supporting_episode_ids", []))
            upd = {"assumption_id": c["id"], "action": "keep", "reason": "The answer does not clearly bear on this candidate.", "supporting": [], "contradicting": []}
            if theme == "evaluation" and positive and not eval_negative:
                upd.update(action="reject" if n_support <= 1 else "revise", reason="The user reports that a freely chosen but evaluated activity felt fine, which this candidate predicts against.",
                           new_weakeners=["Freely chosen work that was evaluated felt fine"], contradicting=[{"episode_id": turn["episode_id"], "quotation": quote}])
            elif theme == "autonomy" and positive and not eval_negative:
                upd.update(action="strengthen", reason="Being watched did not trigger the reaction when the activity was self-chosen; this points to the commitment, not the audience.",
                           supporting=[{"episode_id": turn["episode_id"], "quotation": quote}])
            elif theme == "evaluation" and eval_negative:
                upd.update(action="strengthen", reason="The user reports evaluation anxiety even when the activity was chosen.", supporting=[{"episode_id": turn["episode_id"], "quotation": quote}])
            elif theme == "autonomy" and eval_negative:
                upd.update(action="revise", reason="Evaluation anxiety is present independently of commitment; this candidate needs narrowing.", new_weakeners=["Distress appears when watched even without any commitment"])
            elif theme == "fatigue" and rested_too:
                upd.update(action="reject" if n_support <= 1 else "revise", reason="The user reports the same reaction on rested days, so exhaustion does not explain the pattern on its own.",
                           new_weakeners=["The same reaction occurs on well-rested days"], contradicting=[{"episode_id": turn["episode_id"], "quotation": quote}])
            elif theme == "autonomy" and rested_too:
                upd.update(action="strengthen", reason="The reaction persists when rested, which favours a general assumption over a situational one.", supporting=[{"episode_id": turn["episode_id"], "quotation": quote}])
            elif theme == "fatigue" and tired:
                upd.update(action="strengthen", reason="The user confirms exhaustion at the time.", supporting=[{"episode_id": turn["episode_id"], "quotation": quote}])
            elif theme == "autonomy" and _has(answer, ("my own time", "in charge", "had not chosen", "hadn't chosen", "lose control", "fill up")):
                upd.update(action="strengthen", reason="The expected consequence the user names is loss of control over their time, which is what the proposition states.",
                           supporting=[{"episode_id": turn["episode_id"], "quotation": quote}])
            elif theme == "autonomy" and tired and not positive:
                upd.update(action="revise", reason="Exhaustion was present; the autonomy reading must account for it.", new_weakeners=["The reaction only appears on exhausted days"])
            elif denial and c["id"] in (turn.get("discriminates_between") or [])[:1]:
                upd.update(action="revise", reason="The user says the proposition does not fit as worded; it is revised rather than defended.", new_weakeners=[quote])
            updates.append(upd)
        stop = _has(answer, ("speculat", "guessing", "making this up"))
        return {"updates": updates, "new_candidates": [], "stop_interview": stop, "stop_reason": "The user indicates the answers are becoming speculation." if stop else ""}

    def _revise_correction(self, p: dict) -> dict:
        a = p["assumption"]
        correction = (p.get("correction") or "").strip()
        if p.get("action") == "reject":
            theme = _theme_of_proposition(a["proposition"])
            alt_theme = next((t for t in THEMES if t != theme and any(_theme_score(_main_text(e), t) for e in p.get("episodes", []))), None)
            alt = None
            if alt_theme:
                th = THEMES[alt_theme]
                alt = {"proposition": th["proposition"], "kind": th["kind"], "situations": list(th["situations"]), "alternative_explanations": list(th["alternatives"]),
                       "prediction_new_situation": th["prediction"], "weakeners": list(th["weakeners"]), "supporting": [], "contradicting": [], "rationale": "Alternative offered after the user rejected the previous reading."}
            return {"proposition": a["proposition"], "kind": a["kind"], "situations": a["situations"], "weakeners": a["weakeners"], "alternative_explanations": a["alternative_explanations"],
                    "prediction_new_situation": a["prediction_new_situation"], "reason": "The user rejected this interpretation. That is information about their experience; the candidate is set aside and an alternative is offered.", "propose_alternative": alt}
        looks_like_prop = bool(re.match(r"^(if|to|when|unless)\b", correction, re.I)) or " then " in correction.lower() or " must " in correction.lower()
        if looks_like_prop:
            return {"proposition": correction, "kind": "must" if " must " in correction.lower() else a["kind"], "situations": a["situations"], "weakeners": a["weakeners"],
                    "alternative_explanations": a["alternative_explanations"], "prediction_new_situation": a["prediction_new_situation"],
                    "reason": "Replaced the proposition with the user's own wording.", "propose_alternative": None}
        return {"proposition": a["proposition"], "kind": a["kind"], "situations": list(a["situations"]) + [f"Qualified by the user: {correction}"], "weakeners": a["weakeners"],
                "alternative_explanations": a["alternative_explanations"], "prediction_new_situation": a["prediction_new_situation"],
                "reason": "Narrowed the situations to honour the user's correction.", "propose_alternative": None}

    # ------------------------------------------------------------------ holdout
    def _holdout_predict(self, p: dict) -> dict:
        situation = p.get("situation", "")
        preds = []
        for c in p.get("candidates", []):
            theme = _theme_of_proposition(c["proposition"])
            applies = bool(theme and _theme_score(situation, theme))
            if applies:
                t, e, act = THEMES[theme]["predicted"]
                preds.append({"assumption_id": c["id"], "applies": True, "predicted_thought": t, "predicted_emotion": e, "predicted_action": act, "confidence": 0.7,
                              "rationale": f"The situation matches the '{theme}' pattern this assumption describes."})
            else:
                preds.append({"assumption_id": c["id"], "applies": False, "predicted_thought": "No strong reaction predicted by this assumption.", "predicted_emotion": "Neutral or mild",
                              "predicted_action": "Respond in an ordinary way", "confidence": 0.3, "rationale": "The situation does not fall under the assumption's stated situations."})
        baseline = {"predicted_thought": "Consider the request on its merits.", "predicted_emotion": "Mild, mixed feelings", "predicted_action": "Respond according to mood and availability",
                    "rationale": "Generic description with no person-specific assumption."}
        return {"predictions": preds, "baseline": baseline}

    def _holdout_compare(self, p: dict) -> dict:
        rec = normalize(" ".join(str(p["recorded"].get(k, "")) for k in ("immediate_thought", "emotion_body", "action_outcome")))
        families = {"irritation": ("irritat", "annoy", "resent", "trapped", "have to", "tight", "clench", "jaw"), "nervous": ("nervous", "anxious", "worried", "heart", "judg", "good enough"),
                    "calm": ("calm", "fine", "glad", "relaxed", "easy", "enjoy", "ordinary", "mild", "neutral", "merits", "pleased", "curious"), "drained": ("drained", "flat", "tired", "exhaust")}
        def fam(text: str) -> set[str]:
            return {f for f, ks in families.items() if any(k in text for k in ks)}
        rec_fam = fam(rec)
        rec_tokens = set(tokenize(rec))
        out = []
        for pr in p.get("predictions", []):
            txt = normalize(" ".join(str(pr.get(k, "")) for k in ("predicted_thought", "predicted_emotion", "predicted_action")))
            overlap = len(rec_tokens & set(tokenize(txt))) / (len(set(tokenize(txt))) or 1)
            f = fam(txt)
            fam_match = 1.0 if (f & rec_fam) else (0.0 if f and rec_fam else 0.5)
            score = round(0.7 * fam_match + 0.3 * min(1.0, overlap * 3), 2)
            note = "Predicted reaction is in the same family as the recorded one." if fam_match == 1.0 else ("Predicted reaction differs from the recorded one." if fam_match == 0.0 else "Could not tell: neither side is specific enough.")
            out.append({"assumption_id": pr.get("assumption_id"), "match_score": score, "note": note})
        return {"comparisons": out}
