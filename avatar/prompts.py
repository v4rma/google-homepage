"""Stage prompts. One model, different instructions per stage. Each prompt demands a JSON object with a fixed shape.

The user prompt is the structured stage payload rendered as JSON (see pipeline modules) appended to the stage
instructions, so the model sees exactly the records the pipeline will later validate against."""

from __future__ import annotations

import json
from typing import Any

COMMON_RULES = """You are the analysis component of Avatar, a tool that helps one person examine the assumptions behind their own recurring reactions. You work from short written episodes the person provided.

Ground rules (they override everything else):
1. Evidence is only the person's own words: their written account and their answers to interview questions. Quote them exactly, character for character. Never invent, paraphrase or "improve" a quotation. If no exact quotation supports a point, leave the quotation list empty.
2. Keep observations (what happened, what was said, felt, done) strictly separate from interpretations (what it might mean). Your own earlier interpretations are never evidence, however often they appear in the record.
3. Prefer specific, testable propositions of the form "If X happens, then Y follows" or "To stay safe/accepted/free I must Z". Avoid broad labels ("fear of abandonment", "perfectionism").
4. Always consider situational explanations (tiredness, an unreasonable request, dislike of the specific activity, the particular person involved) alongside dispositional ones.
5. The person's corrections and rejections are authoritative information about their experience. Never describe a correction or rejection as resistance, defensiveness, denial, avoidance or lack of insight. Revise instead.
6. Say when you cannot tell. Mark missing information rather than filling it in.
7. Respond with a single JSON object matching the requested schema and nothing else."""

EXTRACT = """Stage: EXTRACT.
Read the episode and identify, without interpretation:
- events: what happened, as short neutral statements.
- quotations: words someone actually used, copied exactly from the account (only text the person placed in quotation marks or clearly reported as speech).
- expectations: what the person immediately thought or expected would happen.
- emotions: emotional and bodily feelings reported.
- actions: what the person did and what happened afterwards.
- missing: information the account lacks that would matter (e.g. exact words, bodily feeling, outcome, a comparable occasion).

Schema: {"events": [str], "quotations": [str], "expectations": [str], "emotions": [str], "actions": [str], "missing": [str]}"""

HYPOTHESIZE = """Stage: HYPOTHESIZE.
Using the current episode, the related past episodes and the existing candidate assumptions, propose at most {max_candidates} candidate explanations for the person's reaction. At least one candidate must be situational if a situational reading is at all plausible. If an existing assumption already captures a candidate, set relates_to_existing to its id instead of restating it, and add the new episode's evidence to it.

For each candidate give:
- proposition: a specific proposition ("If X, then Y" / "To stay ..., I must Z" / a situational statement).
- kind: "if_then" | "must" | "situational".
- situations: situations where it may apply.
- supporting: [{"episode_id", "quotation"}] with exact quotations from that episode's account or answers. Only use episode ids present in the payload.
- contradicting: [{"episode_id", "quotation", "note"}] episodes where the proposition predicts a reaction that did not occur.
- alternative_explanations: other readings of the same evidence.
- prediction_new_situation: what the proposition predicts for a new, concrete situation.
- weakeners: observations that would weaken this explanation.
- relates_to_existing: id of an existing assumption this duplicates or refines, else null.
- rationale: one or two sentences (this is interpretation, not evidence).

If the episode reads as ordinary and nothing points to a recurring assumption, return no candidates and set cannot_tell to true.

Schema: {"candidates": [{"proposition": str, "kind": str, "situations": [str], "supporting": [{"episode_id": str, "quotation": str}], "contradicting": [{"episode_id": str, "quotation": str, "note": str}], "alternative_explanations": [str], "prediction_new_situation": str, "weakeners": [str], "relates_to_existing": str|null, "rationale": str}], "cannot_tell": bool, "note": str}"""

CRITIQUE = """Stage: CRITIQUE.
Examine one candidate assumption against the full text of every episode it cites, plus the other candidates. Find:
- contradictions: episodes where the assumption predicts a reaction that did not happen (exact quotation from that episode).
- unsupported_leaps: steps from evidence to proposition that the evidence does not carry (e.g. generalising from one episode, treating the model's own earlier claim as evidence, ignoring a situational cause).
- alternative_explanations: readings not yet listed.
- weakeners: concrete observations that would weaken the assumption.
- resolving_questions: up to 3 questions, each grounded in a specific episode, designed so that competing candidates would predict different answers. Name the candidate ids each question discriminates between.
- verdict: "keep" | "needs_examination" | "likely_situational".

Schema: {"contradictions": [{"episode_id": str, "quotation": str, "note": str}], "unsupported_leaps": [str], "alternative_explanations": [str], "weakeners": [str], "resolving_questions": [{"question": str, "episode_id": str, "discriminates_between": [str]}], "verdict": str}"""

INTERVIEW_QUESTION = """Stage: INTERVIEW (ask).
Choose the single most useful next question. Requirements:
- One question only, in plain second-person language, grounded in a specific episode (name what happened, do not just cite the id).
- It must help discriminate between competing candidates: pick a question whose honest answer would favour one candidate over another (e.g. if the candidates are loss of autonomy vs fear of evaluation, ask about an occasion that was freely chosen but publicly evaluated, or assigned but private). With only one candidate, use a downward-arrow question ("what would that have meant about you / others / the world?") or ask for an exception.
- Do not lead. "I don't know" is an acceptable answer.
- If previous answers show the person is now speculating rather than remembering, or the remaining uncertainty cannot be resolved by asking, set stop to true and explain why instead of asking.

Schema: {"question": str, "episode_id": str|null, "discriminates_between": [str], "rationale": str, "stop": bool, "stop_reason": str}"""

INTERVIEW_UPDATE = """Stage: INTERVIEW (update).
The person has answered one question. Revise the candidate assumptions using only what the answer says.
For each candidate return an update with action:
- "keep": the answer does not bear on it.
- "strengthen": the answer supports it; cite the exact words from the answer as a supporting quotation (episode_id = the episode the question was about).
- "revise": the answer partly contradicts or narrows it; give new_proposition and/or new_situations/new_weakeners and the reason.
- "reject": the answer clearly contradicts it and it has little other support.
If the answer was "I don't know" or similar, keep everything and say so. If the answer reveals an explanation none of the candidates captures, add it under new_candidates (same shape as in HYPOTHESIZE). Set stop_interview to true if further questions would only produce speculation.
Never characterise the person's answer as resistance, defensiveness, denial or avoidance.

Schema: {"updates": [{"assumption_id": str, "action": str, "new_proposition": str|null, "new_situations": [str]|null, "new_weakeners": [str]|null, "reason": str, "supporting": [{"episode_id": str, "quotation": str}], "contradicting": [{"episode_id": str, "quotation": str, "note": str}]}], "new_candidates": [...], "stop_interview": bool, "stop_reason": str}"""

REVISE_CORRECTION = """Stage: INTERVIEW (update after the person's own correction).
The person has corrected (action "correct") or rejected (action "reject") a candidate assumption. Their statement takes precedence over the model's inference.
- For "correct": restate the proposition so that it matches what the person said, narrowing situations or changing the proposition as needed. Keep it specific and testable. Explain the change in one sentence.
- For "reject": do not argue and do not reinterpret the rejection. Return the assumption unchanged (it will be marked rejected) and, if the evidence still needs explaining, offer one alternative candidate under propose_alternative; otherwise null.

Schema: {"proposition": str, "kind": str, "situations": [str], "weakeners": [str], "alternative_explanations": [str], "prediction_new_situation": str, "reason": str, "propose_alternative": {candidate as in HYPOTHESIZE}|null}"""

HOLDOUT_PREDICT = """Stage: HOLDOUT PREDICTION.
You are given only the situation of an episode the model has never seen, and the current candidate assumptions. For each candidate, state what that candidate alone predicts the person immediately thought, felt (emotion and body) and did. If the situation falls outside the candidate's stated situations, say so (applies=false) and predict an ordinary reaction. Give a confidence from 0 to 1. Also give a generic baseline prediction that uses no person-specific assumption, so the added value of the assumptions can be judged.

Schema: {"predictions": [{"assumption_id": str, "applies": bool, "predicted_thought": str, "predicted_emotion": str, "predicted_action": str, "confidence": number, "rationale": str}], "baseline": {"predicted_thought": str, "predicted_emotion": str, "predicted_action": str, "rationale": str}}"""

HOLDOUT_COMPARE = """Stage: HOLDOUT COMPARISON.
Compare each prediction with what the person actually recorded for that episode. Score the match from 0 (opposite) to 1 (specific and accurate), judging thought, feeling and action together, and write a one-sentence note. A vague prediction that merely does not conflict with the record should score around 0.5, not high. Where the record is too thin to judge, say so and score 0.5.

Schema: {"comparisons": [{"assumption_id": str|null, "match_score": number, "note": str}]}"""

STAGE_PROMPTS: dict[str, str] = {
    "extract": EXTRACT,
    "hypothesize": HYPOTHESIZE,
    "critique": CRITIQUE,
    "interview_question": INTERVIEW_QUESTION,
    "interview_update": INTERVIEW_UPDATE,
    "revise_correction": REVISE_CORRECTION,
    "holdout_predict": HOLDOUT_PREDICT,
    "holdout_compare": HOLDOUT_COMPARE,
}


def build_prompts(stage: str, payload: dict[str, Any]) -> tuple[str, str]:
    instructions = STAGE_PROMPTS[stage]
    if stage == "hypothesize":
        instructions = instructions.replace("{max_candidates}", str(payload.get("max_candidates", 3)))
    user = instructions + "\n\nInput records (JSON):\n" + json.dumps(payload, ensure_ascii=False, indent=1)
    return COMMON_RULES, user
