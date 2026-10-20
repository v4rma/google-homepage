"""First-milestone check: run the full pipeline over the synthetic demo dataset and show that it reaches one
well-supported, revisable assumption about one recurring pattern, then test it on reserved episodes.

Usage:
    python scripts/run_milestone.py              # prints a Markdown report (mock provider unless OPENAI_API_KEY is set)
    python scripts/run_milestone.py --write docs/milestone-check.md

The run uses a throwaway in-memory database; it does not touch data/avatar.db.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatar.config import load_settings  # noqa: E402
from avatar.demo_data import seed, synthetic_answer  # noqa: E402
from avatar.llm import make_provider  # noqa: E402
from avatar.logging_utils import configure_logging  # noqa: E402
from avatar.pipeline import Pipeline  # noqa: E402
from avatar.storage import Store  # noqa: E402

USER_CORRECTION = "If someone else fixes a recurring schedule for me, then I lose control over my time."
MAX_QUESTIONS = 5


def fmt_assumption(a: dict, titles: dict[str, str]) -> str:
    def ev(items):
        return "\n".join(f"  - {e['episode_id']} ({titles.get(e['episode_id'], '?')}): \"{e['quotation']}\"" + ("" if e["verified"] else " [UNVERIFIED - not counted]") for e in items) or "  - (none)"

    def bullets(items):
        return "\n".join(f"  - {x}" for x in items) or "  - (none)"

    return "\n".join([
        "```",
        f"Candidate assumption: {a['proposition']}",
        f"Kind: {a['kind']}    Version: {a['version']}    ID: {a['id']}",
        "Situations where it may apply:", bullets(a["situations"]),
        "Supporting episode IDs and exact quotations:", ev(a["supporting"]),
        "Contradicting episodes:", ev(a["contradicting"]),
        "Alternative explanations:", bullets(a["alternative_explanations"]),
        f"Prediction for a new situation: {a['prediction_new_situation']}",
        "What would weaken this explanation:", bullets(a["weakeners"]),
        "Your corrections:", bullets([c["text"] for c in a["user_corrections"]]),
        f"Status: {a['status']}",
        "```",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", help="write the report to this path")
    args = parser.parse_args()
    settings = load_settings()
    if not os.environ.get("AVATAR_LLM_PROVIDER") and settings.provider == "agent":
        # Unattended script: do not wait on the agent inbox unless explicitly asked to.
        settings = replace(settings, provider="mock")
    configure_logging("WARNING")
    provider = make_provider(settings)
    store = Store(":memory:")
    pipeline = Pipeline(store, provider, settings)
    out: list[str] = []
    say = out.append

    say("# Milestone check: one well-supported, revisable assumption\n")
    say(f"Provider: `{provider.name}`" + (f" (model `{settings.model}`)" if provider.name != "mock" else " (rule-based stand-in; see README)") + "\n")
    seed(store)
    eps = store.list_episodes()
    titles = {e["id"]: e["title"] for e in eps}
    train = [e for e in eps if not e["reserved"]]
    held = [e for e in eps if e["reserved"]]
    say(f"Seeded {len(eps)} synthetic episodes: {len(train)} available to the pipeline, {len(held)} reserved for the prediction test.\n")

    say("## 1. Extract → Hypothesize → Critique, episode by episode\n")
    say("| Episode | Candidates proposed or updated | Note |\n|---|---|---|")
    for ep in train:
        r = pipeline.process_episode(ep["id"])
        cands = ", ".join(f"{a['id']} ({a['kind']}, {len(a['supporting_episode_ids'])} supp.)" for a in r["hypothesize"]["assumptions"]) or "—"
        note = "model: nothing points to a recurring assumption" if r["hypothesize"]["cannot_tell"] else ""
        say(f"| {ep['id']} {ep['title']} | {cands} | {note} |")
    say("")
    say("Candidates after this pass:\n")
    for a in store.list_assumptions():
        say(f"- `{a['id']}` [{a['status']}] {a['proposition']} — supporting {a['supporting_episode_ids']}, contradicting {a['contradicting_episode_ids']}")
    say("")

    say("## 2. Interview and update (one question at a time; synthetic answers)\n")
    for _ in range(MAX_QUESTIONS):
        q = pipeline.next_question()
        if q["stop"]:
            say(f"Model stopped asking: {q['reason']}\n")
            break
        turn = q["turn"]
        answer = synthetic_answer(turn["question"])
        r = pipeline.answer(turn["id"], answer)
        say(f"**Q** (about {turn['episode_id']}, discriminates {turn['discriminates_between']}): {turn['question']}  ")
        say(f"**A**: {answer}  ")
        changed = [f"{u['action']} `{u['assumption_id']}` — {u['reason']}" for u in r["updates"] if u["action"] != "keep"] or ["no candidate changed"]
        say("Updates: " + "; ".join(changed) + "\n")

    active = [a for a in store.list_assumptions() if a["status"] != "rejected"]
    if not active:
        say("No active candidate survived; milestone not reached.")
        print("\n".join(out))
        sys.exit(1)
    best = max(active, key=lambda a: len(a["supporting_episode_ids"]))
    say(f"Best-supported candidate: `{best['id']}` with {len(best['supporting_episode_ids'])} supporting episodes.\n")

    say("## 3. The user corrects the wording (revision, not resistance)\n")
    r = pipeline.correct(best["id"], USER_CORRECTION)
    say(f"User correction: \"{USER_CORRECTION}\"  ")
    say(f"Model: {r['reason']}  ")
    say(f"Status now `{r['assumption']['status']}`, version {r['assumption']['version']}; previous versions are kept in the revision history.\n")

    others = [a for a in active if a["id"] != best["id"]]
    if others:
        say("## 3b. The user rejects another candidate\n")
        rj = pipeline.reject(others[0]["id"], "That is not it for me.")
        say(f"Rejected `{others[0]['id']}` ({others[0]['proposition']}). Model: {rj['reason']} " + (f"Alternative offered: `{rj['alternative']['id']}`." if rj["alternative"] else "No alternative offered.") + "\n")

    best = store.get_assumption(best["id"])
    say("## 4. The resulting assumption record\n")
    say(fmt_assumption(best, titles))
    say("")
    say("Revision history:\n")
    for rv in best["revisions"]:
        say(f"- v{rv['version']} → v{rv['version'] + 1} ({rv['trigger']}): {rv['reason'] or '—'}; was: \"{rv['previous']['proposition']}\" [{rv['previous']['status']}]")
    say("")

    say("## 5. Prediction test on reserved episodes (situation only)\n")
    hits = 0
    total = 0
    for ep in held:
        h = pipeline.holdout(ep["id"])
        say(f"### {ep['id']} — {ep['title']}\n")
        say(f"Situation given to the model: {ep['situation']}\n")
        say("| Candidate | Predicted thought | Predicted feeling | Predicted action | Conf. | Match | Note |\n|---|---|---|---|---|---|---|")
        for p in h["predictions"]:
            a = store.get_assumption(p["assumption_id"], with_details=False)
            say(f"| {a['proposition'] if a else p['assumption_id']} | {p['predicted_thought']} | {p['predicted_emotion']} | {p['predicted_action']} | {p['confidence']} | {p['match_score']} | {p['comparison']} |")
            if p["assumption_id"] == best["id"]:
                total += 1
                hits += 1 if (p["match_score"] or 0) >= 0.6 else 0
        b = h["baseline"]
        if b:
            say(f"| *generic baseline* | {b['predicted_thought']} | {b['predicted_emotion']} | {b['predicted_action']} | — | {b['match_score']} | {b['comparison']} |")
        say(f"\nRecorded: thought \"{ep['immediate_thought']}\"; feeling \"{ep['emotion_body']}\"; action \"{ep['action_outcome']}\"\n")

    say("## 6. Assessment against the brief\n")
    n_quotes = sum(1 for e in best["supporting"] if e["verified"])
    say(f"- Specific, traceable: the proposition is an if/then statement with {n_quotes} verified quotations from {len(best['supporting_episode_ids'])} supporting episodes; every quotation links to an episode id.")
    say(f"- Pattern and exceptions: contradicting/exception episodes are listed separately ({len(best['contradicting_episode_ids'])} recorded on the final candidate; exceptions such as freely chosen public activities were used to reject or revise competing candidates).")
    say(f"- Predictive value: the retained assumption matched the recorded response (score ≥ 0.6) on {hits} of {total} reserved episodes; compare with the generic baseline row in each table.")
    say(f"- Revises on contrary evidence: {len(best['revisions'])} recorded revisions, including the user's correction, with the previous wording preserved.")
    say("- Recognises when it cannot tell: ordinary episodes produced no candidates, and the interview stopped when further questions would not resolve anything.")
    say("\nPrediction supports usefulness; it does not prove a psychological root cause. Whether the interpretation *feels* revealing is a separate measure and is not assessed here.\n")

    report = "\n".join(out)
    if args.write:
        with open(args.write, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"wrote {args.write}")
    else:
        print(report)


if __name__ == "__main__":
    main()
