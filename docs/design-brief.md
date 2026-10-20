# Avatar — project context

Design brief for the first prototype, as given by Ragesha Varma (2026-10-20).

## Goal

Build an application around an existing LLM that develops — and tests — a provisional model of the user's assumptions. Start with prompting and a small memory store; training a new model is unnecessary for the first prototype.

The central challenge is distinguishing an assumption that explains the user's experience from a compelling story the AI has invented about them. An LLM can help investigate beliefs the user hasn't articulated, but it cannot directly observe their unconscious.

## 1. Define precisely what it should discover

Use a structure that separates experiences from interpretations. Example (autonomy):

| Level | Example |
|---|---|
| Situation | Someone asks me to commit to a daily activity. |
| Immediate thought | "Now I have to do this." |
| Emotion and action | Irritation; postponement or refusal. |
| Expected consequence | "If I agree, I'll lose control over my time." |
| Possible underlying assumption | "Commitment means surrendering my freedom." |

The final row is a candidate explanation. Other explanations — exhaustion, dislike of the activity, fear of evaluation, or an unreasonable demand — may fit better.

Represent assumptions as specific propositions: "If X happens, then Y follows" or "To remain safe/accepted/free, I must Z." These are easier to examine than broad labels such as "fear of abandonment."

## 2. Collect concrete episodes before asking for interpretations

Gather roughly 10–15 short accounts across different situations (practical starting target, not a scientific minimum). For each account, record:

- What happened, including the words someone actually used.
- What the user immediately thought or expected.
- What they felt emotionally and physically.
- What they did, and what happened afterward.
- A comparable occasion when they responded differently.

Include ordinary and positive experiences, otherwise the model learns a disproportionately troubled picture.

The user writes the first account before seeing the AI's interpretation. Preserve that original separately from subsequent discussion so the system can distinguish the initial experience from ideas introduced during the interview.

## 3. Investigate meaning through questions

Starting method: guided discovery, including the CBT "downward arrow" (what a thought would imply about oneself, other people, or the world). Jointly examine evidence rather than assuming a belief is mistaken. These are borrowed methods; their clinical use does not validate an automated version.

Useful questions:

- "At that moment, what did you expect would happen next?"
- "What would that outcome have meant to you?"
- "What felt at stake?"
- "When has a similar situation felt different?"
- "What explanation have we overlooked?"

Ask one question at a time, grounded in the episode. Allow "I don't know" and stop when further questioning produces speculation.

Key difference from a journaling app: each question should help distinguish competing explanations. E.g. if the candidates are loss of autonomy vs. fear of evaluation, ask about occasions when the activity was freely chosen but publicly evaluated, and occasions when it was assigned but private.

## 4. Four processing stages

Same model, different instructions per stage:

| Stage | Responsibility |
|---|---|
| Extract | Identify reported events, exact quotations, expectations, emotions, and actions. Mark missing information. |
| Hypothesize | Propose up to three explanations using relevant episodes, including situational explanations. |
| Critique | Find contradictions, unsupported leaps, alternative explanations, and questions that could resolve uncertainty. |
| Interview and update | Ask the most useful next question, then revise the hypotheses using the answer. |

For every candidate assumption, store:

```
Candidate assumption:
Situations where it may apply:
Supporting episode IDs and exact quotations:
Contradicting episodes:
Alternative explanations:
Prediction for a new situation:
What would weaken this explanation:
Your corrections:
Status: proposed / under examination / revised / rejected
```

Keep observations and interpretations in separate records. An AI-generated claim must never become supporting evidence merely because it appears repeatedly in the conversation. The user's rejection of an interpretation should prompt revision, not be reinterpreted as "resistance."

## 5. Small first implementation

- One capable LLM accessed through an API.
- A simple interface for entering episodes and answering questions.
- A local SQLite database containing episodes, hypotheses, and revisions.
- Retrieval of a few relevant past episodes for each new conversation.
- A visible panel to inspect, correct, or delete the model's assumptions.

Begin with records the user deliberately provides. Keep sensitive material out of diagnostic logs; check provider data-retention settings before sending it.

Personalization initially comes from retrieved evidence and preserved corrections. Consider fine-tuning only later, with enough reviewed examples demonstrating a recurring failure that prompting and retrieval do not solve.

Technical precedents:

- Mind Mapper (MIT Media Lab): multi-stage LLM pipeline extracting behavioral patterns from everyday conversations; evaluated on perceived accuracy/usefulness.
- Behavior Latticing (Stanford preprint, arxiv.org): connecting separate observations into inferred motivations; appendix includes prompts to study.

## 6. Test whether it learns something useful

Reserve a few episodes the model has not seen. Give it the situation while withholding the response, ask what each candidate assumption predicts, and compare with what was recorded.

Assess whether it:

- Produces specific, traceable explanations.
- Explains both the pattern and its exceptions.
- Adds predictive value beyond a generic description.
- Revises its interpretation when contrary evidence appears.
- Recognizes when it cannot tell.

Prediction supports a hypothesis's usefulness; it does not prove a psychological root cause. Also record whether an interpretation feels revealing, but treat that as a separate measure (sycophancy risk: people can prefer convincing agreement over accurate responses).

## First milestone

One well-supported, revisable assumption about one recurring pattern. Later components (digital twin perspective, pratipaksha-bhavana-inspired alternative responses) build on how the system earns an interpretation.
