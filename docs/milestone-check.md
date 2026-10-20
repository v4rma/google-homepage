# Milestone check: one well-supported, revisable assumption

Provider: `mock` (rule-based stand-in; see README)

Seeded 13 synthetic episodes: 10 available to the pipeline, 3 reserved for the prediction test.

## 1. Extract → Hypothesize → Critique, episode by episode

| Episode | Candidates proposed or updated | Note |
|---|---|---|
| ep-demo01 Invitation to a weekly running club | as-ffea9898 (if_then, 5 supp.), as-e465ca8e (situational, 1 supp.) |  |
| ep-demo02 New daily standup announced | as-ffea9898 (if_then, 6 supp.) |  |
| ep-demo03 Sister's book club | as-ffea9898 (if_then, 6 supp.), as-8d27c9a6 (if_then, 0 supp.) |  |
| ep-demo04 Volunteered to present a side project | as-8d27c9a6 (if_then, 0 supp.), as-ffea9898 (if_then, 6 supp.) |  |
| ep-demo05 Landlord asks me to water plants daily | as-ffea9898 (if_then, 6 supp.) |  |
| ep-demo06 Meal-prep plan after a night shift | as-ffea9898 (if_then, 6 supp.), as-e465ca8e (situational, 1 supp.), as-8d27c9a6 (if_then, 0 supp.) |  |
| ep-demo07 A walk in the park | — | model: nothing points to a recurring assumption |
| ep-demo08 Neighbour's sofa | — | model: nothing points to a recurring assumption |
| ep-demo09 Twelve-month gym contract | as-ffea9898 (if_then, 6 supp.) |  |
| ep-demo10 Detailed code review | as-ffea9898 (if_then, 6 supp.), as-8d27c9a6 (if_then, 0 supp.), as-e465ca8e (situational, 1 supp.) |  |

Candidates after this pass:

- `as-ffea9898` [under examination] If I agree to a recurring commitment, then I will lose control over my time. — supporting ['ep-demo01', 'ep-demo02', 'ep-demo03', 'ep-demo05', 'ep-demo06', 'ep-demo09'], contradicting []
- `as-e465ca8e` [under examination] The irritation on that occasion is explained by exhaustion at the time rather than by a general belief. — supporting ['ep-demo06'], contradicting []
- `as-8d27c9a6` [under examination] If my performance is visible to others, then I will be judged as inadequate. — supporting [], contradicting ['ep-demo04', 'ep-demo10']

## 2. Interview and update (one question at a time; synthetic answers)

**Q** (about ep-demo06, discriminates ['as-ffea9898', 'as-e465ca8e']): On the occasion in the episode ep-demo06, how rested were you? Has the same kind of request come on a day you felt rested, and what happened then?  
**A**: I was wrecked that day, but honestly the same thing happens when I have slept well.  
Updates: strengthen `as-ffea9898` — The reaction persists when rested, which favours a general assumption over a situational one.; reject `as-e465ca8e` — The user reports the same reaction on rested days, so exhaustion does not explain the pattern on its own.

**Q** (about ep-demo04, discriminates ['as-ffea9898', 'as-8d27c9a6']): Think of a time when you chose an activity yourself but other people watched or reviewed it. What went through your mind then, compared with the episode ep-demo04?  
**A**: When I chose to present my side project it felt fine, even with the whole team watching. It is the fixed schedule that gets me, not the audience.  
Updates: strengthen `as-ffea9898` — Being watched did not trigger the reaction when the activity was self-chosen; this points to the commitment, not the audience.; reject `as-8d27c9a6` — The user reports that a freely chosen but evaluated activity felt fine, which this candidate predicts against.

**Q** (about ep-demo01, discriminates ['as-ffea9898']): In the episode ep-demo01, what did you expect would happen next, and what would that outcome have meant to you?  
**A**: I expected my weeks to fill up with things I had not chosen. It would have meant I am not in charge of my own time any more.  
Updates: strengthen `as-ffea9898` — The expected consequence the user names is loss of control over their time, which is what the proposition states.

Model stopped asking: The remaining uncertainty cannot be resolved by asking; it needs new episodes.

Best-supported candidate: `as-ffea9898` with 7 supporting episodes.

## 3. The user corrects the wording (revision, not resistance)

User correction: "If someone else fixes a recurring schedule for me, then I lose control over my time."  
Model: Replaced the proposition with the user's own wording.  
Status now `revised`, version 3; previous versions are kept in the revision history.

## 4. The resulting assumption record

```
Candidate assumption: If someone else fixes a recurring schedule for me, then I lose control over my time.
Kind: if_then    Version: 3    ID: as-ffea9898
Situations where it may apply:
  - Being asked to commit to a regular or recurring activity
  - Signing up for something with a fixed schedule set by someone else
Supporting episode IDs and exact quotations:
  - ep-demo01 (Invitation to a weekly running club): "Want to join our Tuesday run? It's every week, same time."
  - ep-demo09 (Twelve-month gym contract): "Twelve months. I'd be stuck."
  - ep-demo05 (Landlord asks me to water plants daily): "Could you water the plants in the hall every day for the next two weeks?"
  - ep-demo06 (Meal-prep plan after a night shift): "Let's plan weekly meal prep, same slot every Sunday."
  - ep-demo02 (New daily standup announced): "From now on everyone joins the 9am standup every day."
  - ep-demo03 (Sister's book club): "Can you commit to reading one book a month with us?"
  - ep-demo06 (Meal-prep plan after a night shift): "I was wrecked that day, but honestly the same thing happens when I have slept well."
  - ep-demo04 (Volunteered to present a side project): "When I chose to present my side project it felt fine, even with the whole team watching."
  - ep-demo01 (Invitation to a weekly running club): "I expected my weeks to fill up with things I had not chosen."
Contradicting episodes:
  - (none)
Alternative explanations:
  - Tiredness or low energy on that particular day
  - Dislike of the specific activity rather than of commitment itself
  - The request itself was unreasonable
Prediction for a new situation: When asked to commit to a weekly activity, the first thought will be about losing control of time, followed by irritation and postponing the answer.
What would weaken this explanation:
  - An occasion where a recurring commitment was accepted calmly and kept without irritation
  - Irritation that appears equally for one-off requests
Your corrections:
  - If someone else fixes a recurring schedule for me, then I lose control over my time.
Status: revised
```

Revision history:

- v1 → v2 (critique): critique produced; awaiting interview; was: "If I agree to a recurring commitment, then I will lose control over my time." [proposed]
- v2 → v3 (user_correction): Replaced the proposition with the user's own wording.; was: "If I agree to a recurring commitment, then I will lose control over my time." [under examination]

## 5. Prediction test on reserved episodes (situation only)

### ep-demo11 — Fixed slot at the community garden

Situation given to the model: The community garden coordinator asked: "Could you take a fixed slot every Saturday morning?"

| Candidate | Predicted thought | Predicted feeling | Predicted action | Conf. | Match | Note |
|---|---|---|---|---|---|---|
| If someone else fixes a recurring schedule for me, then I lose control over my time. | Now I have to do this. | Irritation; tightness in the chest or jaw | Postpone the answer or decline | 0.7 | 0.81 | Predicted reaction is in the same family as the recorded one. |
| *generic baseline* | Consider the request on its merits. | Mild, mixed feelings | Respond according to mood and availability | — | 0.0 | Predicted reaction differs from the recorded one. |

Recorded: thought "There goes every Saturday."; feeling "Irritation, tight shoulders."; action "I said I would rather drop in when I am free. She seemed a little disappointed."

### ep-demo12 — One-off review of a friend's talk

Situation given to the model: A friend asked whether I could review the slides for her talk once before Friday and give honest feedback.

| Candidate | Predicted thought | Predicted feeling | Predicted action | Conf. | Match | Note |
|---|---|---|---|---|---|---|
| If someone else fixes a recurring schedule for me, then I lose control over my time. | No strong reaction predicted by this assumption. | Neutral or mild | Respond in an ordinary way | 0.3 | 0.7 | Predicted reaction is in the same family as the recorded one. |
| *generic baseline* | Consider the request on its merits. | Mild, mixed feelings | Respond according to mood and availability | — | 0.7 | Predicted reaction is in the same family as the recorded one. |

Recorded: thought "Happy to."; feeling "Calm, a little flattered."; action "I did it that evening and we talked it through for half an hour."

### ep-demo13 — Dinner invitation

Situation given to the model: Two friends invited me to dinner on Friday at their place.

| Candidate | Predicted thought | Predicted feeling | Predicted action | Conf. | Match | Note |
|---|---|---|---|---|---|---|
| If someone else fixes a recurring schedule for me, then I lose control over my time. | No strong reaction predicted by this assumption. | Neutral or mild | Respond in an ordinary way | 0.3 | 0.7 | Predicted reaction is in the same family as the recorded one. |
| *generic baseline* | Consider the request on its merits. | Mild, mixed feelings | Respond according to mood and availability | — | 0.7 | Predicted reaction is in the same family as the recorded one. |

Recorded: thought "Nice."; feeling "Glad, relaxed."; action "I went, stayed late, and brought dessert."

## 6. Assessment against the brief

- Specific, traceable: the proposition is an if/then statement with 9 verified quotations from 7 supporting episodes; every quotation links to an episode id.
- Pattern and exceptions: contradicting/exception episodes are listed separately (0 recorded on the final candidate; exceptions such as freely chosen public activities were used to reject or revise competing candidates).
- Predictive value: the retained assumption matched the recorded response (score ≥ 0.6) on 3 of 3 reserved episodes; compare with the generic baseline row in each table.
- Revises on contrary evidence: 2 recorded revisions, including the user's correction, with the previous wording preserved.
- Recognises when it cannot tell: ordinary episodes produced no candidates, and the interview stopped when further questions would not resolve anything.

Prediction supports usefulness; it does not prove a psychological root cause. Whether the interpretation *feels* revealing is a separate measure and is not assessed here.

