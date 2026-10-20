"""Clearly synthetic demo dataset: "Sam", an invented person. No real individual's material is included.

The set contains one recurring pattern (irritation when someone else fixes a recurring schedule), its exceptions
(freely chosen but public; one-off requests), a situational confound (exhaustion) and ordinary/positive episodes,
so the model is not trained on a disproportionately troubled picture. Three episodes are reserved for the holdout
prediction test and are never shown to the hypothesis stages."""

from __future__ import annotations

DEMO_EPISODES: list[dict] = [
    {
        "id": "ep-demo01",
        "title": "Invitation to a weekly running club",
        "situation": "A friend texted: \"Want to join our Tuesday run? It's every week, same time.\" I was at home on a quiet evening.",
        "immediate_thought": "Now I have to do this every week.",
        "emotion_body": "Irritation. My jaw went tight and I noticed I was holding my breath.",
        "action_outcome": "I left the message unanswered for two days and then said I would see how things go. I have not gone.",
        "comparable_occasion": "I run on my own two or three times a week whenever I feel like it and I enjoy it.",
    },
    {
        "id": "ep-demo02",
        "title": "New daily standup announced",
        "situation": "My manager said in a meeting: \"From now on everyone joins the 9am standup every day.\"",
        "immediate_thought": "There goes my morning.",
        "emotion_body": "Irritation and a tight feeling across the chest.",
        "action_outcome": "I said fine. I have arrived late to about half of them since, which I feel slightly bad about.",
        "comparable_occasion": "I chose to attend an optional reading group on Thursdays last year and I was happy to be there.",
    },
    {
        "id": "ep-demo03",
        "title": "Sister's book club",
        "situation": "My sister asked: \"Can you commit to reading one book a month with us?\"",
        "immediate_thought": "I'll be trapped in this.",
        "emotion_body": "Resentment, a heavy feeling in my stomach.",
        "action_outcome": "I said maybe and never followed up. She has not mentioned it again.",
        "comparable_occasion": "",
    },
    {
        "id": "ep-demo04",
        "title": "Volunteered to present a side project",
        "situation": "I offered to present my side project to the whole team at the Friday demo. About twenty people watched.",
        "immediate_thought": "I hope this is useful to them.",
        "emotion_body": "Curious, a bit of adrenaline, pleased when it started.",
        "action_outcome": "I presented for fifteen minutes, took questions, and people were enthusiastic.",
        "comparable_occasion": "When I was assigned a monthly status report that nobody reads, it felt heavy and I kept putting it off.",
    },
    {
        "id": "ep-demo05",
        "title": "Landlord asks me to water plants daily",
        "situation": "My landlord asked: \"Could you water the plants in the hall every day for the next two weeks?\" Nobody would check.",
        "immediate_thought": "Every single day?",
        "emotion_body": "Irritation, a sigh, shoulders up.",
        "action_outcome": "I did it, but resented it each morning and skipped two days.",
        "comparable_occasion": "When a neighbour asked me to water her plants once while she was away, I was glad to do it.",
    },
    {
        "id": "ep-demo06",
        "title": "Meal-prep plan after a night shift",
        "situation": "After a night shift, my partner said: \"Let's plan weekly meal prep, same slot every Sunday.\"",
        "immediate_thought": "I cannot deal with this right now.",
        "emotion_body": "Exhausted, drained, then irritated.",
        "action_outcome": "I snapped that I did not want another schedule, then apologised an hour later.",
        "comparable_occasion": "On a rested weekend we planned meals together and I enjoyed it.",
    },
    {
        "id": "ep-demo07",
        "title": "A walk in the park",
        "situation": "I went for a walk in the park after lunch and saw a heron standing in the stream.",
        "immediate_thought": "That's lovely.",
        "emotion_body": "Calm, light, warm.",
        "action_outcome": "I stayed and watched it for twenty minutes and went back to work in a good mood.",
        "comparable_occasion": "",
    },
    {
        "id": "ep-demo08",
        "title": "Neighbour's sofa",
        "situation": "My neighbour knocked and asked: \"Could you give me a hand with this sofa?\"",
        "immediate_thought": "Sure, that's quick.",
        "emotion_body": "Warm, energetic.",
        "action_outcome": "We carried it up two flights and had a cup of tea afterwards.",
        "comparable_occasion": "",
    },
    {
        "id": "ep-demo09",
        "title": "Twelve-month gym contract",
        "situation": "A friend suggested we sign up together for a twelve-month gym contract to keep each other going.",
        "immediate_thought": "Twelve months. I'd be stuck.",
        "emotion_body": "Chest tightness, irritation that I tried to hide.",
        "action_outcome": "I said I would think about it and did not bring it up again.",
        "comparable_occasion": "I go to pay-as-you-go classes that I pick each week and I like that a lot.",
    },
    {
        "id": "ep-demo10",
        "title": "Detailed code review",
        "situation": "A colleague left thirty comments of feedback on my pull request, some of them quite blunt.",
        "immediate_thought": "Fair points, mostly.",
        "emotion_body": "Neutral, mildly pleased that someone read it carefully.",
        "action_outcome": "I fixed the issues the same afternoon and thanked them.",
        "comparable_occasion": "",
    },
    # ---- reserved for the holdout prediction test ----
    {
        "id": "ep-demo11",
        "title": "Fixed slot at the community garden",
        "situation": "The community garden coordinator asked: \"Could you take a fixed slot every Saturday morning?\"",
        "immediate_thought": "There goes every Saturday.",
        "emotion_body": "Irritation, tight shoulders.",
        "action_outcome": "I said I would rather drop in when I am free. She seemed a little disappointed.",
        "comparable_occasion": "I have enjoyed the drop-in sessions when I went on a whim.",
        "reserved": True,
    },
    {
        "id": "ep-demo12",
        "title": "One-off review of a friend's talk",
        "situation": "A friend asked whether I could review the slides for her talk once before Friday and give honest feedback.",
        "immediate_thought": "Happy to.",
        "emotion_body": "Calm, a little flattered.",
        "action_outcome": "I did it that evening and we talked it through for half an hour.",
        "comparable_occasion": "",
        "reserved": True,
    },
    {
        "id": "ep-demo13",
        "title": "Dinner invitation",
        "situation": "Two friends invited me to dinner on Friday at their place.",
        "immediate_thought": "Nice.",
        "emotion_body": "Glad, relaxed.",
        "action_outcome": "I went, stayed late, and brought dessert.",
        "comparable_occasion": "",
        "reserved": True,
    },
]


def synthetic_answer(question: str) -> str:
    """Answers the synthetic persona gives during the scripted milestone run. Chosen by what the question is about,
    so the script works with the mock and with a real model."""
    q = question.lower()
    if any(k in q for k in ("watched", "reviewed", "evaluat", "audience", "in front of", "presented", "judged")):
        return ("When I chose to present my side project it felt fine, even with the whole team watching. "
                "It is the fixed schedule that gets me, not the audience.")
    if any(k in q for k in ("rested", "tired", "slept", "exhaust", "night shift")):
        return "I was wrecked that day, but honestly the same thing happens when I have slept well."
    if any(k in q for k in ("mean", "at stake", "expect")):
        return ("I expected my weeks to fill up with things I had not chosen. "
                "It would have meant I am not in charge of my own time any more.")
    return "I don't know."


def seed(store, include_reserved: bool = True) -> list[dict]:
    created = []
    for ep in DEMO_EPISODES:
        if ep.get("reserved") and not include_reserved:
            continue
        if store.get_episode(ep["id"]):
            continue
        created.append(store.add_episode(
            situation=ep["situation"], immediate_thought=ep["immediate_thought"], emotion_body=ep["emotion_body"],
            action_outcome=ep["action_outcome"], comparable_occasion=ep.get("comparable_occasion", ""),
            title=ep["title"], reserved=bool(ep.get("reserved")), episode_id=ep["id"],
        ))
    return created
