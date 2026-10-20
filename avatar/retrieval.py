"""Keyword retrieval (BM25) of relevant past episodes. Reserved (holdout) episodes are never retrieved."""

from __future__ import annotations

import math
from collections import Counter

from .storage import Store
from .textutil import tokenize


def bm25_rank(query: str, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
    q_terms = tokenize(query)
    if not q_terms or not docs:
        return []
    tokenized = {doc_id: tokenize(text) for doc_id, text in docs.items()}
    n = len(tokenized)
    avgdl = sum(len(t) for t in tokenized.values()) / n
    df: Counter[str] = Counter()
    for toks in tokenized.values():
        for term in set(toks):
            df[term] += 1
    scores: list[tuple[str, float]] = []
    for doc_id, toks in tokenized.items():
        tf = Counter(toks)
        dl = len(toks) or 1
        score = 0.0
        for term in set(q_terms):
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            scores.append((doc_id, score))
    scores.sort(key=lambda s: (-s[1], s[0]))
    return scores


def retrieve_related_episodes(store: Store, episode_id: str, k: int = 4, query: str | None = None) -> list[dict]:
    """Return up to k non-reserved episodes most similar to the given episode (or an explicit query)."""
    current = store.get_episode(episode_id)
    candidates = {ep["id"]: store.episode_text(ep) for ep in store.list_episodes(include_reserved=False) if ep["id"] != episode_id}
    q = query if query is not None else (store.episode_text(current) if current else "")
    ranked = bm25_rank(q, candidates)
    return [store.get_episode(doc_id) for doc_id, _ in ranked[:k]]
