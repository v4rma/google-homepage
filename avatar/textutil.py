"""Text helpers: normalisation, quotation verification, tokenisation."""

from __future__ import annotations

import re
import secrets

_QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9']+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "to", "of", "in", "on", "at", "for", "with", "i", "me", "my",
    "we", "you", "he", "she", "it", "they", "them", "was", "were", "is", "are", "be", "been", "that", "this", "had",
    "have", "has", "do", "did", "not", "so", "as", "by", "from", "up", "out", "about", "just", "no", "yes", "said",
    "would", "could", "should", "when", "what", "there", "which", "who", "how", "very", "into", "than", "too", "also",
    "after", "before", "because", "like", "felt", "feel", "really", "some", "all", "one", "her", "his", "our", "its",
}


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


def normalize(text: str) -> str:
    text = (text or "").translate(_QUOTE_MAP).lower()
    return _WS.sub(" ", text).strip()


def strip_quotes(text: str) -> str:
    text = (text or "").strip()
    return text.strip("\"'“”‘’ ").strip()


def verify_quotation(quotation: str, sources: list[str]) -> bool:
    """True iff the quotation appears verbatim (modulo whitespace/quote style/case) in one of the sources."""
    q = normalize(strip_quotes(quotation))
    if len(q) < 3:
        return False
    return any(q in normalize(s) for s in sources if s)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def quoted_spans(text: str) -> list[str]:
    """Return strings the user placed inside quotation marks (words someone actually used)."""
    text = (text or "").translate(_QUOTE_MAP)
    spans = re.findall(r'"([^"]{3,200})"', text)
    return [s.strip() for s in spans]


def looks_like_dont_know(answer: str) -> bool:
    a = normalize(answer)
    if not a:
        return True
    patterns = ("don't know", "dont know", "not sure", "no idea", "can't tell", "cant tell", "can't say", "unsure", "i do not know")
    return any(p in a for p in patterns) and len(a) < 160
