"""SQLite schema. Observations (user-provided material and what was extracted from it) and interpretations
(anything the model generated) live in separate tables, and the user's original account is immutable."""

from __future__ import annotations

import os
import sqlite3

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    situation TEXT NOT NULL,
    immediate_thought TEXT NOT NULL,
    emotion_body TEXT NOT NULL,
    action_outcome TEXT NOT NULL,
    comparable_occasion TEXT NOT NULL DEFAULT '',
    reserved INTEGER NOT NULL DEFAULT 0
);

-- The original account must never be edited. Only the holdout flag may change.
CREATE TRIGGER IF NOT EXISTS episodes_immutable
BEFORE UPDATE OF created_at, title, situation, immediate_thought, emotion_body, action_outcome, comparable_occasion
ON episodes
BEGIN
    SELECT RAISE(ABORT, 'episode original account is immutable');
END;

CREATE TABLE IF NOT EXISTS interview_turns (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    discriminates_between TEXT NOT NULL DEFAULT '[]',
    asked_at TEXT NOT NULL,
    answer TEXT,
    answered_at TEXT,
    dont_know INTEGER NOT NULL DEFAULT 0
);

-- Observations: what the user reported (events, exact quotations, expectations, emotions, actions)
-- and what is missing. Extracted from user text; every quotation is checked against the source text.
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    turn_id TEXT REFERENCES interview_turns(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assumptions (
    id TEXT PRIMARY KEY,
    proposition TEXT NOT NULL,
    kind TEXT NOT NULL,
    situations TEXT NOT NULL DEFAULT '[]',
    alternative_explanations TEXT NOT NULL DEFAULT '[]',
    prediction_new_situation TEXT NOT NULL DEFAULT '',
    weakeners TEXT NOT NULL DEFAULT '[]',
    user_corrections TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'proposed',
    version INTEGER NOT NULL DEFAULT 1,
    origin_episode_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Evidence links: supporting or contradicting episodes with exact quotations. Only verified quotations
-- (verbatim from the user's account or answers) count as evidence.
CREATE TABLE IF NOT EXISTS assumption_evidence (
    id TEXT PRIMARY KEY,
    assumption_id TEXT NOT NULL REFERENCES assumptions(id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    turn_id TEXT REFERENCES interview_turns(id) ON DELETE SET NULL,
    role TEXT NOT NULL,
    quotation TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(assumption_id, episode_id, role, quotation)
);

-- Interpretations: everything the model generated (hypothesis rationale, critiques, updates).
-- Never used as evidence.
CREATE TABLE IF NOT EXISTS interpretations (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    assumption_id TEXT REFERENCES assumptions(id) ON DELETE CASCADE,
    episode_id TEXT REFERENCES episodes(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS revisions (
    id TEXT PRIMARY KEY,
    assumption_id TEXT NOT NULL REFERENCES assumptions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    trigger TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    previous TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    assumption_id TEXT REFERENCES assumptions(id) ON DELETE CASCADE,
    predicted_thought TEXT NOT NULL DEFAULT '',
    predicted_emotion TEXT NOT NULL DEFAULT '',
    predicted_action TEXT NOT NULL DEFAULT '',
    confidence REAL,
    rationale TEXT NOT NULL DEFAULT '',
    match_score REAL,
    comparison TEXT NOT NULL DEFAULT '',
    user_rating TEXT,
    batch_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Pipeline run bookkeeping: stage, ids, timing and error type only (no content).
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    episode_id TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    ok INTEGER,
    error_type TEXT
);
"""


def connect(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
