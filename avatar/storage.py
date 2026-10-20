"""Repository layer over SQLite. All content handling lives here; no logging of content."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

from .db import connect
from .textutil import new_id, verify_quotation

ACTIVE_STATUSES = ("proposed", "under examination", "revised")
ALL_STATUSES = ACTIVE_STATUSES + ("rejected",)
ASSUMPTION_KINDS = ("if_then", "must", "situational")
EPISODE_CONTENT_FIELDS = ("situation", "immediate_thought", "emotion_body", "action_outcome", "comparable_occasion")

_JSON_FIELDS = {
    "assumptions": ("situations", "alternative_explanations", "weakeners", "user_corrections"),
    "interview_turns": ("discriminates_between",),
    "interpretations": ("content",),
    "revisions": ("previous",),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(table: str, row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for f in _JSON_FIELDS.get(table, ()):
        if f in d and isinstance(d[f], str):
            try:
                d[f] = json.loads(d[f])
            except json.JSONDecodeError:
                d[f] = []
    for f in ("reserved", "verified", "dont_know", "ok"):
        if f in d and d[f] is not None:
            d[f] = bool(d[f])
    return d


class ImmutableEpisodeError(Exception):
    pass


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = connect(path)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ episodes
    def add_episode(
        self,
        situation: str,
        immediate_thought: str,
        emotion_body: str,
        action_outcome: str,
        comparable_occasion: str = "",
        title: str = "",
        reserved: bool = False,
        episode_id: str | None = None,
    ) -> dict:
        eid = episode_id or new_id("ep")
        self.conn.execute(
            "INSERT INTO episodes (id, created_at, title, situation, immediate_thought, emotion_body, action_outcome,"
            " comparable_occasion, reserved) VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, _now(), title, situation, immediate_thought, emotion_body, action_outcome, comparable_occasion, int(reserved)),
        )
        self.conn.commit()
        return self.get_episode(eid)

    def get_episode(self, episode_id: str) -> dict | None:
        return _row("episodes", self.conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone())

    def list_episodes(self, include_reserved: bool = True) -> list[dict]:
        sql = "SELECT * FROM episodes" + ("" if include_reserved else " WHERE reserved=0") + " ORDER BY created_at, rowid"
        return [_row("episodes", r) for r in self.conn.execute(sql)]

    def set_reserved(self, episode_id: str, reserved: bool) -> dict:
        self.conn.execute("UPDATE episodes SET reserved=? WHERE id=?", (int(reserved), episode_id))
        self.conn.commit()
        return self.get_episode(episode_id)

    def update_episode_content(self, episode_id: str, **fields: str) -> None:
        """Always fails: exists only to make the immutability guarantee explicit and testable."""
        cols = ", ".join(f"{k}=?" for k in fields)
        try:
            self.conn.execute(f"UPDATE episodes SET {cols} WHERE id=?", (*fields.values(), episode_id))
        except sqlite3.IntegrityError as exc:
            raise ImmutableEpisodeError(str(exc)) from exc
        raise ImmutableEpisodeError("episode original account is immutable")

    def delete_episode(self, episode_id: str) -> None:
        self.conn.execute("DELETE FROM episodes WHERE id=?", (episode_id,))
        self.conn.commit()

    @staticmethod
    def episode_text(ep: dict) -> str:
        """The user's original account only (never interview material)."""
        return "\n".join(ep.get(f) or "" for f in EPISODE_CONTENT_FIELDS)

    def user_sources(self, episode_id: str) -> list[str]:
        """All user-authored text for an episode: the original account plus the user's own interview answers."""
        ep = self.get_episode(episode_id)
        if not ep:
            return []
        sources = [ep[f] for f in EPISODE_CONTENT_FIELDS if ep.get(f)]
        sources += [t["answer"] for t in self.list_turns(episode_id) if t.get("answer")]
        return sources

    # ------------------------------------------------------------------ interview turns
    def add_turn(self, episode_id: str, question: str, discriminates_between: Iterable[str] = (), rationale: str = "") -> dict:
        tid = new_id("q")
        self.conn.execute(
            "INSERT INTO interview_turns (id, episode_id, question, rationale, discriminates_between, asked_at) VALUES (?,?,?,?,?,?)",
            (tid, episode_id, question, rationale, json.dumps(list(discriminates_between)), _now()),
        )
        self.conn.commit()
        return self.get_turn(tid)

    def get_turn(self, turn_id: str) -> dict | None:
        return _row("interview_turns", self.conn.execute("SELECT * FROM interview_turns WHERE id=?", (turn_id,)).fetchone())

    def answer_turn(self, turn_id: str, answer: str, dont_know: bool) -> dict:
        self.conn.execute(
            "UPDATE interview_turns SET answer=?, answered_at=?, dont_know=? WHERE id=?", (answer, _now(), int(dont_know), turn_id)
        )
        self.conn.commit()
        return self.get_turn(turn_id)

    def pending_turn(self) -> dict | None:
        return _row(
            "interview_turns",
            self.conn.execute("SELECT * FROM interview_turns WHERE answer IS NULL ORDER BY asked_at DESC LIMIT 1").fetchone(),
        )

    def list_turns(self, episode_id: str | None = None) -> list[dict]:
        if episode_id:
            rows = self.conn.execute("SELECT * FROM interview_turns WHERE episode_id=? ORDER BY asked_at, rowid", (episode_id,))
        else:
            rows = self.conn.execute("SELECT * FROM interview_turns ORDER BY asked_at, rowid")
        return [_row("interview_turns", r) for r in rows]

    def delete_turn(self, turn_id: str) -> None:
        self.conn.execute("DELETE FROM interview_turns WHERE id=?", (turn_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ observations
    def add_observation(self, episode_id: str, kind: str, content: str, source: str, verified: bool, turn_id: str | None = None) -> dict:
        oid = new_id("ob")
        self.conn.execute(
            "INSERT INTO observations (id, episode_id, turn_id, kind, content, source, verified, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (oid, episode_id, turn_id, kind, content, source, int(verified), _now()),
        )
        self.conn.commit()
        return _row("observations", self.conn.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone())

    def list_observations(self, episode_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM observations WHERE episode_id=? ORDER BY created_at, rowid", (episode_id,))
        return [_row("observations", r) for r in rows]

    def clear_observations(self, episode_id: str, source: str) -> None:
        self.conn.execute("DELETE FROM observations WHERE episode_id=? AND source=?", (episode_id, source))
        self.conn.commit()

    # ------------------------------------------------------------------ assumptions
    def add_assumption(
        self,
        proposition: str,
        kind: str = "if_then",
        situations: Iterable[str] = (),
        alternative_explanations: Iterable[str] = (),
        prediction_new_situation: str = "",
        weakeners: Iterable[str] = (),
        status: str = "proposed",
        origin_episode_id: str | None = None,
    ) -> dict:
        if kind not in ASSUMPTION_KINDS:
            kind = "if_then"
        if status not in ALL_STATUSES:
            status = "proposed"
        aid = new_id("as")
        now = _now()
        self.conn.execute(
            "INSERT INTO assumptions (id, proposition, kind, situations, alternative_explanations, prediction_new_situation,"
            " weakeners, user_corrections, status, version, origin_episode_id, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                aid, proposition, kind, json.dumps(list(situations)), json.dumps(list(alternative_explanations)),
                prediction_new_situation, json.dumps(list(weakeners)), "[]", status, 1, origin_episode_id, now, now,
            ),
        )
        self.conn.commit()
        return self.get_assumption(aid)

    def get_assumption(self, assumption_id: str, with_details: bool = True) -> dict | None:
        a = _row("assumptions", self.conn.execute("SELECT * FROM assumptions WHERE id=?", (assumption_id,)).fetchone())
        if a and with_details:
            ev = self.list_evidence(assumption_id)
            a["supporting"] = [e for e in ev if e["role"] == "supporting"]
            a["contradicting"] = [e for e in ev if e["role"] == "contradicting"]
            a["supporting_episode_ids"] = sorted({e["episode_id"] for e in a["supporting"] if e["verified"]})
            a["contradicting_episode_ids"] = sorted({e["episode_id"] for e in a["contradicting"]})
            a["revisions"] = self.list_revisions(assumption_id)
        return a

    def list_assumptions(self, statuses: Iterable[str] | None = None, with_details: bool = True) -> list[dict]:
        if statuses:
            st = list(statuses)
            rows = self.conn.execute(
                f"SELECT id FROM assumptions WHERE status IN ({','.join('?' * len(st))}) ORDER BY created_at, rowid", st
            )
        else:
            rows = self.conn.execute("SELECT id FROM assumptions ORDER BY created_at, rowid")
        return [self.get_assumption(r["id"], with_details=with_details) for r in rows]

    def _snapshot(self, assumption_id: str) -> dict:
        a = self.get_assumption(assumption_id, with_details=False)
        return {k: a[k] for k in ("proposition", "kind", "situations", "alternative_explanations", "prediction_new_situation", "weakeners", "status", "version")}

    def revise_assumption(self, assumption_id: str, changes: dict[str, Any], trigger: str, reason: str = "") -> dict:
        """Apply changes, recording the previous state as a revision and bumping the version."""
        prev = self._snapshot(assumption_id)
        allowed = {"proposition", "kind", "situations", "alternative_explanations", "prediction_new_situation", "weakeners", "status"}
        changes = {k: v for k, v in changes.items() if k in allowed and v is not None}
        if "status" in changes and changes["status"] not in ALL_STATUSES:
            del changes["status"]
        if "kind" in changes and changes["kind"] not in ASSUMPTION_KINDS:
            del changes["kind"]
        if not changes:
            return self.get_assumption(assumption_id)
        sets, vals = [], []
        for k, v in changes.items():
            sets.append(f"{k}=?")
            vals.append(json.dumps(list(v)) if k in ("situations", "alternative_explanations", "weakeners") else v)
        new_version = prev["version"] + 1
        self.conn.execute(
            "INSERT INTO revisions (id, assumption_id, version, trigger, reason, previous, created_at) VALUES (?,?,?,?,?,?,?)",
            (new_id("rv"), assumption_id, prev["version"], trigger, reason, json.dumps(prev), _now()),
        )
        self.conn.execute(
            f"UPDATE assumptions SET {', '.join(sets)}, version=?, updated_at=? WHERE id=?", (*vals, new_version, _now(), assumption_id)
        )
        self.conn.commit()
        return self.get_assumption(assumption_id)

    def add_correction(self, assumption_id: str, text: str) -> dict:
        a = self.get_assumption(assumption_id, with_details=False)
        corrections = list(a["user_corrections"]) + [{"text": text, "at": _now(), "version": a["version"]}]
        self.conn.execute("UPDATE assumptions SET user_corrections=? WHERE id=?", (json.dumps(corrections), assumption_id))
        self.conn.commit()
        return self.get_assumption(assumption_id)

    def set_status(self, assumption_id: str, status: str, trigger: str, reason: str = "") -> dict:
        return self.revise_assumption(assumption_id, {"status": status}, trigger=trigger, reason=reason)

    def delete_assumption(self, assumption_id: str) -> None:
        self.conn.execute("DELETE FROM assumptions WHERE id=?", (assumption_id,))
        self.conn.commit()

    def merge_lists(self, assumption_id: str, field: str, items: Iterable[str], trigger: str, reason: str = "") -> dict:
        a = self.get_assumption(assumption_id, with_details=False)
        current = list(a[field])
        added = [i for i in items if i and i not in current]
        if not added:
            return self.get_assumption(assumption_id)
        return self.revise_assumption(assumption_id, {field: current + added}, trigger=trigger, reason=reason)

    # ------------------------------------------------------------------ evidence
    def add_evidence(
        self, assumption_id: str, episode_id: str, quotation: str, role: str, note: str = "", turn_id: str | None = None
    ) -> dict | None:
        """Link an episode to an assumption. The quotation is verified against the user's own words; an unverified
        quotation is stored but flagged and never counted. Episodes unknown to the store are refused."""
        if role not in ("supporting", "contradicting") or not self.get_episode(episode_id):
            return None
        verified = verify_quotation(quotation, self.user_sources(episode_id))
        eid = new_id("ev")
        try:
            self.conn.execute(
                "INSERT INTO assumption_evidence (id, assumption_id, episode_id, turn_id, role, quotation, note, verified, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, assumption_id, episode_id, turn_id, role, quotation.strip(), note, int(verified), _now()),
            )
        except sqlite3.IntegrityError:
            return None
        self.conn.commit()
        return _row("assumption_evidence", self.conn.execute("SELECT * FROM assumption_evidence WHERE id=?", (eid,)).fetchone())

    def list_evidence(self, assumption_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM assumption_evidence WHERE assumption_id=? ORDER BY created_at, rowid", (assumption_id,))
        return [_row("assumption_evidence", r) for r in rows]

    def delete_evidence(self, evidence_id: str) -> None:
        self.conn.execute("DELETE FROM assumption_evidence WHERE id=?", (evidence_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ interpretations
    def add_interpretation(self, stage: str, content: dict, assumption_id: str | None = None, episode_id: str | None = None) -> dict:
        iid = new_id("in")
        self.conn.execute(
            "INSERT INTO interpretations (id, stage, assumption_id, episode_id, content, created_at) VALUES (?,?,?,?,?,?)",
            (iid, stage, assumption_id, episode_id, json.dumps(content), _now()),
        )
        self.conn.commit()
        return _row("interpretations", self.conn.execute("SELECT * FROM interpretations WHERE id=?", (iid,)).fetchone())

    def list_interpretations(self, assumption_id: str | None = None, episode_id: str | None = None) -> list[dict]:
        clauses, vals = [], []
        if assumption_id:
            clauses.append("assumption_id=?")
            vals.append(assumption_id)
        if episode_id:
            clauses.append("episode_id=?")
            vals.append(episode_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM interpretations{where} ORDER BY created_at, rowid", vals)
        return [_row("interpretations", r) for r in rows]

    # ------------------------------------------------------------------ revisions
    def list_revisions(self, assumption_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM revisions WHERE assumption_id=? ORDER BY version, created_at", (assumption_id,))
        return [_row("revisions", r) for r in rows]

    # ------------------------------------------------------------------ predictions
    def add_prediction(self, episode_id: str, batch_id: str, assumption_id: str | None, **fields: Any) -> dict:
        pid = new_id("pr")
        cols = ["predicted_thought", "predicted_emotion", "predicted_action", "confidence", "rationale", "match_score", "comparison"]
        vals = [fields.get(c, "" if c not in ("confidence", "match_score") else None) for c in cols]
        self.conn.execute(
            f"INSERT INTO predictions (id, episode_id, assumption_id, {', '.join(cols)}, batch_id, created_at)"
            f" VALUES (?,?,?,{','.join('?' * len(cols))},?,?)",
            (pid, episode_id, assumption_id, *vals, batch_id, _now()),
        )
        self.conn.commit()
        return _row("predictions", self.conn.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone())

    def rate_prediction(self, prediction_id: str, rating: str) -> dict | None:
        self.conn.execute("UPDATE predictions SET user_rating=? WHERE id=?", (rating, prediction_id))
        self.conn.commit()
        return _row("predictions", self.conn.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone())

    def list_predictions(self, episode_id: str | None = None) -> list[dict]:
        if episode_id:
            rows = self.conn.execute("SELECT * FROM predictions WHERE episode_id=? ORDER BY created_at, rowid", (episode_id,))
        else:
            rows = self.conn.execute("SELECT * FROM predictions ORDER BY created_at, rowid")
        return [_row("predictions", r) for r in rows]

    # ------------------------------------------------------------------ runs
    def start_run(self, stage: str, episode_id: str | None = None) -> str:
        rid = new_id("run")
        self.conn.execute("INSERT INTO pipeline_runs (id, stage, episode_id, started_at) VALUES (?,?,?,?)", (rid, stage, episode_id, _now()))
        self.conn.commit()
        return rid

    def finish_run(self, run_id: str, ok: bool, error_type: str | None = None) -> None:
        self.conn.execute("UPDATE pipeline_runs SET finished_at=?, ok=?, error_type=? WHERE id=?", (_now(), int(ok), error_type, run_id))
        self.conn.commit()

    def list_runs(self) -> list[dict]:
        return [_row("pipeline_runs", r) for r in self.conn.execute("SELECT * FROM pipeline_runs ORDER BY started_at")]

    def counts(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "episodes": q("SELECT COUNT(*) FROM episodes"),
            "reserved_episodes": q("SELECT COUNT(*) FROM episodes WHERE reserved=1"),
            "assumptions": q("SELECT COUNT(*) FROM assumptions"),
            "active_assumptions": q("SELECT COUNT(*) FROM assumptions WHERE status != 'rejected'"),
            "observations": q("SELECT COUNT(*) FROM observations"),
            "interpretations": q("SELECT COUNT(*) FROM interpretations"),
            "interview_turns": q("SELECT COUNT(*) FROM interview_turns"),
        }
