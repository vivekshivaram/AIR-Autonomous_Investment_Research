"""Persistent SQLite memory: compact lessons from earlier runs and a record of every run.

Before planning, the orchestrator calls ``retrieve_lessons(symbol, question)``; the planner applies the
lessons' machine-readable hints (``require_subagents``, ``extra_criteria``), and the report writer reads
``report_guidance`` hints. After the run, reflection stores new lessons with ``store_lesson`` and the run
itself with ``record_run``.

Memory is guidance, never evidence. Lesson IDs (MEM-0001) are not evidence IDs, so a report that cites a
lesson fails the evaluator's grounding check, and nothing from memory enters the evidence pack.

The database lives at ``Settings.db_path`` (``agent_memory.db``, gitignored). ``AGENT_MEMORY_DB`` or
``use_database(path)`` points it elsewhere, which the notebook and the tests use for a private file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS, utc_now_iso
from .planner import interpret_question
from .tool_registry import default_subagent_registry

RETRIEVE_LIMIT = 5  # the planner reads at most five lessons (planner.MAX_LESSONS)
MAX_LESSONS_PER_SYMBOL = 20  # older, unused lessons beyond this are deactivated to keep memory compact
TOPIC_BY_AGENT = {SUBAGENT_MARKET: "price", SUBAGENT_EARNINGS: "financials", SUBAGENT_NEWS: "news"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    memory_id       TEXT PRIMARY KEY,
    fingerprint     TEXT NOT NULL UNIQUE,
    scope           TEXT NOT NULL CHECK (scope IN ('general', 'symbol')),
    symbol          TEXT,
    tags            TEXT NOT NULL DEFAULT '[]',
    lesson          TEXT NOT NULL,
    hint            TEXT NOT NULL DEFAULT '{}',
    trigger         TEXT NOT NULL DEFAULT '',
    source_run_id   TEXT,
    created_at      TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    last_used_at    TEXT,
    times_seen      INTEGER NOT NULL DEFAULT 1,
    times_retrieved INTEGER NOT NULL DEFAULT 0,
    times_applied   INTEGER NOT NULL DEFAULT 0,
    active          INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    symbol             TEXT NOT NULL,
    question           TEXT NOT NULL,
    scope              TEXT,
    selected_subagents TEXT NOT NULL DEFAULT '[]',
    initial_score      REAL,
    final_score        REAL,
    refinement_cycles  INTEGER,
    stop_reason        TEXT,
    llm                TEXT,
    lessons_retrieved  TEXT NOT NULL DEFAULT '[]',
    lessons_applied    TEXT NOT NULL DEFAULT '[]',
    reflection         TEXT NOT NULL DEFAULT '{}',
    started_at         TEXT,
    recorded_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lesson_usage (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    event     TEXT NOT NULL CHECK (event IN ('retrieved', 'applied')),
    at        TEXT NOT NULL
);
"""


def question_topics(question: str) -> list[str]:
    """Topics a question touches (price, financials, news, comprehensive), using the planner's own keywords."""
    profile = interpret_question(question, default_subagent_registry())
    topics = {TOPIC_BY_AGENT[agent] for agent in profile.matches if agent in TOPIC_BY_AGENT}
    if profile.scope == "comprehensive" and not profile.broad_fallback:
        topics.add("comprehensive")
    return sorted(topics)


class MemoryStore:
    """Lessons and run records in one SQLite file. Every call opens and closes its own connection."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return _Transaction(connection)

    # ---- lessons ------------------------------------------------------------------------------

    def store_lesson(
        self,
        lesson: str,
        *,
        scope: str = "general",
        symbol: str | None = None,
        tags: Iterable[str] = (),
        hint: dict[str, Any] | None = None,
        trigger: str = "",
        source_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Save a lesson, or refresh the existing copy when the same lesson was learned before."""
        text = " ".join(str(lesson).split())
        if not text:
            raise ValueError("lesson text must not be empty")
        scope = "symbol" if scope == "symbol" and symbol else "general"
        symbol = symbol.upper() if scope == "symbol" and symbol else None
        fingerprint = hashlib.sha1(f"{scope}|{symbol or ''}|{text.lower()}".encode()).hexdigest()
        now = utc_now_iso()
        with self._db() as db:
            row = db.execute("SELECT memory_id FROM lessons WHERE fingerprint = ?", (fingerprint,)).fetchone()
            if row:
                db.execute(
                    "UPDATE lessons SET times_seen = times_seen + 1, last_seen_at = ?, hint = ?, tags = ?, active = 1 WHERE memory_id = ?",
                    (now, json.dumps(hint or {}), json.dumps(sorted(set(tags))), row["memory_id"]),
                )
                memory_id = row["memory_id"]
            else:
                number = db.execute("SELECT COALESCE(MAX(CAST(SUBSTR(memory_id, 5) AS INTEGER)), 0) + 1 FROM lessons").fetchone()[0]
                memory_id = f"MEM-{number:04d}"
                db.execute(
                    "INSERT INTO lessons (memory_id, fingerprint, scope, symbol, tags, lesson, hint, trigger, source_run_id, created_at, last_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (memory_id, fingerprint, scope, symbol, json.dumps(sorted(set(tags))), text, json.dumps(hint or {}), trigger, source_run_id, now, now),
                )
            if symbol:
                self._compact(db, symbol)
        return self.get_lesson(memory_id)

    def _compact(self, db: sqlite3.Connection, symbol: str) -> None:
        """Keep at most MAX_LESSONS_PER_SYMBOL active lessons per symbol; the least used and oldest go first."""
        rows = db.execute(
            "SELECT memory_id FROM lessons WHERE symbol = ? AND active = 1 ORDER BY times_applied DESC, last_seen_at DESC",
            (symbol,),
        ).fetchall()
        for row in rows[MAX_LESSONS_PER_SYMBOL:]:
            db.execute("UPDATE lessons SET active = 0 WHERE memory_id = ?", (row["memory_id"],))

    def get_lesson(self, memory_id: str) -> dict[str, Any]:
        """One lesson as a dict, with tags and hint decoded."""
        with self._db() as db:
            row = db.execute("SELECT * FROM lessons WHERE memory_id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return _lesson_dict(row)

    def retrieve_lessons(self, symbol: str, question: str, limit: int = RETRIEVE_LIMIT) -> list[dict[str, Any]]:
        """The most relevant active lessons for this symbol and question, in the shape ``MemoryItem`` reads.

        Ranking: lessons about this symbol come before general ones, then lessons whose topics overlap the
        question, then recent ones. A lesson that adds a specialist is returned only when the question
        shares one of its topics, so a price lesson cannot add news to an earnings-only question.
        """
        symbol = (symbol or "").upper()
        topics = set(question_topics(question))
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM lessons WHERE active = 1 AND (scope = 'general' OR symbol = ?)", (symbol,)
            ).fetchall()
            ranked = []
            for row in rows:
                lesson = _lesson_dict(row)
                tags = set(lesson["tags"])
                if lesson["hint"].get("require_subagents") and tags and not (tags & topics):
                    continue
                overlap = len(tags & topics) / len(tags) if tags else 0.5
                age_days = max((_utc_now() - _parse_time(lesson["created_at"])).total_seconds() / 86400, 0.0)
                score = (0.6 if lesson["scope"] == "symbol" else 0.3) + 0.3 * overlap + 0.1 * math.exp(-age_days / 30)
                lesson["score"] = round(min(score, 1.0), 3)
                ranked.append(lesson)
            ranked.sort(key=lambda item: (-item["score"], item["memory_id"]))
            chosen = ranked[:limit]
            now = utc_now_iso()
            for lesson in chosen:
                db.execute(
                    "UPDATE lessons SET times_retrieved = times_retrieved + 1, last_used_at = ? WHERE memory_id = ?",
                    (now, lesson["memory_id"]),
                )
        return [
            {k: lesson[k] for k in ("memory_id", "lesson", "scope", "symbol", "score", "hint", "tags")}
            for lesson in chosen
        ]

    def lessons(self, active_only: bool = False) -> list[dict[str, Any]]:
        """Every stored lesson, newest first."""
        query = "SELECT * FROM lessons" + (" WHERE active = 1" if active_only else "") + " ORDER BY memory_id DESC"
        with self._db() as db:
            return [_lesson_dict(row) for row in db.execute(query).fetchall()]

    # ---- runs ---------------------------------------------------------------------------------

    def record_run(
        self,
        *,
        run_id: str,
        symbol: str,
        question: str,
        scope: str = "",
        selected_subagents: Iterable[str] = (),
        initial_score: float | None = None,
        final_score: float | None = None,
        refinement_cycles: int = 0,
        stop_reason: str = "",
        llm: str = "",
        lessons_retrieved: Iterable[str] = (),
        lessons_applied: Iterable[str] = (),
        reflection: dict[str, Any] | None = None,
        started_at: str | None = None,
    ) -> None:
        """Save one run and mark which lessons it retrieved and applied."""
        retrieved, applied = list(dict.fromkeys(lessons_retrieved)), list(dict.fromkeys(lessons_applied))
        now = utc_now_iso()
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO runs (run_id, symbol, question, scope, selected_subagents, initial_score, final_score, "
                "refinement_cycles, stop_reason, llm, lessons_retrieved, lessons_applied, reflection, started_at, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, symbol.upper(), question, scope, json.dumps(list(selected_subagents)), initial_score, final_score,
                    refinement_cycles, stop_reason, llm, json.dumps(retrieved), json.dumps(applied),
                    json.dumps(reflection or {}, default=str), started_at, now,
                ),
            )
            for event, ids in (("retrieved", retrieved), ("applied", applied)):
                for memory_id in ids:
                    db.execute("INSERT INTO lesson_usage (run_id, memory_id, event, at) VALUES (?, ?, ?, ?)", (run_id, memory_id, event, now))
            for memory_id in applied:
                db.execute("UPDATE lessons SET times_applied = times_applied + 1, last_used_at = ? WHERE memory_id = ?", (now, memory_id))

    def runs(self) -> list[dict[str, Any]]:
        """Every recorded run, oldest first, with JSON columns decoded."""
        with self._db() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY recorded_at, rowid").fetchall()
        decoded = []
        for row in rows:
            item = dict(row)
            for column in ("selected_subagents", "lessons_retrieved", "lessons_applied", "reflection"):
                item[column] = json.loads(item[column] or "null")
            decoded.append(item)
        return decoded

    def usage(self) -> list[dict[str, Any]]:
        """Lesson retrieval and application events, in order."""
        with self._db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM lesson_usage ORDER BY id").fetchall()]

    def reset(self) -> None:
        """Delete every lesson and run (used by the notebook so each execution starts from empty memory)."""
        with self._db() as db:
            for table in ("lesson_usage", "runs", "lessons"):
                db.execute(f"DELETE FROM {table}")


class _Transaction:
    """Context manager that commits on success, rolls back on error and always closes the connection."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self.connection

    def __exit__(self, exc_type, exc, tb) -> None:
        with closing(self.connection):
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()


def _lesson_dict(row: sqlite3.Row) -> dict[str, Any]:
    lesson = dict(row)
    lesson["tags"] = json.loads(lesson.get("tags") or "[]")
    lesson["hint"] = json.loads(lesson.get("hint") or "{}")
    lesson["active"] = bool(lesson.get("active", 1))
    lesson.pop("fingerprint", None)
    return lesson


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return _utc_now()
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---- module-level API (what bootstrap registers as the memory tools) ------------------------------

_store: MemoryStore | None = None


def default_db_path() -> Path:
    """AGENT_MEMORY_DB when set, otherwise ``Settings.db_path``."""
    return Path(os.getenv("AGENT_MEMORY_DB") or Settings().db_path)


def get_store() -> MemoryStore:
    """The store the module-level functions use, created on first use."""
    global _store
    if _store is None:
        _store = MemoryStore(default_db_path())
    return _store


def use_database(path: str | Path, reset: bool = False) -> MemoryStore:
    """Point the module-level functions at another file, optionally emptying it first."""
    global _store
    _store = MemoryStore(path)
    if reset:
        _store.reset()
    return _store


def retrieve_lessons(symbol: str, question: str) -> list[dict[str, Any]]:
    """Memory retrieval tool: relevant lessons for planning. Guidance only, never evidence."""
    return get_store().retrieve_lessons(symbol, question)


def store_lesson(lesson: str, **kwargs: Any) -> dict[str, Any]:
    """Memory storage tool: save one lesson (see ``MemoryStore.store_lesson`` for the arguments)."""
    return get_store().store_lesson(lesson, **kwargs)


def record_run(**run: Any) -> None:
    """Save one run record (see ``MemoryStore.record_run``)."""
    get_store().record_run(**run)
