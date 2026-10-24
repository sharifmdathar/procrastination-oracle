"""SQLite event log for Procrastination Oracle.

The database is the source of truth. Collectors (Todoist or synthetic) write
task snapshots here, ``features.py`` reads them back, and the agent records
each intervention it takes.

Timestamps are stored as ISO-8601 strings in UTC so that SQLite's lexical
ordering is also chronological ordering.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "oracle.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS task_events (
    task_id        TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    category       TEXT,
    created_at     TEXT NOT NULL,
    due_at         TEXT,
    deferred_count INTEGER NOT NULL DEFAULT 0,
    completed_at   TEXT,
    deleted_at     TEXT,
    source         TEXT NOT NULL DEFAULT 'unknown'
);

CREATE INDEX IF NOT EXISTS idx_task_events_created
    ON task_events (created_at);

CREATE TABLE IF NOT EXISTS interventions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_at     TEXT NOT NULL,
    risk       REAL NOT NULL,
    tier       TEXT NOT NULL,
    action     TEXT NOT NULL,
    body       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'llm'
);

CREATE INDEX IF NOT EXISTS idx_interventions_run
    ON interventions (run_at);

CREATE TABLE IF NOT EXISTS runs (
    run_at       TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    n_tasks      INTEGER NOT NULL,
    n_at_risk    INTEGER NOT NULL,
    synthetic    INTEGER NOT NULL DEFAULT 0
);
"""


def resolve_db_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit path, then ``ORACLE_DB``, then ``./oracle.db``."""
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get("ORACLE_DB")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_DB_PATH


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and initialise) the event log."""
    db_path = resolve_db_path(path)
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the schema if it does not exist yet."""
    conn.executescript(SCHEMA)
    conn.commit()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_as_of(value: str | None) -> datetime:
    """Parse an ISO-8601 ``as_of`` snapshot; ``None`` means now (UTC).

    Pinning this timestamp makes an evaluation run reproducible: the feature
    frame (overdue days, task age) and the label windows all hang off it.
    """
    if value is None:
        return utcnow()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_dt(value: str | datetime | None) -> datetime | None:
    """Parse an ISO timestamp; naive values are treated as UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class TaskEvent:
    """One row of the event log."""

    task_id: str
    title: str
    category: str | None
    created_at: datetime
    due_at: datetime | None = None
    deferred_count: int = 0
    completed_at: datetime | None = None
    deleted_at: datetime | None = None
    source: str = "unknown"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TaskEvent":
        created_at = parse_dt(row["created_at"])
        assert created_at is not None
        return cls(
            task_id=row["task_id"],
            title=row["title"],
            category=row["category"],
            created_at=created_at,
            due_at=parse_dt(row["due_at"]),
            deferred_count=int(row["deferred_count"] or 0),
            completed_at=parse_dt(row["completed_at"]),
            deleted_at=parse_dt(row["deleted_at"]),
            source=row["source"] or "unknown",
        )

    @property
    def is_open(self) -> bool:
        return self.completed_at is None and self.deleted_at is None


def upsert_events(
    conn: sqlite3.Connection,
    events: Iterable[TaskEvent],
    *,
    source: str | None = None,
) -> int:
    """Insert or replace task snapshots. Returns the number written."""
    rows = [
        (
            e.task_id,
            e.title,
            e.category,
            to_iso(e.created_at),
            to_iso(e.due_at),
            int(e.deferred_count),
            to_iso(e.completed_at),
            to_iso(e.deleted_at),
            source or e.source,
        )
        for e in events
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO task_events
            (task_id, title, category, created_at, due_at, deferred_count,
             completed_at, deleted_at, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            title          = excluded.title,
            category       = excluded.category,
            created_at     = excluded.created_at,
            due_at         = excluded.due_at,
            deferred_count = excluded.deferred_count,
            completed_at   = excluded.completed_at,
            deleted_at     = excluded.deleted_at,
            source         = excluded.source
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_events(conn: sqlite3.Connection) -> list[TaskEvent]:
    """All task snapshots, oldest first."""
    rows = conn.execute(
        "SELECT * FROM task_events ORDER BY created_at ASC, task_id ASC"
    ).fetchall()
    return [TaskEvent.from_row(r) for r in rows]


def count_events(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM task_events").fetchone()
    return int(row["n"]) if row else 0


def delete_synthetic(conn: sqlite3.Connection) -> int:
    """Remove generated rows, leaving real Todoist data untouched."""
    cur = conn.execute("DELETE FROM task_events WHERE source = 'synthetic'")
    conn.commit()
    return cur.rowcount


def sources(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT DISTINCT source FROM task_events").fetchall()
    return {str(r["source"]) for r in rows}


def record_intervention(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    risk: float,
    tier: str,
    action: str,
    body: str,
    run_at: datetime,
    source: str = "llm",
) -> None:
    """Persist one intervention from a nightly run."""
    conn.execute(
        """
        INSERT INTO interventions
            (task_id, run_at, risk, tier, action, body, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, to_iso(run_at), float(risk), tier, action, body, source),
    )
    conn.commit()


def record_run(
    conn: sqlite3.Connection,
    *,
    run_at: datetime,
    n_tasks: int,
    n_at_risk: int,
    label: str = "nightly",
    synthetic: bool = False,
) -> None:
    """Record one orchestration run for provenance."""
    conn.execute(
        """
        INSERT INTO runs (run_at, label, n_tasks, n_at_risk, synthetic)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_at) DO UPDATE SET
            n_tasks = excluded.n_tasks,
            n_at_risk = excluded.n_at_risk,
            synthetic = excluded.synthetic
        """,
        (to_iso(run_at), label, int(n_tasks), int(n_at_risk), int(synthetic)),
    )
    conn.commit()


def latest_run_at(conn: sqlite3.Connection) -> datetime | None:
    row = conn.execute("SELECT MAX(run_at) AS run_at FROM interventions").fetchone()
    return parse_dt(row["run_at"]) if row and row["run_at"] else None


def load_interventions(
    conn: sqlite3.Connection, run_at: datetime | None = None
) -> list[sqlite3.Row]:
    """Interventions from one run (defaults to the most recent run)."""
    run_at = run_at or latest_run_at(conn)
    if run_at is None:
        return []
    return conn.execute(
        "SELECT * FROM interventions WHERE run_at = ? ORDER BY risk DESC",
        (to_iso(run_at),),
    ).fetchall()


def reset(conn: sqlite3.Connection) -> None:
    """Drop all rows, keeping the schema. Intended for demos and tests."""
    conn.executescript(
        "DELETE FROM interventions; DELETE FROM runs; DELETE FROM task_events;"
    )
    conn.commit()


def rows_to_events(rows: Sequence[sqlite3.Row]) -> list[TaskEvent]:
    return [TaskEvent.from_row(r) for r in rows]
