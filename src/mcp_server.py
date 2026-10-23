"""FastMCP server for Claude Desktop and other MCP clients.

Two tools:

* ``predict_task_risk(tasks_csv)`` — score a CSV of open tasks with TabPFN-3.5.
* ``get_my_patterns()`` — summarize the local event log's completion patterns.

The server is read-only with respect to tasks; it never mutates Todoist. Run it
over stdio:

    python src/mcp_server.py

or register it with Claude Desktop, e.g.::

    "mcpServers": {
      "procrastination-oracle": {
        "command": "python",
        "args": ["/absolute/path/to/src/mcp_server.py"]
      }
    }
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from fastmcp import FastMCP  # noqa: E402

import db  # noqa: E402
import features  # noqa: E402
import oracle  # noqa: E402
from collectors import synthetic  # noqa: E402

load_dotenv(ROOT / ".env")

mcp = FastMCP(
    "procrastination-oracle",
    instructions=(
        "Local-first procrastination predictions. Call predict_task_risk with a "
        "CSV of open tasks to get a 48h skip risk, or get_my_patterns to see "
        "when this person actually finishes work. Predictions are probabilities, "
        "not instructions."
    ),
)


def model_path() -> Path:
    return Path(os.environ.get("ORACLE_MODEL", str(oracle.DEFAULT_MODEL_PATH)))


def _load_model() -> oracle.TaskRiskOracle:
    return oracle.load_oracle(model_path())


def _task_events(tasks_csv: str, as_of: datetime) -> list[db.TaskEvent]:
    reader = csv.DictReader(io.StringIO(tasks_csv.strip()))
    if reader.fieldnames is None:
        raise ValueError("tasks_csv needs a header row: title,category,due_at,...")
    if "title" not in reader.fieldnames:
        raise ValueError("tasks_csv needs a 'title' column")

    events: list[db.TaskEvent] = []
    for index, row in enumerate(reader, 1):
        title = (row.get("title") or "").strip()
        if not title:
            continue
        created = db.parse_dt(row.get("created_at")) or as_of
        due = db.parse_dt(row.get("due_at"))
        try:
            deferred = int(row.get("deferred_count") or 0)
        except ValueError:
            deferred = 0
        events.append(
            db.TaskEvent(
                task_id=(row.get("task_id") or f"csv-{index:03d}").strip(),
                title=title,
                category=(row.get("category") or "uncategorized").strip(),
                created_at=created,
                due_at=due,
                deferred_count=max(0, deferred),
                source="mcp",
            )
        )
    if not events:
        raise ValueError("no tasks found in tasks_csv")
    return events


@mcp.tool
def predict_task_risk(tasks_csv: str) -> str:
    """Predict the probability each task will be skipped for 48 hours.

    Args:
        tasks_csv: CSV text with a header row. Required column: ``title``.
            Optional: ``category``, ``due_at`` (ISO date or datetime),
            ``created_at`` (ISO), ``deferred_count``, ``task_id``.

    Returns:
        JSON with one prediction per task: risk (0-1), tier (HIGH/MED/LOW),
        days_until_due, and the model version.
    """
    as_of = db.utcnow()
    try:
        events = _task_events(tasks_csv, as_of)
        model = _load_model()
        scored = oracle.score_tasks(model, events, as_of=as_of)
    except FileNotFoundError as exc:
        return json.dumps(
            {
                "error": "no fitted model",
                "hint": "Run `python scripts/train_and_eval.py` first, then set ORACLE_MODEL if needed.",
                "detail": str(exc),
            },
            indent=2,
        )
    except (ValueError, oracle.TabPFNUnavailable) as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    order = {event.task_id: i for i, event in enumerate(events)}
    payload = {
        "as_of": db.to_iso(as_of),
        "model": oracle.MODEL_DISPLAY_NAME,
        "predictions": [
            {
                "task_id": item.task_id,
                "title": item.title,
                "risk": round(item.risk, 4),
                "tier": item.tier,
                "days_until_due": (
                    round(item.days_until_due, 2)
                    if item.days_until_due is not None
                    else None
                ),
                "deferred_count": item.deferred_count,
            }
            for item in sorted(scored, key=lambda s: order.get(s.task_id, 0))
        ],
        "note": "Risk is P(not completed within 48h of creation).",
    }
    return json.dumps(payload, indent=2)


@mcp.tool
def get_my_patterns() -> str:
    """Summarize this person's completion patterns from the local event log.

    Returns:
        JSON with 48h completion rate, per-category and per-weekday rates,
        the most-deferred tasks, and the best completion slot. Includes a
        ``synthetic`` flag when the log is demo data.
    """
    try:
        conn = db.connect()
        try:
            events = db.load_events(conn)
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - filesystem edge case
        return json.dumps({"error": f"could not read the event log: {exc}"}, indent=2)

    patterns = features.summarize_patterns(events)
    patterns["synthetic"] = any(e.source == synthetic.SYNTHETIC_SOURCE for e in events)
    patterns["n_events"] = len(events)
    return json.dumps(patterns, indent=2)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
