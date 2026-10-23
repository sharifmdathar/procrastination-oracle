"""Todoist → SQLite collector (real data).

Uses the Todoist REST API directly through ``urllib`` so the demo has no extra
HTTP dependency. Set ``TODOIST_API_TOKEN`` in ``.env`` or pass a token.

Known limitations, kept explicit instead of invented:

* Todoist does not expose a "times deferred" counter, so ``deferred_count`` is
  0 for freshly imported tasks. The feature is still used for anything already
  in the log.
* All-day due dates become 17:00 UTC (Todoist's API omits the user's timezone).
* Completed tasks without ``added_at`` are skipped unless the task is already
  in the log — creating a fake creation time would poison the 48h label.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import db

API_BASE = "https://api.todoist.com/api/v1"
DEFAULT_SINCE_DAYS = 90


class TodoistError(RuntimeError):
    pass


def _request(path: str, params: dict[str, str], token: str, timeout: float) -> dict:
    url = f"{API_BASE}{path}"
    if params:
        url += "?" + urlencode(params)
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "procrastination-oracle/0.1",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:  # pragma: no cover - error body best effort
            pass
        if exc.code == 401:
            raise TodoistError(
                "Todoist rejected the token (401). Check TODOIST_API_TOKEN."
            ) from exc
        raise TodoistError(f"Todoist API {exc.code}: {detail or exc.reason}") from exc
    except URLError as exc:
        raise TodoistError(f"could not reach Todoist: {exc.reason}") from exc


def _paginate(
    path: str, params: dict[str, str], token: str, timeout: float
) -> Iterator[dict]:
    cursor: str | None = None
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        payload = _request(path, page_params, token, timeout)
        items = payload.get("results") or payload.get("items") or []
        for item in items:
            if isinstance(item, dict):
                yield item
        cursor = payload.get("next_cursor") or payload.get("nextCursor")
        if not cursor:
            break


def _parse_due(task: dict) -> datetime | None:
    due = task.get("due") or {}
    if not isinstance(due, dict):
        return None
    if due.get("datetime"):
        return db.parse_dt(due["datetime"])
    if due.get("date"):
        # All-day deadline; 17:00 UTC is an explicit approximation, not data.
        return db.parse_dt(f"{due['date']}T17:00:00+00:00")
    return None


def _project_names(token: str, timeout: float) -> dict[str, str]:
    projects = list(_paginate("/projects", {}, token, timeout))
    return {str(p.get("id")): str(p.get("name") or "inbox") for p in projects}


def sync(
    conn: sqlite3.Connection,
    *,
    token: str | None = None,
    since_days: int = DEFAULT_SINCE_DAYS,
    timeout: float = 30.0,
) -> dict[str, int]:
    """Pull active + recently completed tasks into ``task_events``."""
    token = token or os.environ.get("TODOIST_API_TOKEN", "")
    if not token:
        raise TodoistError(
            "TODOIST_API_TOKEN is not set. Copy .env.example to .env and add it."
        )

    now = db.utcnow()
    since = now - timedelta(days=since_days)
    projects = _project_names(token, timeout)
    existing = {event.task_id: event for event in db.load_events(conn)}

    imported: dict[str, db.TaskEvent] = {}
    active_count = 0
    for task in _paginate("/tasks", {"limit": "200"}, token, timeout):
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            continue
        created_at = db.parse_dt(task.get("added_at")) or now
        project = projects.get(str(task.get("project_id")), "inbox")
        previous = existing.get(task_id)
        imported[task_id] = db.TaskEvent(
            task_id=task_id,
            title=str(task.get("content") or "Untitled task").strip(),
            category=project,
            created_at=created_at,
            due_at=_parse_due(task),
            deferred_count=previous.deferred_count if previous else 0,
            completed_at=None,
            deleted_at=None,
            source="todoist",
        )
        active_count += 1

    completed_count = 0
    skipped_completed = 0
    completed_params = {
        "since": since.isoformat(),
        "until": now.isoformat(),
        "limit": "200",
    }
    try:
        completed_items = list(
            _paginate("/tasks/completed/by_completion_date", completed_params, token, timeout)
        )
    except TodoistError as exc:
        # Completed endpoints are the most likely to change; keep the active import.
        if "404" not in str(exc):
            raise
        completed_items = []
        skipped_completed = -1  # signals unsupported endpoint in the summary

    for item in completed_items:
        completed_at = db.parse_dt(item.get("completed_at"))
        if completed_at is None or completed_at < since:
            continue
        task_id = str(item.get("task_id") or item.get("id") or "").strip()
        if not task_id:
            continue
        if task_id in imported:
            imported[task_id] = replace(imported[task_id], completed_at=completed_at)
            completed_count += 1
            continue
        previous = existing.get(task_id)
        added_at = db.parse_dt(item.get("added_at"))
        if previous is not None:
            imported[task_id] = replace(
                previous,
                completed_at=completed_at,
                source="todoist",
            )
            completed_count += 1
        elif added_at is not None:
            imported[task_id] = db.TaskEvent(
                task_id=task_id,
                title=str(item.get("content") or "Untitled task").strip(),
                category=projects.get(str(item.get("project_id")), "inbox"),
                created_at=added_at,
                due_at=None,
                deferred_count=0,
                completed_at=completed_at,
                source="todoist",
            )
            completed_count += 1
        else:
            skipped_completed += 1

    written = db.upsert_events(conn, imported.values(), source="todoist")
    return {
        "active": active_count,
        "completed": completed_count,
        "skipped_completed": skipped_completed,
        "written": written,
        "since": since.date().isoformat(),
    }
