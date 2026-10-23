"""Event log -> feature matrix.

Design rules:

* No scaling, one-hot encoding, imputation, or text vectorisation. TabPFN-3.5
  consumes the raw strings/numbers directly, and doing the transformation
  here would silently reproduce the preprocessing TabPFN is meant to replace.
* One row per task, scored at a single ``as_of`` snapshot. Features only use
  fields that are observable at scoring time: ``completed_at`` and
  ``deleted_at`` never enter the feature frame.
* The label is causal: 1 only when the task was actually completed within 48h
  of ``created_at``. Tasks whose 48h window has not elapsed and which are
  still open are excluded — their label is not knowable yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import db

LABEL_COLUMN = "completed_within_48h"
LABEL_WINDOW = timedelta(hours=48)

FEATURE_COLUMNS: list[str] = [
    "title",
    "category",
    "days_until_due",
    "has_deadline",
    "day_of_week",
    "task_age_days",
    "deferred_count",
    "title_len",
]

META_COLUMNS: list[str] = ["task_id", "created_at", "source"]


def label_for(task: db.TaskEvent, as_of: datetime) -> int | None:
    """1 = completed within 48h, 0 = not, None = window still open."""
    if task.completed_at is not None and task.completed_at <= as_of:
        return int(task.completed_at - task.created_at <= LABEL_WINDOW)
    if task.deleted_at is not None and task.deleted_at <= as_of:
        return 0
    if task.created_at + LABEL_WINDOW <= as_of:
        return 0
    return None


def _row(task: db.TaskEvent, as_of: datetime, label: float) -> dict[str, object]:
    days_until_due = (
        (task.due_at - as_of).total_seconds() / 86400
        if task.due_at is not None
        else float("nan")
    )
    return {
        "task_id": task.task_id,
        "title": task.title,
        "category": task.category or "uncategorized",
        "days_until_due": round(days_until_due, 3),
        "has_deadline": task.due_at is not None,
        "day_of_week": task.created_at.strftime("%A"),
        "task_age_days": round((as_of - task.created_at).total_seconds() / 86400, 3),
        "deferred_count": int(task.deferred_count),
        "title_len": len(task.title),
        "created_at": task.created_at,
        "source": task.source,
        LABEL_COLUMN: label,
    }


def build_features(
    events: Iterable[db.TaskEvent],
    as_of: datetime | None = None,
    *,
    include_unresolved: bool = False,
) -> pd.DataFrame:
    """Build the training/inference frame, newest task last.

    ``as_of`` defaults to ``db.utcnow()``. All rows share it so the model sees
    the same notion of "now" that it will see at inference time.

    Unresolved tasks (open, less than 48h old) are dropped by default because
    their label is unknowable. Live inference passes ``include_unresolved=True``
    to score them; their label is then NaN and must be ignored.
    """
    as_of = (as_of or db.utcnow()).astimezone(timezone.utc)
    rows: list[dict[str, object]] = []
    for task in events:
        label = label_for(task, as_of)
        if label is None:
            if not include_unresolved:
                continue
            label = float("nan")  # type: ignore[assignment]
        rows.append(_row(task, as_of, label))

    frame = pd.DataFrame(rows, columns=META_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN])
    if not frame.empty:
        frame = frame.sort_values(["created_at", "task_id"], kind="mergesort")
        frame = frame.reset_index(drop=True)
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
        frame["has_deadline"] = frame["has_deadline"].astype(bool)
        frame["deferred_count"] = frame["deferred_count"].astype("int64")
        frame["title_len"] = frame["title_len"].astype("int64")
        frame["days_until_due"] = frame["days_until_due"].astype("float64")
        frame["task_age_days"] = frame["task_age_days"].astype("float64")
        if include_unresolved:
            frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype("float64")
        else:
            frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype("int64")
    return frame


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The exact columns handed to every model, in a stable order."""
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return frame[FEATURE_COLUMNS].copy()


def time_split(
    frame: pd.DataFrame, test_fraction: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — the last ``test_fraction`` of tasks are the test set.

    Never shuffles: the test set must be strictly newer than the training set,
    otherwise the evaluation leaks the future into the model.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("need at least 2 rows to split")
    ordered = frame.sort_values(["created_at", "task_id"], kind="mergesort")
    n_test = max(1, int(round(len(ordered) * test_fraction)))
    n_test = min(n_test, len(ordered) - 1)
    train = ordered.iloc[:-n_test].reset_index(drop=True)
    test = ordered.iloc[-n_test:].reset_index(drop=True)
    return train, test


def completion_slot(events: Sequence[db.TaskEvent]) -> dict[str, object]:
    """The weekday/hour when this person actually finishes things."""
    completed = [e for e in events if e.completed_at is not None]
    if not completed:
        return {
            "weekday": "Saturday",
            "hour": 10,
            "label": "Saturday 10:00",
            "n_completions": 0,
        }
    buckets: dict[tuple[int, int], int] = {}
    for event in completed:
        key = (event.completed_at.weekday(), event.completed_at.hour)  # type: ignore[union-attr]
        buckets[key] = buckets.get(key, 0) + 1
    (weekday, hour), _ = max(buckets.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
    label = f"{_weekday_name(weekday)} {hour:02d}:00"
    return {
        "weekday": _weekday_name(weekday),
        "hour": hour,
        "label": label,
        "n_completions": len(completed),
    }


def _weekday_name(index: int) -> str:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][
        index % 7
    ]


def summarize_patterns(
    events: Sequence[db.TaskEvent], as_of: datetime | None = None
) -> dict[str, object]:
    """Human-readable behavior patterns, shared by the briefing and MCP."""
    as_of = (as_of or db.utcnow()).astimezone(timezone.utc)
    frame = build_features(events, as_of=as_of)
    rows = list(events)
    if frame.empty:
        return {
            "n_tasks": len(rows),
            "n_labeled": 0,
            "completion_rate_48h": 0.0,
            "avg_deferred": 0.0,
            "by_category": {},
            "by_weekday": {},
            "worst_category": None,
            "most_deferred": [],
            "best_completion_slot": completion_slot(rows),
            "as_of": as_of.isoformat(),
        }

    by_category: dict[str, dict[str, float]] = {}
    for category, group in frame.groupby("category", dropna=False):
        by_category[str(category)] = {
            "n": int(len(group)),
            "completion_rate_48h": round(float(group[LABEL_COLUMN].mean()), 3),
            "avg_deferred": round(float(group["deferred_count"].mean()), 2),
        }

    by_weekday: dict[str, dict[str, float]] = {}
    for weekday, group in frame.groupby("day_of_week", dropna=False):
        by_weekday[str(weekday)] = {
            "n": int(len(group)),
            "completion_rate_48h": round(float(group[LABEL_COLUMN].mean()), 3),
        }

    avoidant = frame.groupby("title", dropna=False)["deferred_count"].agg(["mean", "count"])
    avoidant = avoidant[avoidant["count"] >= 2].sort_values("mean", ascending=False)
    most_deferred = [
        {
            "title": str(title),
            "avg_deferred": round(float(row["mean"]), 2),
            "n": int(row["count"]),
        }
        for title, row in avoidant.head(5).iterrows()
    ]

    worst = min(
        (c for c in by_category.items() if c[1]["n"] >= 3),
        key=lambda item: item[1]["completion_rate_48h"],
        default=(None, {}),
    )

    return {
        "n_tasks": len(rows),
        "n_labeled": int(len(frame)),
        "completion_rate_48h": round(float(frame[LABEL_COLUMN].mean()), 3),
        "avg_deferred": round(float(frame["deferred_count"].mean()), 2),
        "by_category": by_category,
        "by_weekday": by_weekday,
        "worst_category": (
            {
                "category": worst[0],
                **worst[1],
            }
            if worst[0]
            else None
        ),
        "most_deferred": most_deferred,
        "best_completion_slot": completion_slot(rows),
        "as_of": as_of.isoformat(),
    }


def risk_tier(probability: float) -> str:
    """Single source of truth for HIGH / MED / LOW thresholds."""
    if probability >= 0.60:
        return "HIGH"
    if probability >= 0.30:
        return "MED"
    return "LOW"


def is_synthetic(frame: pd.DataFrame) -> bool:
    return bool((frame.get("source", pd.Series(dtype=str)) == "synthetic").any())


def empty_feature_frame() -> pd.DataFrame:
    """Used by callers that need a typed, empty frame (e.g. no open tasks)."""
    frame = pd.DataFrame(columns=META_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN])
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(np.int64)
    return frame
