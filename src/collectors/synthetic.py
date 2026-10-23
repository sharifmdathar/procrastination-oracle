"""SYNTHETIC DATA — every row produced here is fake.

This module exists so the app can be demoed without connecting a real Todoist
account. Generated tasks are tagged ``source='synthetic'`` in SQLite, the CLI
prints a warning, and the briefing renders a banner. Do not mix these rows
with a real event log and mistake the output for a prediction about you.

The generator is deterministic for a given seed and models a plausible
procrastinator: social plans get done, ``Call the dentist`` gets deferred nine
times and never happens, evening admin tasks rot, and deadlines create a small
urgency bump. Labels are causal — a task is positive only if it was actually
completed within 48 hours of entering the list.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import db

SYNTHETIC_SOURCE = "synthetic"
SYNTHETIC_BANNER = (
    "SYNTHETIC DATA: these tasks were generated to demo the Oracle. "
    "They are not a record of your behavior."
)

HORIZON_DAYS = 120
LABEL_WINDOW_HOURS = 48

BASE_COMPLETION = {
    "social": 0.84,
    "errands": 0.62,
    "work": 0.60,
    "home": 0.50,
    "health": 0.46,
    "admin": 0.42,
    "creative": 0.38,
}

QUICK_SHARE = {
    "social": 0.90,
    "errands": 0.78,
    "work": 0.62,
    "home": 0.64,
    "health": 0.58,
    "admin": 0.50,
    "creative": 0.56,
}

# More avoidant tasks start with more deferrals and are less likely to be done.
DEFERRAL_WEIGHTS = [0, 1, 2, 3, 4, 5, 6, 8]
DEFERRAL_PROBS = [0.30, 0.22, 0.16, 0.12, 0.08, 0.055, 0.035, 0.02]


@dataclass(frozen=True)
class TaskTemplate:
    title: str
    category: str
    bias: float = 0.0
    avoidant: float = 0.0
    weight: float = 1.0


TEMPLATES: list[TaskTemplate] = [
    # The dentist task is pinned separately in _signature_tasks so its nine
    # deferrals stay visible in every generated dataset.
    TaskTemplate("Book a physio appointment", "health", -0.10, 0.4),
    TaskTemplate("Schedule annual checkup", "health", -0.15, 0.5),
    TaskTemplate("Refill prescription", "health", 0.05, 0.1),
    TaskTemplate("File quarterly taxes", "admin", -0.25, 0.8),
    TaskTemplate("Reply to landlord email", "admin", 0.0, 0.2),
    TaskTemplate("Renew passport", "admin", -0.20, 0.7),
    TaskTemplate("Update mailing address", "admin", -0.10, 0.4),
    TaskTemplate("Fix the leaking faucet", "home", -0.12, 0.5),
    TaskTemplate("Assemble the bookshelf", "home", -0.05, 0.3),
    TaskTemplate("Deep clean the kitchen", "home", -0.15, 0.6),
    TaskTemplate("Do the laundry", "home", 0.10, 0.0),
    TaskTemplate("Draft the project proposal", "work", -0.10, 0.3),
    TaskTemplate("Review the open pull request", "work", 0.02, 0.1),
    TaskTemplate("Prepare slides for standup", "work", 0.0, 0.1),
    TaskTemplate("Write the weekly report", "work", -0.08, 0.3),
    TaskTemplate("Write a blog post", "creative", -0.10, 0.4),
    TaskTemplate("Edit the portfolio", "creative", -0.12, 0.5),
    TaskTemplate("Practice guitar for 20 minutes", "creative", 0.05, 0.0),
    TaskTemplate("Sketch logo concepts", "creative", -0.05, 0.2),
    TaskTemplate("Return the Amazon package", "errands", -0.18, 0.7),
    TaskTemplate("Buy groceries", "errands", 0.15, 0.0),
    TaskTemplate("Pick up dry cleaning", "errands", -0.05, 0.3),
    TaskTemplate("Get the bike tire fixed", "errands", -0.10, 0.4),
    TaskTemplate("Call Mom", "social", 0.05, 0.0),
    TaskTemplate("Plan the birthday dinner", "social", 0.0, 0.1),
    TaskTemplate("RSVP to the wedding", "social", -0.05, 0.2),
    # "Coffee with Sam" is pinned weekly in _signature_tasks.
    TaskTemplate("Grab lunch with Priya", "social", 0.12, 0.0, 0.6),
]

TEMPLATE_WEIGHTS = [t.weight for t in TEMPLATES]


def _clamp(value: float, low: float = 0.02, high: float = 0.97) -> float:
    return max(low, min(high, value))


def _sample_deferrals(rng: random.Random, avoidant: float) -> int:
    base = rng.choices(DEFERRAL_WEIGHTS, weights=DEFERRAL_PROBS, k=1)[0]
    if avoidant <= 0:
        return base
    extra = rng.random() < avoidant
    return base + (rng.randint(1, 5) if extra else 0)


def _completion_probability(
    template: TaskTemplate, created_at: datetime, due_at: datetime | None, deferred: int
) -> float:
    p = BASE_COMPLETION[template.category] + template.bias
    if due_at is not None:
        lead = (due_at - created_at).total_seconds() / 3600
        if lead <= 48:
            p += 0.15  # deadline pressure
        elif lead > 10 * 24:
            p -= 0.05  # "later" is a dangerous place
        if lead < 0:
            p -= 0.20  # already overdue, avoidance wins
    else:
        p -= 0.04
    p -= 0.035 * deferred
    return _clamp(p)


def _with_time(base: datetime, hour: int, minute: int = 0) -> datetime:
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _make_task(
    rng: random.Random,
    task_id: str,
    template: TaskTemplate,
    created_at: datetime,
    end_date: datetime,
    *,
    force_deferred: int | None = None,
) -> db.TaskEvent:
    """Build one task with an outcome that is only knowable after it happens."""
    created_at = created_at.astimezone(timezone.utc).replace(microsecond=0)

    due_at: datetime | None = None
    if rng.random() < 0.62:
        offset_days = rng.choices(
            [-3, -1, 0, 1, 2, 3, 5, 7, 10, 14],
            weights=[2, 3, 4, 8, 9, 8, 6, 4, 3, 2],
            k=1,
        )[0]
        due_at = _with_time(
            created_at + timedelta(days=offset_days),
            rng.choice([9, 12, 17, 18]),
        )

    deferred = (
        force_deferred
        if force_deferred is not None
        else _sample_deferrals(rng, template.avoidant)
    )

    completed_at: datetime | None = None
    deleted_at: datetime | None = None

    if rng.random() < _completion_probability(template, created_at, due_at, deferred):
        quick = rng.random() < QUICK_SHARE[template.category]
        delay_hours = (
            rng.uniform(0.5, LABEL_WINDOW_HOURS - 1)
            if quick
            else rng.uniform(LABEL_WINDOW_HOURS + 1, 24 * 9)
        )
        candidate = created_at + timedelta(hours=delay_hours)
        if candidate <= end_date:
            completed_at = candidate
        # else: the outcome has not happened yet as of end_date; task is open.

    if completed_at is None and rng.random() < 0.07:
        candidate = created_at + timedelta(hours=rng.uniform(2, 24 * 6))
        if candidate <= end_date:
            deleted_at = candidate

    return db.TaskEvent(
        task_id=task_id,
        title=template.title,
        category=template.category,
        created_at=created_at,
        due_at=due_at,
        deferred_count=deferred,
        completed_at=completed_at,
        deleted_at=deleted_at,
        source=SYNTHETIC_SOURCE,
    )


def _signature_tasks(
    seed: int, end_date: datetime, rng: random.Random
) -> list[db.TaskEvent]:
    """The behaviours the demo is named after, pinned so they are always visible."""
    tasks: list[db.TaskEvent] = []

    dentist_created = _with_time(end_date - timedelta(days=31), 10, 12)
    tasks.append(
        db.TaskEvent(
            task_id=f"syn-{seed}-dentist",
            title="Call the dentist about the filling",
            category="health",
            created_at=dentist_created,
            due_at=_with_time(dentist_created + timedelta(days=2), 17),
            deferred_count=9,
            completed_at=None,
            deleted_at=None,
            source=SYNTHETIC_SOURCE,
        )
    )

    taxes_created = _with_time(end_date - timedelta(days=68), 20, 41)
    tasks.append(
        db.TaskEvent(
            task_id=f"syn-{seed}-taxes",
            title="File quarterly taxes",
            category="admin",
            created_at=taxes_created,
            due_at=_with_time(taxes_created + timedelta(days=4), 23, 59),
            deferred_count=6,
            completed_at=None,
            deleted_at=None,
            source=SYNTHETIC_SOURCE,
        )
    )

    # Coffee with Sam happens every week, on time, no matter what.
    for i in range(12):
        coffee_day = _with_time(end_date - timedelta(days=7 * i + 3), 8, 30)
        tasks.append(
            db.TaskEvent(
                task_id=f"syn-{seed}-coffee-{i:02d}",
                title="Coffee with Sam",
                category="social",
                created_at=coffee_day,
                due_at=_with_time(coffee_day, 10),
                deferred_count=0,
                completed_at=_with_time(coffee_day, 9, 50),
                deleted_at=None,
                source=SYNTHETIC_SOURCE,
            )
        )

    return tasks


def generate_tasks(
    n: int = 400,
    seed: int = 7,
    end_date: datetime | None = None,
) -> list[db.TaskEvent]:
    """Generate ``n`` synthetic tasks ending at ``end_date`` (default: now)."""
    if n < 20:
        raise ValueError("generate at least 20 tasks so both classes appear")
    rng = random.Random(seed)
    end = (end_date or db.utcnow()).astimezone(timezone.utc).replace(microsecond=0)

    tasks = _signature_tasks(seed, end, rng)
    # Signature rows are part of the requested count, but cap them if n is tiny.
    signature_count = min(len(tasks), max(3, n // 10))
    tasks = tasks[:signature_count]

    index = 0
    while len(tasks) < n:
        template = rng.choices(TEMPLATES, weights=TEMPLATE_WEIGHTS, k=1)[0]
        days_ago = rng.uniform(0, HORIZON_DAYS)
        created = _with_time(
            end - timedelta(days=days_ago),
            rng.choices([7, 8, 9, 12, 13, 17, 20, 21], weights=[2, 5, 6, 4, 4, 5, 3, 2], k=1)[0],
            rng.choice([0, 5, 15, 30, 45]),
        ).replace(microsecond=0)
        if created >= end:
            created = end - timedelta(hours=1)
        tasks.append(
            _make_task(
                rng,
                task_id=f"syn-{seed}-{index:04d}",
                template=template,
                created_at=created,
                end_date=end,
            )
        )
        index += 1

    return sorted(tasks, key=lambda t: (t.created_at, t.task_id))


def generate_and_store(
    db_path: str | Path | None = None,
    n: int = 400,
    seed: int = 7,
    *,
    reset: bool = True,
    end_date: datetime | None = None,
) -> dict[str, object]:
    """Write a fresh synthetic log to SQLite and return summary statistics."""
    tasks = generate_tasks(n=n, seed=seed, end_date=end_date)
    conn = db.connect(db_path)
    try:
        if reset:
            db.delete_synthetic(conn)
        db.upsert_events(conn, tasks, source=SYNTHETIC_SOURCE)
    finally:
        conn.close()
    return summarise(tasks, seed=seed)


def summarise(tasks: Iterable[db.TaskEvent], seed: int = 7) -> dict[str, object]:
    rows = list(tasks)
    completed = [t for t in rows if t.completed_at is not None]
    within_48h = [
        t
        for t in completed
        if t.completed_at is not None
        and t.completed_at - t.created_at <= timedelta(hours=LABEL_WINDOW_HOURS)
    ]
    deferred = [t.deferred_count for t in rows]
    by_category: dict[str, dict[str, float]] = {}
    for task in rows:
        bucket = by_category.setdefault(
            task.category or "uncategorized", {"n": 0, "completed_48h": 0}
        )
        bucket["n"] += 1
        if task in within_48h:
            bucket["completed_48h"] += 1
    for bucket in by_category.values():
        bucket["completion_rate_48h"] = round(
            bucket["completed_48h"] / bucket["n"], 3
        )
    return {
        "total": len(rows),
        "completed": len(completed),
        "completed_within_48h": len(within_48h),
        "completion_rate_48h": round(len(within_48h) / len(rows), 3) if rows else 0.0,
        "open": sum(1 for t in rows if t.is_open),
        "deleted": sum(1 for t in rows if t.deleted_at is not None),
        "avg_deferred": round(statistics.fmean(deferred), 2) if deferred else 0.0,
        "by_category": by_category,
        "seed": seed,
        "source": SYNTHETIC_SOURCE,
    }


def main() -> None:  # pragma: no cover - convenience entry point
    stats = generate_and_store()
    print(SYNTHETIC_BANNER)
    print(stats)
