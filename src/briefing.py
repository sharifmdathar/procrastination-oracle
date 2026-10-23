"""Morning briefing rendering.

Two lines of truth, a risk table, and the interventions the agent already
took. Written to ``morning_briefing.md`` so it can be read before opening
Todoist (or any other app).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import db
import features
from agent.interventions import ACTION_LABELS, Intervention
from oracle import ScoredTask

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIEFING_PATH = ROOT / "morning_briefing.md"
MAX_TABLE_ROWS = 12


def _due(task: ScoredTask) -> str:
    if task.due_at is None:
        return "—"
    days = task.days_until_due
    if days is not None and days < 0:
        return f"overdue {abs(days):.0f}d"
    if days is not None and days < 1:
        return "today"
    return task.due_at.strftime("%b %d")


def _cell(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _summary(
    scored: Sequence[ScoredTask],
    interventions: Sequence[Intervention],
    patterns: dict[str, object] | None,
) -> tuple[str, str]:
    total = len(scored)
    high = sum(1 for t in scored if t.tier == "HIGH")
    med = sum(1 for t in scored if t.tier == "MED")
    low = sum(1 for t in scored if t.tier == "LOW")
    acted = len(interventions)

    if total == 0:
        first = "**Nothing is at risk today.** The Oracle found no open tasks to worry about."
    elif high == 0 and med == 0:
        first = (
            f"**{total} open task{'s' if total != 1 else ''}, none in the danger zone.** "
            f"{acted} gentle nudge{'s' if acted != 1 else ''} prepared."
        )
    else:
        first = (
            f"**{high} task{'s' if high != 1 else ''} at high risk today, "
            f"{med} on the edge.** The agent already took {acted} "
            f"intervention{'s' if acted != 1 else ''} so you don't have to decide."
        )

    rate = float(patterns.get("completion_rate_48h", 0.0)) if patterns else 0.0
    slot = patterns.get("best_completion_slot") if patterns else None
    slot_label = slot.get("label", "unknown") if isinstance(slot, dict) else "unknown"
    worst = patterns.get("worst_category") if patterns else None
    second = (
        f"You finish {rate:.0%} of tasks within 48h; your best window is "
        f"{slot_label}"
    )
    if isinstance(worst, dict) and worst.get("category"):
        second += (
            f", and **{worst['category']}** is where things stall "
            f"({float(worst.get('completion_rate_48h', 0)):.0%} done)."
        )
    else:
        second += "."
    return first, second


def render_briefing(
    *,
    scored: Sequence[ScoredTask],
    interventions: Sequence[Intervention],
    patterns: dict[str, object] | None = None,
    as_of: datetime | None = None,
    synthetic: bool = False,
    title: str = "Morning Briefing",
) -> str:
    as_of = (as_of or db.utcnow()).astimezone(timezone.utc)
    first, second = _summary(scored, interventions, patterns)

    lines = [f"# {title} — {as_of:%A, %B %d, %Y}", ""]
    if synthetic:
        lines += [
            "> **SYNTHETIC DATA** — this log belongs to a fake procrastinator "
            "generated for the demo, not to you.",
            "",
        ]
    lines += [first, "", second, ""]

    lines += ["## Risk table", ""]
    if not scored:
        lines += ["_No open tasks._", ""]
    else:
        lines += [
            "| Task | Due | Risk | Tier | Deferred | Agent |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        by_task = {item.task_id: item for item in interventions}
        for task in scored[:MAX_TABLE_ROWS]:
            taken = "✓" if task.task_id in by_task else "—"
            lines.append(
                f"| {_cell(task.title)} | {_due(task)} | {task.risk:.0%} | "
                f"{task.tier} | {task.deferred_count}× | {taken} |"
            )
        if len(scored) > MAX_TABLE_ROWS:
            lines.append(
                f"| _…and {len(scored) - MAX_TABLE_ROWS} more open tasks_ | | | | | |"
            )
        lines.append("")

    lines += ["## Interventions already taken", ""]
    if not interventions:
        lines += ["_None needed tonight._", ""]
    else:
        for item in sorted(interventions, key=lambda i: -i.risk):
            lines += [
                f"### {_cell(item.title)}",
                "",
                f"`{item.risk:.0%} risk` · **{item.tier}** · "
                f"{ACTION_LABELS.get(item.tier, item.action)}",
                "",
                item.to_markdown(),
                "",
            ]

    lines += [
        "---",
        "",
        f"_Generated locally at {as_of:%Y-%m-%d %H:%M UTC} by Procrastination Oracle. "
        "Probabilities, not destiny._",
        "",
    ]
    return "\n".join(lines)


def write_briefing(
    text: str,
    path: str | Path = DEFAULT_BRIEFING_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def build_and_write(
    *,
    scored: Sequence[ScoredTask],
    interventions: Sequence[Intervention],
    patterns: dict[str, object] | None = None,
    path: str | Path = DEFAULT_BRIEFING_PATH,
    as_of: datetime | None = None,
    synthetic: bool = False,
) -> Path:
    text = render_briefing(
        scored=scored,
        interventions=interventions,
        patterns=patterns,
        as_of=as_of,
        synthetic=synthetic,
    )
    return write_briefing(text, path)


def patterns_from_events(events: Sequence[db.TaskEvent]) -> dict[str, object]:
    return features.summarize_patterns(events)
