"""Risk-tiered interventions.

    HIGH → break the task into three tiny subtasks
    MED  → reschedule it to the user's best completion slot
    LOW  → draft the first two-minute step

The Ollama model writes the specifics. If it is not running (or returns
garbage), a deterministic fallback with the same shape is used and tagged as
such — the briefing never silently pretends an LLM wrote it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import db
import features
from oracle import ScoredTask

from .llm import LLMUnavailable, OllamaLLM

logger = logging.getLogger(__name__)

ACTIONS = {
    "HIGH": "break_into_subtasks",
    "MED": "reschedule",
    "LOW": "draft_first_step",
}
ACTION_LABELS = {
    "HIGH": "break into 3 tiny subtasks",
    "MED": "reschedule to your best slot",
    "LOW": "draft the first 2-minute step",
}
SYSTEM_PROMPT = (
    "You are the intervention engine of a personal productivity app. "
    "You are direct, warm, and concrete. Reply with a single JSON object and "
    "nothing else."
)


@dataclass(frozen=True)
class Intervention:
    task_id: str
    title: str
    risk: float
    tier: str
    action: str
    headline: str
    details: tuple[str, ...]
    reason: str
    source: str  # "llm" or "fallback"
    created_at: datetime

    def to_markdown(self) -> str:
        steps = "\n".join(f"{i}. {step}" for i, step in enumerate(self.details, 1))
        attribution = "Ollama" if self.source == "llm" else "heuristic fallback"
        return (
            f"**{self.headline}**  \n"
            f"{steps}  \n"
            f"_Why: {self.reason}_ · `{attribution}`"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "risk": round(self.risk, 4),
            "tier": self.tier,
            "action": self.action,
            "headline": self.headline,
            "details": list(self.details),
            "reason": self.reason,
            "source": self.source,
            "created_at": db.to_iso(self.created_at),
        }


def _due_label(task: ScoredTask) -> str:
    if task.due_at is None:
        return "no deadline"
    if task.days_until_due is None:
        return task.due_at.strftime("%b %d")
    if task.days_until_due < 0:
        return f"overdue by {abs(task.days_until_due):.0f}d"
    if task.days_until_due < 1:
        return f"due today ({task.due_at:%H:%M})"
    return f"due {task.due_at:%b %d} ({task.days_until_due:.0f}d)"


def _prompt(
    task: ScoredTask,
    *,
    slot_label: str,
    patterns: dict[str, object] | None,
) -> str:
    tier = task.tier
    lines = [
        "A TabPFN model predicts this task may be skipped.",
        "",
        "TASK",
        f"- title: {task.title}",
        f"- category: {task.category or 'uncategorized'}",
        f"- times deferred: {task.deferred_count}",
        f"- deadline: {_due_label(task)}",
        f"- risk of NOT completing within 48h: {task.risk:.0%} ({tier})",
        f"- best completion slot: {slot_label}",
    ]
    if patterns:
        worst = patterns.get("worst_category")
        if isinstance(worst, dict) and worst.get("category"):
            lines.append(
                f"- user's worst category: {worst['category']} "
                f"({worst.get('completion_rate_48h', 0):.0%} completion)"
            )
        lines.append(
            f"- overall 48h completion rate: {patterns.get('completion_rate_48h', 0):.0%}"
        )

    lines += ["", "INTERVENTION"]
    if tier == "HIGH":
        lines += [
            "Break the task into exactly THREE subtasks, each doable in 15 minutes",
            "or less. Make the first one almost embarrassingly easy. Use concrete",
            "verbs, not categories.",
        ]
    elif tier == "MED":
        lines += [
            f"Reschedule the task to the user's best completion slot ({slot_label}).",
            "Explain the time block concretely and add one line of preparation the",
            "night before. Do not shame the user.",
        ]
    else:
        lines += [
            "Draft the FIRST two-minute step. It must be a physical action the user",
            "can start immediately (open X, write one sentence, send one line). Add",
            "permission to stop after two minutes.",
        ]

    lines += [
        "",
        "Return JSON with exactly these keys:",
        '  "headline": one short imperative sentence (max 90 chars)',
        '  "details": array of 1-3 strings, each one concrete action',
        '  "reason": one sentence tying the action to the risk, with no numbers',
    ]
    return "\n".join(lines)


def _clean_text(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return text[:limit].strip()


def _validate(payload: dict[str, object], tier: str) -> tuple[str, tuple[str, ...], str]:
    headline = _clean_text(payload.get("headline", ""), 140)
    reason = _clean_text(payload.get("reason", ""), 240)
    raw_details = payload.get("details", [])
    if isinstance(raw_details, str):
        raw_details = [part for part in raw_details.splitlines() if part.strip()]
    if not isinstance(raw_details, list):
        raise LLMUnavailable("intervention details must be a list")

    details = tuple(
        _clean_text(item, 240) for item in raw_details if _clean_text(item, 240)
    )
    if not headline or not reason or not details:
        raise LLMUnavailable("intervention is missing headline, details, or reason")
    if tier == "HIGH" and len(details) < 3:
        raise LLMUnavailable("HIGH intervention needs three subtasks")
    return headline, details[:3], reason


def _fallback(task: ScoredTask, *, slot_label: str) -> Intervention:
    risk_note = f"{task.risk:.0%} probability of slipping past 48 hours."
    if task.tier == "HIGH":
        headline = f"Break '{task.title}' into three tiny steps"
        details = (
            "Open everything you need and look at it for two minutes.",
            "Do the smallest visible piece for ten minutes — finishing is not allowed.",
            "Write the next action on a sticky note, then stop.",
        )
        reason = (
            "The model puts the odds of slipping past 48 hours at "
            f"{task.risk:.0%}."
        )
    elif task.tier == "MED":
        headline = f"Move '{task.title}' to {slot_label}"
        details = (
            f"Block 25 minutes at {slot_label}; that is when you actually finish things.",
            "Put the task where you will see it the night before.",
            "Show up even if you only do the first five minutes.",
        )
        reason = f"Medium risk ({risk_note}) — a better slot is cheaper than more willpower."
    else:
        headline = f"Start '{task.title}' for two minutes"
        details = (
            "Do only the first physical action: open the email, doc, or app.",
            "Set a two-minute timer; stopping when it rings is allowed.",
        )
        reason = f"Low risk ({risk_note}) — this is about momentum, not effort."

    return Intervention(
        task_id=task.task_id,
        title=task.title,
        risk=task.risk,
        tier=task.tier,
        action=ACTIONS[task.tier],
        headline=headline,
        details=details,
        reason=reason,
        source="fallback",
        created_at=db.utcnow(),
    )


def intervene(
    task: ScoredTask,
    *,
    llm: OllamaLLM | None = None,
    slot_label: str = "your best slot",
    patterns: dict[str, object] | None = None,
    now: datetime | None = None,
) -> Intervention:
    """Take one intervention for one task, falling back if the LLM fails."""
    tier = task.tier if task.tier in ACTIONS else features.risk_tier(task.risk)
    if llm is not None:
        try:
            payload = llm.generate_json(
                _prompt(task, slot_label=slot_label, patterns=patterns),
                system=SYSTEM_PROMPT,
            )
            headline, details, reason = _validate(payload, tier)
            return Intervention(
                task_id=task.task_id,
                title=task.title,
                risk=task.risk,
                tier=tier,
                action=ACTIONS[tier],
                headline=headline,
                details=details,
                reason=reason,
                source="llm",
                created_at=now or db.utcnow(),
            )
        except LLMUnavailable as exc:
            logger.warning("LLM intervention failed for %r: %s", task.task_id, exc)

    return _fallback(task, slot_label=slot_label)


def run_interventions(
    tasks: Sequence[ScoredTask],
    *,
    llm: OllamaLLM | None = None,
    slot_label: str = "your best slot",
    patterns: dict[str, object] | None = None,
    max_tasks: int = 6,
) -> list[Intervention]:
    """Intervene on the highest-risk tasks, at most ``max_tasks``."""
    chosen = sorted(tasks, key=lambda t: -t.risk)[: max(0, max_tasks)]
    return [
        intervene(
            task,
            llm=llm,
            slot_label=slot_label,
            patterns=patterns,
            now=db.utcnow(),
        )
        for task in chosen
    ]


def interventions_from_rows(rows: Sequence[dict[str, object]]) -> list[Intervention]:
    """Rebuild interventions from Temporal-serialised dicts (for tests/tools)."""
    rebuilt: list[Intervention] = []
    for row in rows:
        created = db.parse_dt(str(row.get("created_at"))) or db.utcnow()
        rebuilt.append(
            Intervention(
                task_id=str(row["task_id"]),
                title=str(row["title"]),
                risk=float(row["risk"]),
                tier=str(row["tier"]),
                action=str(row["action"]),
                headline=str(row["headline"]),
                details=tuple(str(d) for d in row.get("details", [])),
                reason=str(row["reason"]),
                source=str(row.get("source", "fallback")),
                created_at=created,
            )
        )
    return rebuilt
