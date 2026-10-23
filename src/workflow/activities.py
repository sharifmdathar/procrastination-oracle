"""Temporal activities: sync → predict → intervene → brief.

Each activity is a plain function; Temporal retries them according to the
workflow's policy. The standalone CLI (``scripts/run_nightly.py``) calls the
same libraries directly, which is why none of the business logic lives here.
"""

from __future__ import annotations

from typing import Any

from temporalio import activity

import briefing
import db
import features
import oracle
from agent.interventions import Intervention, interventions_from_rows, run_interventions
from agent.llm import OllamaLLM
from collectors import synthetic, todoist

TASK_QUEUE = "procrastination-oracle"


def _request_path(request: dict[str, Any], key: str) -> str | None:
    value = request.get(key)
    return str(value) if value else None


def _scored_from_dict(row: dict[str, Any]) -> oracle.ScoredTask:
    return oracle.ScoredTask(
        task_id=str(row["task_id"]),
        title=str(row["title"]),
        category=(str(row["category"]) if row.get("category") else None),
        created_at=db.parse_dt(row["created_at"]) or db.utcnow(),
        due_at=db.parse_dt(row.get("due_at")),
        risk=float(row["risk"]),
        tier=str(row["tier"]),
        days_until_due=(
            float(row["days_until_due"]) if row.get("days_until_due") is not None else None
        ),
        deferred_count=int(row.get("deferred_count", 0)),
        source=str(row.get("source", "unknown")),
    )


@activity.defn
def sync_tasks(request: dict[str, Any]) -> dict[str, Any]:
    """Pull fresh Todoist tasks when a token is configured; otherwise no-op."""
    conn = db.connect(_request_path(request, "db"))
    try:
        summary = todoist.sync(conn, since_days=int(request.get("since_days", 90)))
        return {"synced": True, **summary}
    except todoist.TodoistError as exc:
        activity.logger.warning("Todoist sync skipped: %s", exc)
        return {"synced": False, "detail": str(exc)}
    finally:
        conn.close()


@activity.defn
def predict_risks(request: dict[str, Any]) -> dict[str, Any]:
    """Score every open task with the saved TabPFN-3.5 model."""
    conn = db.connect(_request_path(request, "db"))
    try:
        events = db.load_events(conn)
    finally:
        conn.close()
    if not events:
        raise ValueError("event log is empty; run the synthetic generator or sync Todoist")

    model_path = _request_path(request, "model") or str(oracle.DEFAULT_MODEL_PATH)
    model = oracle.load_oracle(model_path)
    as_of = db.utcnow()
    scored = oracle.score_tasks(model, events, as_of=as_of)
    patterns = features.summarize_patterns(events, as_of=as_of)
    return {
        "as_of": db.to_iso(as_of),
        "model_version": model.model_version,
        "n_tasks": len(events),
        "scored": [item.as_dict() for item in scored],
        "patterns": patterns,
        "synthetic": any(e.source == synthetic.SYNTHETIC_SOURCE for e in events),
    }


@activity.defn
def take_interventions(
    request: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, Any]:
    """Let the agent act on the highest-risk tasks and persist what it did."""
    scored = [_scored_from_dict(row) for row in prediction["scored"]]
    patterns = prediction.get("patterns") or {}
    slot = patterns.get("best_completion_slot") or {}
    slot_label = str(slot.get("label", "your best slot"))

    llm = None
    if not request.get("no_llm"):
        candidate = OllamaLLM()
        if candidate.status().available:
            llm = candidate

    interventions = run_interventions(
        scored,
        llm=llm,
        slot_label=slot_label,
        patterns=patterns,
        max_tasks=int(request.get("max_interventions", 6)),
    )
    run_at = db.parse_dt(prediction["as_of"]) or db.utcnow()

    conn = db.connect(_request_path(request, "db"))
    try:
        for item in interventions:
            db.record_intervention(
                conn,
                task_id=item.task_id,
                risk=item.risk,
                tier=item.tier,
                action=item.action,
                body=item.to_markdown(),
                run_at=run_at,
                source=item.source,
            )
        db.record_run(
            conn,
            run_at=run_at,
            n_tasks=int(prediction.get("n_tasks", 0)),
            n_at_risk=sum(1 for t in scored if t.tier in {"HIGH", "MED"}),
            label="temporal-nightly",
            synthetic=bool(prediction.get("synthetic")),
        )
    finally:
        conn.close()

    return {
        "run_at": db.to_iso(run_at),
        "interventions": [item.as_dict() for item in interventions],
        "llm_used": sum(1 for item in interventions if item.source == "llm"),
    }


@activity.defn
def write_briefing(
    request: dict[str, Any],
    prediction: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Render morning_briefing.md from predictions and interventions."""
    scored = [_scored_from_dict(row) for row in prediction["scored"]]
    interventions: list[Intervention] = interventions_from_rows(result["interventions"])
    path = briefing.build_and_write(
        scored=scored,
        interventions=interventions,
        patterns=prediction.get("patterns"),
        path=_request_path(request, "out") or str(briefing.DEFAULT_BRIEFING_PATH),
        as_of=db.parse_dt(prediction["as_of"]),
        synthetic=bool(prediction.get("synthetic")),
    )
    return {"path": str(path), "n_interventions": len(interventions)}

