#!/usr/bin/env python
"""Run the whole nightly pipeline without Temporal or any other infra.

    python scripts/run_nightly.py

Flow: (optional Todoist sync) → score open tasks with the saved TabPFN model →
take tier-specific interventions via Ollama (or deterministic fallbacks) →
write morning_briefing.md. Safe to run from cron.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

import briefing  # noqa: E402
import db  # noqa: E402
import features  # noqa: E402
import oracle  # noqa: E402
from agent.interventions import run_interventions  # noqa: E402
from agent.llm import OllamaLLM  # noqa: E402
from collectors import synthetic, todoist  # noqa: E402

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=oracle.DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=briefing.DEFAULT_BRIEFING_PATH)
    parser.add_argument("--max-interventions", type=int, default=6)
    parser.add_argument(
        "--sync-todoist", action="store_true", help="pull Todoist before scoring"
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="do not generate demo data when the log is empty",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="use deterministic fallbacks only"
    )
    return parser.parse_args()


def _ensure_data(conn, db_path: Path | None, no_synthetic: bool) -> bool:
    if db.count_events(conn) > 0:
        return False
    if no_synthetic:
        raise SystemExit("The event log is empty. Run scripts/generate_synthetic.py first.")
    print(synthetic.SYNTHETIC_BANNER)
    print("Event log is empty; generating 400 synthetic tasks for the demo.\n")
    synthetic.generate_and_store(
        db_path, n=400, seed=7, reset=True, end_date=db.utcnow()
    )
    return True


def main() -> int:
    args = parse_args()
    as_of = db.utcnow()
    conn = db.connect(args.db)
    try:
        generated = False
        if args.sync_todoist:
            existing = db.load_events(conn)
            if existing and any(
                e.source == synthetic.SYNTHETIC_SOURCE for e in existing
            ):
                print(
                    "Warning: this log contains synthetic rows. Syncing real tasks "
                    "into it mixes demo and real data; use a separate --db to compare."
                )
            try:
                summary = todoist.sync(conn)
                print(
                    f"Todoist: {summary['active']} active, "
                    f"{summary['completed']} completed imported."
                )
            except todoist.TodoistError as exc:
                print(f"Todoist sync skipped: {exc}")
            if db.count_events(conn) == 0:
                print("No tasks available after sync; nothing to score.")
                return 1
        else:
            generated = _ensure_data(conn, args.db, args.no_synthetic)

        events = db.load_events(conn)
        synthetic_run = generated or any(
            e.source == synthetic.SYNTHETIC_SOURCE for e in events
        )
        if synthetic_run:
            print(synthetic.SYNTHETIC_BANNER)
            print("Predictions describe a fake user, not you.\n")

        if not args.model.exists():
            print(f"No fitted model at {args.model}.")
            print("Train one first:  python scripts/train_and_eval.py")
            print("(The first TabPFN run needs TABPFN_TOKEN in .env; after that it is offline.)")
            return 2

        model = oracle.load_oracle(args.model)
        scored = oracle.score_tasks(model, events, as_of=as_of)
        patterns = features.summarize_patterns(events, as_of=as_of)
        slot = patterns.get("best_completion_slot") or {}
        slot_label = str(slot.get("label", "your best slot"))

        llm = None
        if args.no_llm:
            print("LLM disabled (--no-llm); using deterministic interventions.")
        else:
            candidate = OllamaLLM()
            status = candidate.status()
            if status.available:
                llm = candidate
                print(f"LLM: {status.model} via Ollama")
            else:
                print(f"LLM: unavailable — {status.detail or 'Ollama not running'}")
                print("     Using deterministic interventions; they are labeled as fallbacks.")

        interventions = run_interventions(
            scored,
            llm=llm,
            slot_label=slot_label,
            patterns=patterns,
            max_tasks=args.max_interventions,
        )

        for item in interventions:
            db.record_intervention(
                conn,
                task_id=item.task_id,
                risk=item.risk,
                tier=item.tier,
                action=item.action,
                body=item.to_markdown(),
                run_at=as_of,
                source=item.source,
            )
        db.record_run(
            conn,
            run_at=as_of,
            n_tasks=len(events),
            n_at_risk=sum(1 for t in scored if t.tier in {"HIGH", "MED"}),
            label="nightly",
            synthetic=synthetic_run,
        )

        path = briefing.build_and_write(
            scored=scored,
            interventions=interventions,
            patterns=patterns,
            path=args.out,
            as_of=as_of,
            synthetic=synthetic_run,
        )
    finally:
        conn.close()

    high = sum(1 for t in scored if t.tier == "HIGH")
    med = sum(1 for t in scored if t.tier == "MED")
    low = sum(1 for t in scored if t.tier == "LOW")
    llm_used = sum(1 for i in interventions if i.source == "llm")
    print()
    print(f"Open tasks scored: {len(scored)}  (HIGH {high} · MED {med} · LOW {low})")
    print(
        f"Interventions: {len(interventions)} "
        f"({llm_used} written by Ollama, {len(interventions) - llm_used} fallbacks)"
    )
    print(f"Briefing: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
