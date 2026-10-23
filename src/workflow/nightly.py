"""Temporal workflow + worker for the 9pm nightly run.

Optional infrastructure. The app works without it via
``python scripts/run_nightly.py``. After ``pip install -e .``:

    temporal server start-dev                    # local Temporal
    python -m workflow.nightly --schedule        # worker + 21:00 schedule
    python -m workflow.nightly --once            # run the workflow immediately
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from temporalio import workflow  # noqa: E402
from temporalio.client import (  # noqa: E402
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleSpec,
)
from temporalio.common import RetryPolicy  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

with workflow.unsafe.imports_passed_through():
    from workflow import activities

TASK_QUEUE = activities.TASK_QUEUE
SCHEDULE_ID = "procrastination-oracle-nightly"


@workflow.defn
class NightlyOracleWorkflow:
    """One insomnia-free pass: sync, score, intervene, brief."""

    @workflow.run
    async def run(self, request: dict) -> dict:
        timeout = timedelta(minutes=10)
        retry = RetryPolicy(maximum_attempts=2)
        sync_result = await workflow.execute_activity(
            activities.sync_tasks,
            request,
            start_to_close_timeout=timeout,
            retry_policy=retry,
        )
        prediction = await workflow.execute_activity(
            activities.predict_risks,
            request,
            start_to_close_timeout=timeout,
            retry_policy=retry,
        )
        result = await workflow.execute_activity(
            activities.take_interventions,
            request,
            prediction,
            start_to_close_timeout=timeout,
            retry_policy=retry,
        )
        briefing_result = await workflow.execute_activity(
            activities.write_briefing,
            request,
            prediction,
            result,
            start_to_close_timeout=timeout,
            retry_policy=retry,
        )
        return {
            "sync": sync_result,
            "n_tasks": prediction.get("n_tasks"),
            "n_scored": len(prediction.get("scored", [])),
            "n_interventions": len(result.get("interventions", [])),
            "briefing": briefing_result.get("path"),
        }


def _request_from_args(args: argparse.Namespace) -> dict:
    return {
        "db": str(args.db) if args.db else None,
        "model": str(args.model),
        "out": str(args.out),
        "max_interventions": args.max_interventions,
        "no_llm": args.no_llm,
        "since_days": 90,
    }


def _activities() -> list:
    return [
        activities.sync_tasks,
        activities.predict_risks,
        activities.take_interventions,
        activities.write_briefing,
    ]


async def _run_worker(args: argparse.Namespace, client: Client) -> None:
    if args.schedule:
        try:
            await client.create_schedule(
                SCHEDULE_ID,
                Schedule(
                    action=ScheduleActionStartWorkflow(
                        NightlyOracleWorkflow.run,
                        _request_from_args(args),
                        id=SCHEDULE_ID,
                        task_queue=TASK_QUEUE,
                    ),
                    spec=ScheduleSpec(
                        cron_expressions=[args.cron],
                        time_zone_name=args.timezone,
                    ),
                ),
            )
            print(f"Created schedule {SCHEDULE_ID!r} ({args.cron}, {args.timezone}).")
        except ScheduleAlreadyRunningError:
            print(f"Schedule {SCHEDULE_ID!r} already exists; leaving it untouched.")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NightlyOracleWorkflow],
        activities=_activities(),
        # The activities are synchronous (they use sqlite3 and requests); the
        # thread pool keeps them from blocking the worker's event loop.
        activity_executor=ThreadPoolExecutor(max_workers=4),
    )
    print(f"Worker listening on {TASK_QUEUE!r}. Ctrl-C to stop.")
    await worker.run()


async def _run_once(args: argparse.Namespace, client: Client) -> None:
    """Self-contained one-shot run: ephemeral worker + one workflow."""
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NightlyOracleWorkflow],
        activities=_activities(),
        activity_executor=ThreadPoolExecutor(max_workers=4),
    )
    async with worker:
        handle = await client.start_workflow(
            NightlyOracleWorkflow.run,
            _request_from_args(args),
            id=f"nightly-manual-{os.getpid()}-{int(time.time())}",
            task_queue=TASK_QUEUE,
        )
        print(f"Started {handle.id}; waiting for the result...")
        print(await handle.result())


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    parser.add_argument("--namespace", default=os.environ.get("TEMPORAL_NAMESPACE", "default"))
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--model", type=Path, default=root / "models" / "oracle_tabpfn_v3_5.joblib"
    )
    parser.add_argument("--out", type=Path, default=root / "morning_briefing.md")
    parser.add_argument("--max-interventions", type=int, default=6)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--schedule", action="store_true", help="create the 21:00 schedule")
    parser.add_argument("--cron", default="0 21 * * *")
    parser.add_argument(
        "--timezone",
        default=os.environ.get("ORACLE_TZ", "UTC"),
        help="IANA timezone for the cron schedule",
    )
    parser.add_argument("--once", action="store_true", help="start one workflow, then exit")
    args = parser.parse_args()

    async def runner() -> None:
        client = await Client.connect(args.address, namespace=args.namespace)
        if args.once:
            await _run_once(args, client)
        else:
            await _run_worker(args, client)

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
