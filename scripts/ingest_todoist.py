#!/usr/bin/env python
"""One-shot Todoist import into the local event log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

import db  # noqa: E402
from collectors import todoist  # noqa: E402

load_dotenv(ROOT / ".env")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--token", default=None, help="defaults to TODOIST_API_TOKEN")
    parser.add_argument("--since-days", type=int, default=90)
    args = parser.parse_args()

    conn = db.connect(args.db)
    try:
        summary = todoist.sync(conn, token=args.token, since_days=args.since_days)
    except todoist.TodoistError as exc:
        print(f"Todoist sync failed: {exc}")
        return 1
    finally:
        conn.close()

    print(f"Imported Todoist tasks into {db.resolve_db_path(args.db)}")
    print(f"  active: {summary['active']}")
    print(f"  completed (since {summary['since']}): {summary['completed']}")
    if summary["skipped_completed"] == -1:
        print("  completed endpoint unsupported; active tasks only")
    elif summary["skipped_completed"]:
        print(f"  skipped completed without added_at: {summary['skipped_completed']}")
    print(f"  rows written: {summary['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
