#!/usr/bin/env python
"""Populate the event log with a fake procrastinator.

Every row is tagged ``source='synthetic'``. This is demo data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import db  # noqa: E402
from collectors import synthetic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="append instead of replacing previous synthetic rows",
    )
    args = parser.parse_args()

    print(synthetic.SYNTHETIC_BANNER)
    stats = synthetic.generate_and_store(
        args.db, n=args.n, seed=args.seed, reset=not args.keep_existing
    )
    print(f"Wrote {stats['total']} synthetic tasks to {db.resolve_db_path(args.db)}")
    print(
        f"  completed within 48h: {stats['completed_within_48h']} "
        f"({stats['completion_rate_48h']:.0%})"
    )
    print(f"  still open: {stats['open']}   deleted: {stats['deleted']}")
    print(f"  avg deferrals: {stats['avg_deferred']}")
    print("\nSignature behaviors: 'Call the dentist' deferred 9x and never done,")
    print("'Coffee with Sam' completed on time every week.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
