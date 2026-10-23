#!/usr/bin/env python
"""Create the SQLite event log and print its tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    path = db.resolve_db_path(args.db)
    conn = db.connect(path)
    try:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        n_events = db.count_events(conn)
    finally:
        conn.close()
    print(f"Initialised {path}")
    print(f"Tables: {', '.join(tables)}")
    print(f"Tasks logged: {n_events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
