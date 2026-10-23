#!/usr/bin/env python
"""THE money output: TabPFN-3.5 vs logistic regression vs majority class.

Run:
    python scripts/train_and_eval.py

If the event log is empty, a clearly-labelled synthetic user is generated
first. The split is always chronological — the last 25% of tasks by creation
time form the test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

import baselines  # noqa: E402
import db  # noqa: E402
import features  # noqa: E402
import oracle  # noqa: E402
from collectors import synthetic  # noqa: E402

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="SQLite event log")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--model-out", type=Path, default=oracle.DEFAULT_MODEL_PATH
    )
    parser.add_argument("--json-out", type=Path, default=ROOT / "models" / "eval_metrics.json")
    parser.add_argument(
        "--n-synthetic", type=int, default=400, help="rows if the log is empty"
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def ensure_data(db_path: Path | None, n: int, seed: int) -> tuple[list[db.TaskEvent], bool]:
    conn = db.connect(db_path)
    try:
        events = db.load_events(conn)
        if not events:
            print(synthetic.SYNTHETIC_BANNER)
            print(f"Event log is empty; generating {n} synthetic tasks.\n")
            synthetic.generate_and_store(
                db_path, n=n, seed=seed, reset=True, end_date=db.utcnow()
            )
            events = db.load_events(conn)
        synthetic_rows = sum(1 for e in events if e.source == synthetic.SYNTHETIC_SOURCE)
    finally:
        conn.close()
    return events, synthetic_rows > 0


def main() -> int:
    args = parse_args()
    events, has_synthetic = ensure_data(args.db, args.n_synthetic, args.seed)

    if has_synthetic:
        print(synthetic.SYNTHETIC_BANNER)
        print("Metrics below describe a fake user, not you.\n")

    frame = features.build_features(events, as_of=db.utcnow())
    if len(frame) < 20:
        print(f"Only {len(frame)} labelled tasks. Generate more data first.")
        return 1
    if frame[features.LABEL_COLUMN].nunique() < 2:
        print("The event log contains a single class; cannot evaluate a classifier.")
        return 1

    train, test = features.time_split(frame, args.test_fraction)
    X_train = features.feature_frame(train)
    y_train = train[features.LABEL_COLUMN].to_numpy()
    X_test = features.feature_frame(test)
    y_test = test[features.LABEL_COLUMN].to_numpy()

    print("=" * 74)
    print("PROCRASTINATION ORACLE — model comparison")
    print("=" * 74)
    print(f"Tasks: {len(frame)}   train: {len(train)}   test: {len(test)}")
    print(
        "Train window: "
        f"{train['created_at'].min():%Y-%m-%d} → {train['created_at'].max():%Y-%m-%d}"
    )
    print(
        "Test window:  "
        f"{test['created_at'].min():%Y-%m-%d} → {test['created_at'].max():%Y-%m-%d}"
    )
    print(
        f"Label rate: train {y_train.mean():.2f} / test {y_test.mean():.2f}   "
        f"(split is chronological, never random)"
    )
    print(
        "TabPFN input: raw DataFrame "
        f"({len(features.FEATURE_COLUMNS)} columns, no scaling/encoding)\n"
    )

    results: list[baselines.EvalResult] = []
    missing: list[str] = []
    payload: dict[str, object] = {
        "n_tasks": len(frame),
        "n_train": len(train),
        "n_test": len(test),
        "train_start": train["created_at"].min().isoformat(),
        "test_start": test["created_at"].min().isoformat(),
        "label_rate_train": round(float(y_train.mean()), 4),
        "label_rate_test": round(float(y_test.mean()), 4),
        "synthetic": has_synthetic,
        "models": {},
    }

    tabpfn_result: baselines.EvalResult | None = None
    try:
        model = oracle.TaskRiskOracle(device=oracle.default_device()).fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        tabpfn_result = baselines.evaluate("TabPFN-3.5", y_test, probabilities)
        tabpfn_result = baselines.EvalResult(
            name=tabpfn_result.name,
            accuracy=tabpfn_result.accuracy,
            auc=tabpfn_result.auc,
            brier=tabpfn_result.brier,
            n_test=tabpfn_result.n_test,
            positives=tabpfn_result.positives,
            notes="raw features, zero preprocessing",
        )
        saved = model.save(args.model_out)
        print(f"Saved fitted oracle → {saved}")
        print(f"  model cache/token: {'configured' if oracle.license_configured() else 'cached weights'}")
    except oracle.TabPFNUnavailable as exc:
        missing.append("TabPFN-3.5")
        print("!! TabPFN-3.5 could not run in this environment.")
        print(str(exc))
        print()
    except Exception as exc:  # pragma: no cover - environment specific
        missing.append("TabPFN-3.5")
        print(f"!! TabPFN-3.5 failed: {exc}\n")

    logistic = baselines.logistic_regression(train, test)
    majority = baselines.majority_class(train, test)
    results = [r for r in [tabpfn_result, logistic, majority] if r is not None]

    print(baselines.format_comparison(results, missing=missing))
    print()

    for result in results:
        payload["models"][result.name] = result.as_dict()
    if missing:
        payload["unavailable"] = missing

    try:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Metrics written to {args.json_out}")
    except OSError as exc:
        print(f"(could not write metrics json: {exc})")

    if tabpfn_result is not None:
        _print_high_risk(model, events)

    if missing:
        print()
        print("The comparison is incomplete. " + oracle.LICENSE_HINT.splitlines()[0])
        print("See .env.example → TABPFN_TOKEN.")
    return 0


def _print_high_risk(model: oracle.TaskRiskOracle, events: list[db.TaskEvent]) -> None:
    scored = oracle.score_tasks(model, events)
    if not scored:
        return
    print("Highest-risk open tasks (48h completion probability):")
    for task in scored[:5]:
        due = f"{task.due_at:%b %d}" if task.due_at else "no deadline"
        print(
            f"  {task.risk:>5.0%}  {task.tier:<4}  {task.title[:42]:<42} "
            f"(due {due}, deferred {task.deferred_count}x)"
        )
    print()


if __name__ == "__main__":
    raise SystemExit(main())
