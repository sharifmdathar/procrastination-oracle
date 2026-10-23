from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import db
import features

BASE = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
AS_OF = BASE + timedelta(days=10)


def fixture_events() -> list[db.TaskEvent]:
    return [
        # finished 3h after creation -> positive
        db.TaskEvent(
            "done-quick",
            "Buy milk",
            "errands",
            BASE,
            BASE + timedelta(days=2),
            0,
            BASE + timedelta(hours=3),
        ),
        # finished 50h after creation -> negative
        db.TaskEvent(
            "done-slow",
            "File taxes",
            "admin",
            BASE,
            BASE + timedelta(days=1),
            4,
            BASE + timedelta(hours=50),
        ),
        # still open after the 48h window -> negative
        db.TaskEvent(
            "open-old",
            "Call dentist",
            "health",
            BASE,
            BASE + timedelta(days=2),
            9,
        ),
        # deleted before completion -> negative
        db.TaskEvent(
            "deleted",
            "Return package",
            "errands",
            BASE,
            None,
            2,
            None,
            BASE + timedelta(hours=2),
        ),
        # created 10h ago, still open -> label unknown
        db.TaskEvent(
            "fresh",
            "Coffee with Sam",
            "social",
            AS_OF - timedelta(hours=10),
            AS_OF + timedelta(hours=2),
            0,
        ),
    ]


def test_build_features_uses_raw_columns_and_causal_labels() -> None:
    frame = features.build_features(fixture_events(), as_of=AS_OF)

    indexed = frame.set_index("task_id")
    assert set(indexed.index) == {"done-quick", "done-slow", "open-old", "deleted"}
    assert indexed[features.LABEL_COLUMN].to_dict() == {
        "done-quick": 1,
        "done-slow": 0,
        "open-old": 0,
        "deleted": 0,
    }
    assert indexed.loc["done-quick", "title"] == "Buy milk"
    assert indexed.loc["done-slow", "title"] == "File taxes"
    assert frame["has_deadline"].dtype == bool
    assert frame["deferred_count"].dtype == "int64"

    # Label-derived fields must never be features.
    assert "completed_at" not in frame.columns
    assert "deleted_at" not in frame.columns
    assert set(features.FEATURE_COLUMNS).issubset(frame.columns)

    # Raw text is preserved: no encoding, no lowercasing, no vectorisation.
    assert indexed.loc["done-quick", "title"] == "Buy milk"
    assert indexed.loc["open-old", "category"] == "health"
    assert isinstance(indexed.loc["open-old", "day_of_week"], str)
    assert indexed.loc["open-old", "day_of_week"] == "Monday"


def test_relative_features_are_measured_at_as_of() -> None:
    frame = features.build_features(fixture_events(), as_of=AS_OF).set_index("task_id")

    assert frame.loc["open-old", "task_age_days"] == pytest.approx(10.0)
    assert frame.loc["open-old", "days_until_due"] == pytest.approx(-8.0)
    assert frame.loc["deleted", "days_until_due"] != frame.loc["deleted", "days_until_due"]
    assert frame.loc["done-quick", "title_len"] == len("Buy milk")


def test_unresolved_tasks_are_dropped_for_training_but_kept_for_inference() -> None:
    training = features.build_features(fixture_events(), as_of=AS_OF)
    assert "fresh" not in set(training["task_id"])

    live = features.build_features(
        fixture_events(), as_of=AS_OF, include_unresolved=True
    )
    assert "fresh" in set(live["task_id"])
    assert live.loc[live["task_id"] == "fresh", features.LABEL_COLUMN].isna().all()


def test_time_split_is_chronological_and_stable() -> None:
    frame = features.build_features(fixture_events(), as_of=AS_OF)
    shuffled = frame.sample(frac=1.0, random_state=3)

    train, test = features.time_split(shuffled, test_fraction=0.5)
    again_train, again_test = features.time_split(shuffled, test_fraction=0.5)

    assert list(train["task_id"]) == list(again_train["task_id"])
    assert list(test["task_id"]) == list(again_test["task_id"])
    assert len(train) + len(test) == len(frame)
    assert not set(train["task_id"]) & set(test["task_id"])
    assert train["created_at"].max() <= test["created_at"].min()


def test_risk_tiers() -> None:
    assert features.risk_tier(0.95) == "HIGH"
    assert features.risk_tier(0.60) == "HIGH"
    assert features.risk_tier(0.59) == "MED"
    assert features.risk_tier(0.30) == "MED"
    assert features.risk_tier(0.29) == "LOW"


def test_patterns_summary_is_json_safe() -> None:
    patterns = features.summarize_patterns(fixture_events(), as_of=AS_OF)

    assert patterns["n_tasks"] == 5
    assert patterns["n_labeled"] == 4
    assert 0.0 <= patterns["completion_rate_48h"] <= 1.0
    assert patterns["best_completion_slot"]["weekday"] in {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    }
    import json

    json.dumps(patterns)  # must not raise


def test_empty_frame_stays_typed() -> None:
    frame = features.empty_feature_frame()
    assert frame.empty
    assert features.LABEL_COLUMN in frame.columns
