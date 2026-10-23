from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import db
import features
from collectors import synthetic

END = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def tasks() -> list[db.TaskEvent]:
    return synthetic.generate_tasks(n=400, seed=7, end_date=END)


def test_generates_about_400_unique_synthetic_rows(tasks: list[db.TaskEvent]) -> None:
    assert len(tasks) == 400
    assert len({t.task_id for t in tasks}) == 400
    assert {t.source for t in tasks} == {synthetic.SYNTHETIC_SOURCE}
    assert all(t.deferred_count >= 0 for t in tasks)


def test_labels_are_valid_and_causal(tasks: list[db.TaskEvent]) -> None:
    frame = features.build_features(tasks, as_of=END)
    assert set(frame[features.LABEL_COLUMN]).issubset({0, 1})

    for task in tasks:
        if task.completed_at is not None:
            assert task.completed_at >= task.created_at
            assert task.completed_at <= END
            expected = int(task.completed_at - task.created_at <= timedelta(hours=48))
            row = frame.loc[frame["task_id"] == task.task_id]
            if not row.empty:
                assert int(row.iloc[0][features.LABEL_COLUMN]) == expected
        elif task.created_at + timedelta(hours=48) <= END:
            assert frame.loc[frame["task_id"] == task.task_id].empty is False
            assert (
                int(frame.loc[frame["task_id"] == task.task_id, features.LABEL_COLUMN].iloc[0])
                == 0
            )


def test_signature_behaviors(tasks: list[db.TaskEvent]) -> None:
    dentist = [t for t in tasks if "dentist" in t.title.lower()]
    assert len(dentist) == 1
    assert dentist[0].deferred_count == 9
    assert dentist[0].completed_at is None

    coffee = [t for t in tasks if t.title == "Coffee with Sam"]
    assert len(coffee) == 12
    assert all(t.completed_at is not None for t in coffee)
    assert all(
        t.completed_at - t.created_at <= timedelta(hours=48) for t in coffee  # type: ignore[operator]
    )
    assert all(t.deferred_count == 0 for t in coffee)


def test_seed_is_deterministic() -> None:
    first = synthetic.generate_tasks(n=60, seed=42, end_date=END)
    second = synthetic.generate_tasks(n=60, seed=42, end_date=END)
    assert [(t.task_id, t.completed_at, t.deferred_count) for t in first] == [
        (t.task_id, t.completed_at, t.deferred_count) for t in second
    ]


def test_no_leakage_into_feature_frame(tasks: list[db.TaskEvent]) -> None:
    frame = features.build_features(tasks, as_of=END)

    for forbidden in ("completed_at", "deleted_at"):
        assert forbidden not in frame.columns
        assert forbidden not in features.FEATURE_COLUMNS

    # Open tasks whose 48h window is still running cannot be in training data.
    unresolved = [
        t.task_id
        for t in tasks
        if t.is_open and t.created_at + timedelta(hours=48) > END
    ]
    assert not set(unresolved) & set(frame["task_id"])


def test_completion_rate_is_realistic(tasks: list[db.TaskEvent]) -> None:
    stats = synthetic.summarise(tasks, seed=7)
    assert 0.1 < stats["completion_rate_48h"] < 0.9
    assert stats["total"] == 400
    assert stats["source"] == "synthetic"
    assert "health" in stats["by_category"]


def test_generator_rejects_tiny_samples() -> None:
    with pytest.raises(ValueError):
        synthetic.generate_tasks(n=5)
