from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import db
import features
import oracle
from collectors import synthetic

END = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def sample(rows: int = 50) -> tuple[pd.DataFrame, list[db.TaskEvent]]:
    tasks = synthetic.generate_tasks(n=70, seed=11, end_date=END)
    frame = features.build_features(tasks, as_of=END)
    frame = frame.sort_values("created_at").head(rows).reset_index(drop=True)
    assert frame[features.LABEL_COLUMN].nunique() == 2, "fixture needs both classes"
    return frame, tasks


class StubClassifier:
    """A cheap stand-in that exercises the wrapper, not TabPFN's accuracy."""

    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.n_features: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "StubClassifier":
        self.n_features = X.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risk = np.clip(0.05 + 0.05 * X["deferred_count"].to_numpy(dtype=float), 0.01, 0.99)
        return np.column_stack([1.0 - risk, risk])


def test_fit_predict_save_load_roundtrip(tmp_path) -> None:
    frame, tasks = sample(50)
    X = features.feature_frame(frame)
    y = frame[features.LABEL_COLUMN].to_numpy()

    model = oracle.TaskRiskOracle(classifier=StubClassifier())
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (50,)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    assert model.training_rows_ == 50

    path = model.save(tmp_path / "oracle.joblib")
    loaded = oracle.TaskRiskOracle.load(path)
    assert loaded.feature_columns_ == features.FEATURE_COLUMNS
    np.testing.assert_allclose(loaded.predict_proba(X), probabilities)

    # Scoring emits one tiered row per still-open task, risk-sorted.
    scored = oracle.score_tasks(loaded, tasks, as_of=END)
    assert scored
    assert all(item.tier in {"HIGH", "MED", "LOW"} for item in scored)
    assert [item.risk for item in scored] == sorted(
        [item.risk for item in scored], reverse=True
    )


def test_feature_columns_are_enforced_after_fit() -> None:
    frame, _ = sample(50)
    X = features.feature_frame(frame)
    model = oracle.TaskRiskOracle(classifier=StubClassifier()).fit(
        X, frame[features.LABEL_COLUMN].to_numpy()
    )
    with pytest.raises(ValueError, match="feature columns changed"):
        model.predict_proba(X.drop(columns=["title_len"]))


def test_single_class_training_is_rejected() -> None:
    frame, _ = sample(50)
    X = features.feature_frame(frame)
    y = np.zeros(len(frame), dtype=int)
    with pytest.raises(ValueError, match="both classes"):
        oracle.TaskRiskOracle(classifier=StubClassifier()).fit(X, y)


def test_tabpfn_fit_predict_roundtrip(tmp_path) -> None:
    """The real thing: TabPFN-3.5, 50 rows, saved and reloaded."""
    frame, _ = sample(50)
    X = features.feature_frame(frame)
    y = frame[features.LABEL_COLUMN].to_numpy()

    model = oracle.TaskRiskOracle(device="cpu")
    try:
        model.fit(X, y)
    except oracle.TabPFNUnavailable as exc:
        pytest.skip(f"TabPFN-3.5 weights unavailable in this environment: {exc}")

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (50,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert model.model_version == "V3_5"

    path = model.save(tmp_path / "tabpfn.joblib")
    loaded = oracle.TaskRiskOracle.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), probabilities, rtol=1e-6)
    # TabPFN v9 saves fit state as <name>.tabpfn_fit alongside the bundle
