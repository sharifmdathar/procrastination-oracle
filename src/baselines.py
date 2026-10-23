"""Baselines and metrics for the head-to-head comparison.

The logistic regression is deliberately a strong, conventionally preprocessed
baseline (TF-IDF over the title, one-hot categories, scaled numerics, median
imputation). TabPFN gets the raw frame. If the foundation model is worth its
download, it should win without any of that work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import features

NUMERIC_COLUMNS = ["days_until_due", "task_age_days", "deferred_count", "title_len"]
CATEGORICAL_COLUMNS = ["category", "day_of_week"]


@dataclass(frozen=True)
class EvalResult:
    name: str
    accuracy: float
    auc: float
    brier: float
    n_test: int
    positives: int
    notes: str = ""
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "auc": None if np.isnan(self.auc) else round(self.auc, 4),
            "brier": round(self.brier, 4),
            "n_test": self.n_test,
            "positives": self.positives,
            "notes": self.notes,
        }


def evaluate(name: str, y_true: Sequence[int], probabilities: Sequence[float]) -> EvalResult:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
    else:
        auc = float("nan")
    return EvalResult(
        name=name,
        accuracy=float(accuracy_score(y, pred)),
        auc=auc,
        brier=float(brier_score_loss(y, p)),
        n_test=len(y),
        positives=int(y.sum()),
    )


def majority_class(train: pd.DataFrame, test: pd.DataFrame) -> EvalResult:
    """Predict the training set's majority label with its prevalence."""
    prevalence = float(train[features.LABEL_COLUMN].mean())
    label = int(prevalence >= 0.5)
    probabilities = np.full(len(test), prevalence)
    result = evaluate("Majority class", test[features.LABEL_COLUMN], probabilities)
    return EvalResult(
        name=result.name,
        accuracy=result.accuracy,
        auc=result.auc,
        brier=result.brier,
        n_test=result.n_test,
        positives=result.positives,
        notes=f"always predicts {label} (train prevalence {prevalence:.2f})",
    )


def _logistic_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("title", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True), "title"),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("flag", "passthrough", ["has_deadline"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def logistic_regression(train: pd.DataFrame, test: pd.DataFrame) -> EvalResult:
    model = _logistic_pipeline()
    model.fit(
        train[features.FEATURE_COLUMNS],
        train[features.LABEL_COLUMN].astype(int),
    )
    probabilities = model.predict_proba(test[features.FEATURE_COLUMNS])[:, 1]
    result = evaluate("Logistic regression", test[features.LABEL_COLUMN], probabilities)
    return EvalResult(
        name=result.name,
        accuracy=result.accuracy,
        auc=result.auc,
        brier=result.brier,
        n_test=result.n_test,
        positives=result.positives,
        notes="tf-idf + one-hot + scaled",
    )


def format_comparison(
    results: Sequence[EvalResult],
    *,
    missing: Sequence[str] = (),
) -> str:
    """Render the money table: accuracy, AUC, Brier score."""
    header = f"{'Model':<22}{'Accuracy':>10}{'AUC':>8}{'Brier':>8}   Notes"
    lines = [header, "-" * len(header)]
    # Models that could not run still get a row, in their usual place, so the
    # comparison never silently disappears.
    for label in missing:
        lines.append(f"{label:<22}{'—':>9} {'—':>7} {'—':>7}   not run")
    for result in results:
        auc = "  n/a " if np.isnan(result.auc) else f"{result.auc:>7.3f}"
        lines.append(
            f"{result.name:<22}{result.accuracy:>9.3f} {auc} {result.brier:>7.3f}   {result.notes}"
        )
    lines.append("")
    if results:
        first = results[0]
        lines.append(
            f"Test set: {first.n_test} tasks, {first.positives} positives "
            f"({first.positives / first.n_test:.0%}); chronological split, no shuffling."
        )
    return "\n".join(lines)
