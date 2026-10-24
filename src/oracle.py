"""TabPFN-3.5 risk model.

Predicts ``P(task is completed within 48h of being created)``. TabPFN needs no
preprocessing: the raw pandas frame (strings, numbers, missing values) goes
straight in. Features are never scaled or one-hot encoded here.

The fitted classifier is persisted with joblib so a nightly run loads a local
file and never touches the network. The TabPFN weights themselves are cached
by the library after the first download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

import db
import features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "oracle_tabpfn_v3_5.joblib"
MODEL_VERSION_NAME = "V3_5"
MODEL_DISPLAY_NAME = "TabPFN-3.5"

LICENSE_HINT = (
    "TabPFN-3.5 weights are license-gated and need a one-time acceptance.\n"
    "  1. Open https://ux.priorlabs.ai, log in and accept the license.\n"
    "  2. Copy the API key and set TABPFN_TOKEN in .env (or the environment).\n"
    "  3. Run the same command again. After the weights are cached, the app\n"
    "     works fully offline."
)


LICENSE_ACCEPT_URL = "https://ux.priorlabs.ai/account/licenses"
HF_GATED_HINT = (
    "The TabPFN-3.5 checkpoint lives in a gated Hugging Face repo.\n"
    "  1. Accept the terms at https://huggingface.co/Prior-Labs/tabpfn_3_5\n"
    "  2. Create a read token at https://huggingface.co/settings/tokens\n"
    "  3. Put HF_TOKEN=hf_... in .env and rerun."
)


class TabPFNUnavailable(RuntimeError):
    """Raised when TabPFN cannot be imported, licensed, or downloaded."""


def _diagnose_download_failure() -> str:
    """Ask the license server *why* the weights were refused.

    TabPFN's own error path reports "set TABPFN_TOKEN" even when a valid token
    is already configured, which sends people in circles. This distinguishes a
    rejected token from an unaccepted license, and falls back to the generic
    instructions if the server cannot be reached. Never raises: the caller is
    already handling a failure.
    """
    try:
        from tabpfn import browser_auth
        from tabpfn.settings import settings
    except Exception:  # pragma: no cover - depends on install
        return LICENSE_HINT

    try:
        token = browser_auth.get_cached_token()
        if not token:
            return LICENSE_HINT
        api_url = settings.tabpfn.auth_api_url
        if browser_auth.verify_token(token, api_url) is False:
            return (
                "The TABPFN_TOKEN in .env was rejected by the license server.\n"
                "  Copy the API key again from https://ux.priorlabs.ai/account\n"
                "  (no quotes, no trailing whitespace) and rerun."
            )
        license_name = browser_auth._get_license_name("tabpfn_3_5")
        accepted = browser_auth.check_license_accepted(token, api_url, license_name)
        if accepted is False:
            return (
                "Your TABPFN_TOKEN is valid, but this account has not accepted\n"
                f"the TabPFN-3.5 license ({license_name}).\n"
                f"  Accept it once at {LICENSE_ACCEPT_URL}\n"
                "  then rerun — the weights download on the next attempt."
            )
    except Exception:
        return LICENSE_HINT
    return LICENSE_HINT


def default_device() -> str:
    return os.environ.get("TABPFN_DEVICE", "cpu")


def _new_classifier(*, device: str | None = None, random_state: int = 0) -> Any:
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError as exc:  # pragma: no cover - depends on install
        raise TabPFNUnavailable(
            "tabpfn is not installed. Run: pip install 'tabpfn>=9.0.0'"
        ) from exc

    version = getattr(ModelVersion, MODEL_VERSION_NAME)
    extra: dict[str, Any] = {}
    n_estimators = os.environ.get("TABPFN_N_ESTIMATORS")
    if n_estimators:
        extra["n_estimators"] = int(n_estimators)
    try:
        return TabPFNClassifier.create_default_for_version(
            version,
            device=device or default_device(),
            random_state=random_state,
            **extra,
        )
    except Exception as exc:  # pragma: no cover - version specific
        raise TabPFNUnavailable(f"could not construct TabPFN-3.5: {exc}") from exc


def tabpfn_installed() -> bool:
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        return False
    return True


def license_configured() -> bool:
    """True when a token is available (env or a cached browser login)."""
    if os.environ.get("TABPFN_TOKEN"):
        return True
    cache = Path.home() / ".cache" / "tabpfn"
    return any(cache.glob("**/*token*")) if cache.exists() else False


@dataclass(frozen=True)
class ScoredTask:
    """An open task with its risk of being skipped."""

    task_id: str
    title: str
    category: str | None
    created_at: datetime
    due_at: datetime | None
    risk: float
    tier: str
    days_until_due: float | None
    deferred_count: int
    source: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "category": self.category,
            "created_at": db.to_iso(self.created_at),
            "due_at": db.to_iso(self.due_at),
            "risk": round(self.risk, 4),
            "tier": self.tier,
            "days_until_due": (
                round(self.days_until_due, 2) if self.days_until_due is not None else None
            ),
            "deferred_count": self.deferred_count,
            "source": self.source,
        }


class TaskRiskOracle:
    """Thin, persistable wrapper around ``TabPFNClassifier``."""

    def __init__(
        self,
        *,
        classifier: Any | None = None,
        device: str | None = None,
        random_state: int = 0,
    ) -> None:
        self._classifier = classifier
        self.device = device or default_device()
        self.random_state = random_state
        self.classifier_: Any | None = None
        self.feature_columns_: list[str] = []
        self.fitted_at_: datetime | None = None
        self.training_rows_: int = 0
        self.model_version: str = MODEL_VERSION_NAME

    @property
    def is_fitted(self) -> bool:
        return self.classifier_ is not None

    def fit(self, X: pd.DataFrame, y: Sequence[int] | np.ndarray) -> "TaskRiskOracle":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabPFN expects a pandas DataFrame; pass raw features")
        y_array = np.asarray(y).astype(int)
        if len(X) != len(y_array):
            raise ValueError("X and y have different lengths")
        if len(np.unique(y_array)) < 2:
            raise ValueError("training data needs both classes (0 and 1)")

        classifier = self._classifier or _new_classifier(
            device=self.device, random_state=self.random_state
        )
        try:
            classifier.fit(X, y_array)
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc).lower()
            if name == "TabPFNHuggingFaceGatedRepoError" or "gated" in message:
                raise TabPFNUnavailable(HF_GATED_HINT) from exc
            if "license" in message or name == "TabPFNLicenseError":
                raise TabPFNUnavailable(_diagnose_download_failure()) from exc
            raise

        self.classifier_ = classifier
        self.feature_columns_ = list(X.columns)
        self.fitted_at_ = db.utcnow()
        self.training_rows_ = len(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class (completed within 48h)."""
        if self.classifier_ is None:
            raise RuntimeError("oracle is not fitted; call fit() or load()")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabPFN expects a pandas DataFrame")
        if self.feature_columns_ and list(X.columns) != self.feature_columns_:
            raise ValueError(
                f"feature columns changed since fit: {list(X.columns)} != "
                f"{self.feature_columns_}"
            )
        probabilities = np.asarray(self.classifier_.predict_proba(X), dtype=float)
        classes = list(getattr(self.classifier_, "classes_", [0, 1]))
        if 1 not in classes:
            return np.zeros(len(X), dtype=float)
        return probabilities[:, classes.index(1)]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        """Persist the fitted oracle.

        TabPFN exposes ``save_fit_state``/``load_from_fit_state`` for fitted
        estimators, so the checkpoint itself is written next to the metadata
        file. Plain sklearn-like classifiers (used in tests) fall back to
        joblib inside the bundle.
        """
        if self.classifier_ is None:
            raise RuntimeError("cannot save an unfitted oracle")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        bundle: dict[str, Any] = {
            "classifier": None,
            "classifier_state": None,
            "feature_columns": self.feature_columns_,
            "model_version": self.model_version,
            "fitted_at": db.to_iso(self.fitted_at_),
            "training_rows": self.training_rows_,
            "device": self.device,
        }
        if hasattr(self.classifier_, "save_fit_state"):
            state_path = target.parent / (target.name + ".tabpfn_fit")
            self.classifier_.save_fit_state(state_path)
            bundle["classifier_state"] = state_path.name
        else:
            bundle["classifier"] = self.classifier_
        joblib.dump(bundle, target)
        return target

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "TaskRiskOracle":
        target = Path(path)
        bundle = joblib.load(target)
        instance = cls()
        instance.feature_columns_ = list(bundle.get("feature_columns", []))
        instance.model_version = bundle.get("model_version", MODEL_VERSION_NAME)
        instance.training_rows_ = int(bundle.get("training_rows", 0))
        instance.fitted_at_ = db.parse_dt(bundle.get("fitted_at"))
        instance.device = bundle.get("device", default_device())
        instance.classifier_ = bundle.get("classifier")

        state_name = bundle.get("classifier_state")
        if instance.classifier_ is None and state_name:
            try:
                from tabpfn import TabPFNClassifier
            except ImportError as exc:  # pragma: no cover - depends on install
                raise TabPFNUnavailable(
                    "this oracle was saved by TabPFN; install tabpfn to load it"
                ) from exc
            state_path = target.parent / state_name
            if not state_path.exists():
                raise FileNotFoundError(
                    f"missing TabPFN fit state next to {target}: {state_path}"
                )
            instance.classifier_ = TabPFNClassifier.load_from_fit_state(
                state_path, device=instance.device
            )
        if instance.classifier_ is None:
            raise ValueError(f"no classifier found in {target}")
        return instance


def load_oracle(path: str | Path = DEFAULT_MODEL_PATH) -> TaskRiskOracle:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"no fitted oracle at {model_path}. Run: python scripts/train_and_eval.py"
        )
    return TaskRiskOracle.load(model_path)


def score_tasks(
    oracle: TaskRiskOracle,
    events: Sequence[db.TaskEvent],
    as_of: datetime | None = None,
) -> list[ScoredTask]:
    """Score every still-open task and sort by risk (highest first)."""
    as_of = (as_of or db.utcnow()).astimezone(timezone.utc)
    open_tasks = [e for e in events if e.is_open and e.created_at <= as_of]
    if not open_tasks:
        return []

    frame = features.build_features(open_tasks, as_of=as_of, include_unresolved=True)
    if frame.empty:
        return []
    probabilities = oracle.predict_proba(features.feature_frame(frame))

    by_id = {task.task_id: task for task in open_tasks}
    scored: list[ScoredTask] = []
    for task_id, probability in zip(frame["task_id"], probabilities):
        task = by_id[str(task_id)]
        due_days = (
            (task.due_at - as_of).total_seconds() / 86400
            if task.due_at is not None
            else None
        )
        scored.append(
            ScoredTask(
                task_id=task.task_id,
                title=task.title,
                category=task.category,
                created_at=task.created_at,
                due_at=task.due_at,
                risk=float(probability),
                tier=features.risk_tier(float(probability)),
                days_until_due=due_days,
                deferred_count=task.deferred_count,
                source=task.source,
            )
        )
    return sorted(scored, key=lambda item: (-item.risk, item.task_id))
