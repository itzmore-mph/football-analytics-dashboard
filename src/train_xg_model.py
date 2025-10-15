"""Model training utilities for the xG classifier."""

from __future__ import annotations

import json
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier

from .config import settings
from .features_xg import FEATURE_COLUMNS, TARGET_COLUMN


def _split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware split by ``match_id`` to avoid data leakage."""

    if "match_id" not in df.columns:
        msg = "`match_id` column missing from training data"
        raise KeyError(msg)

    groups = df["match_id"].to_numpy()
    unique_groups = pd.unique(groups)
    n_groups = unique_groups.size

    if n_groups < 2:
        msg = "Need at least two matches to perform a train/validation split"
        raise ValueError(msg)

    if n_groups >= 5:
        splitter = GroupKFold(n_splits=5)
        idx_tr, idx_va = next(splitter.split(df, groups=groups))
    else:
        # Ensure at least one group ends up in the validation fold.
        test_size = 0.25 if n_groups > 3 else 1 / n_groups
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42
        )
        idx_tr, idx_va = next(splitter.split(df, groups=groups))

    return df.iloc[idx_tr].copy(), df.iloc[idx_va].copy()


def _build_model(kind: Literal["lr", "xgb"] = "xgb"):
    """Instantiate an un-fitted estimator for the requested ``kind``."""

    if kind == "lr":
        return LogisticRegression(max_iter=1000, solver="liblinear")

    if kind == "xgb":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )

    msg = "unknown model kind: use 'lr' or 'xgb'"
    raise ValueError(msg)


def train(
    kind: Literal["lr", "xgb"] = "xgb", calibration: str = "isotonic"
) -> dict:
    if not settings.processed_shots_csv.exists():
        msg = (
            "Processed shots CSV not found. "
            "Run `python -m src.cli preprocess` first."
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(settings.processed_shots_csv)
    train_df, val_df = _split_train_val(df)

    X_tr = train_df[FEATURE_COLUMNS].values
    y_tr = train_df[TARGET_COLUMN].values
    X_va = val_df[FEATURE_COLUMNS].values
    y_va = val_df[TARGET_COLUMN].values

    base = _build_model(kind)

    method_lookup: dict[str, str] = {
        "isotonic": "isotonic",
        "sigmoid": "sigmoid",
        "platt": "sigmoid",
    }
    method = method_lookup.get(calibration.lower(), "isotonic")

    class_counts = np.bincount(y_tr.astype(int), minlength=2)
    min_class_count = int(class_counts.min()) if class_counts.size else 0
    val_class_count = np.unique(y_va).size

    if min_class_count < 2 or val_class_count < 2:
        base.fit(X_tr, y_tr)
        clf = base
        calibration_used = "none"
    else:
        cv_folds = min(3, min_class_count)
        clf = CalibratedClassifierCV(base, method=method, cv=cv_folds)
        clf.fit(X_tr, y_tr)
        calibration_used = f"{method}-cv{cv_folds}"

    proba = clf.predict_proba(X_va)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_va, proba)),
        "brier": float(brier_score_loss(y_va, proba)),
        "log_loss": float(log_loss(y_va, np.clip(proba, 1e-6, 1 - 1e-6))),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "features": FEATURE_COLUMNS,
        "model": kind,
        "calibration": calibration_used,
    }

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, settings.model_path)

    report_path = settings.models_dir / "model_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
