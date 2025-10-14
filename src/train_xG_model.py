# src/train_xg_model.py
from __future__ import annotations

import json
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
    """Group-aware split by match_id to avoid leakage."""
    groups = df["match_id"].values
    n_groups = pd.unique(groups).size

    if n_groups >= 5:
        gkf = GroupKFold(n_splits=5)
        idx_tr, idx_va = next(gkf.split(df, groups=groups))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        idx_tr, idx_va = next(gss.split(df, groups=groups))

    return df.iloc[idx_tr].copy(), df.iloc[idx_va].copy()


def _build_model(kind: str = "xgb"):
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
            n_jobs=-1,            # use all cores
            random_state=42,
        )
    raise ValueError("unknown model kind: use 'lr' or 'xgb'")


def train(kind: str = "xgb", calibration: str = "isotonic") -> dict:
    df = pd.read_csv(settings.processed_shots_csv)
    train_df, val_df = _split_train_val(df)

    X_tr = train_df[FEATURE_COLUMNS].values
    y_tr = train_df[TARGET_COLUMN].values
    X_va = val_df[FEATURE_COLUMNS].values
    y_va = val_df[TARGET_COLUMN].values

    base = _build_model(kind)
    method = "isotonic" if calibration == "isotonic" else "sigmoid"

    # Option A (default): calibrate with CV inside the training split
    clf = CalibratedClassifierCV(base, method=method, cv=3)

    # Option B (alternative): prefit on train, calibrate on validation
    # base.fit(X_tr, y_tr)
    # calibrator = CalibratedClassifierCV(base_estimator=base, method=method,
    #                                     cv="prefit")
    # calibrator.fit(X_va, y_va)
    # clf = calibrator

    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_va)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_va, proba)),
        "brier": float(brier_score_loss(y_va, proba)),
        "log_loss": float(log_loss(y_va, np.clip(proba, 1e-6, 1 - 1e-6))),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "features": FEATURE_COLUMNS,
        "model": kind,
        "calibration": method,
    }

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, settings.model_path)

    report_path = settings.models_dir / "model_report.json"
    report_path.write_text(json.dumps(metrics, indent=2))

    return metrics
