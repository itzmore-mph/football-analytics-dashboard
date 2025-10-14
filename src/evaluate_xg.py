from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.calibration import calibration_curve

from .config import settings
from .features_xg import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate_and_plot(n_bins: int = 10) -> dict:
    df = pd.read_csv(settings.processed_shots_csv)
    model = joblib.load(settings.model_path)
    proba = model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]

    frac_pos, mean_pred = calibration_curve(
        df[TARGET_COLUMN].values, proba, n_bins=n_bins, strategy="quantile"
    )

    # Plot reliability
    fig = plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    settings.plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.plots_dir / "calibration.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)

    report = {
        "n": int(len(df)),
        "bins": int(n_bins),
        "plot": str(out_path),
    }
    evaluation_path = settings.models_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(report, indent=2))
    return report
