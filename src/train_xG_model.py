from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed_shots.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgboost_xg_model_calibrated.joblib"
METRIC_PATH = MODEL_DIR / "xgboost_metrics.txt"
RELIABILITY_PATH = MODEL_DIR / "xgboost_reliability.csv"
FEATURES_PATH = MODEL_DIR / "features.txt"
CALIB_PNG = MODEL_DIR / "xg_calibration.png"
XG_BUCKETS_CSV = MODEL_DIR / "xg_buckets.csv"

# ---------- Load ----------
df = pd.read_csv(DATA_PATH).copy()
df = df.dropna(subset=["shot_distance", "shot_angle", "goal_scored"])

for col in ["under_pressure", "is_penalty", "is_header"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
for col in ["minute", "second"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median()).astype(float)

# ---------- Features ----------
df["distance_squared"] = df["shot_distance"] ** 2
df["angle_squared"] = df["shot_angle"] ** 2
df["distance_angle_interaction"] = df["shot_distance"] * df["shot_angle"]

numeric_features = [
    "shot_distance",
    "shot_angle",
    "distance_squared",
    "angle_squared",
    "distance_angle_interaction",
]
categorical_features = []
if "body_part" in df.columns:
    categorical_features.append("body_part")
if "technique" in df.columns:
    categorical_features.append("technique")
extra_numeric = []
for col in ["under_pressure", "is_penalty", "is_header", "minute", "second"]:
    if col in df.columns:
        extra_numeric.append(col)

features = numeric_features + extra_numeric + categorical_features
target = "goal_scored"

X = df[features].copy()
y = df[target].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_transformer = MinMaxScaler()
categorical_transformer = (
    OneHotEncoder(
        handle_unknown="ignore", sparse=True
    ) if categorical_features else "drop"
)
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features + extra_numeric),
        *(
            [("cat", categorical_transformer, categorical_features)]
            if categorical_features
            else []
        ),
    ]
)

X_train_t = preprocess.fit_transform(X_train)
X_test_t = preprocess.transform(X_test)
X_all_t = preprocess.transform(X)

pos = y_train.sum()
neg = y_train.shape[0] - pos
scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

base_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    colsample_bytree=0.7,
    subsample=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
)
base_model.fit(
    X_train_t, y_train,
    eval_set=[(X_test_t, y_test)],
    eval_metric="logloss",
    verbose=False,
    early_stopping_rounds=75
)

proba_uncal = base_model.predict_proba(X_test_t)[:, 1]
auc_uncal = roc_auc_score(y_test, proba_uncal)
brier_uncal = brier_score_loss(y_test, proba_uncal)

calibrator = CalibratedClassifierCV(
    base_estimator=base_model,
    cv="prefit",
    method="isotonic"
)
calibrator.fit(X_train_t, y_train)

proba_cal = calibrator.predict_proba(X_test_t)[:, 1]
auc_cal = roc_auc_score(y_test, proba_cal)
brier_cal = brier_score_loss(y_test, proba_cal)

df["xG"] = calibrator.predict_proba(X_all_t)[:, 1]

# Reliability
bins = np.linspace(0, 1, 11)
digitized = np.digitize(proba_cal, bins) - 1
rows = []
for b in range(len(bins) - 1):
    mask = digitized == b
    if mask.sum() == 0:
        continue
    rows.append({
        "bin_left": bins[b],
        "bin_right": bins[b + 1],
        "avg_pred": proba_cal[mask].mean(),
        "avg_true": y_test[mask].mean(),
        "n": int(mask.sum()),
    })
reliability_df = pd.DataFrame(rows)
reliability_df.to_csv(RELIABILITY_PATH, index=False)

# Calibration plot
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.scatter(reliability_df["avg_pred"], reliability_df["avg_true"])
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration (Isotonic)")
plt.tight_layout()
plt.savefig(CALIB_PNG, dpi=160)
plt.close()

# Buckets
df["xg_bucket"] = pd.cut(
    df["xG"],
    bins=[0, 0.05, 0.10, 0.20, 0.30, 1.0],
    labels=["0–0.05", "0.05–0.10", "0.10–0.20", "0.20–0.30", "0.30+"],
    include_lowest=True,
)
df.groupby("xg_bucket")["goal_scored"] \
    .agg(["mean", "count"]) \
    .reset_index() \
    .to_csv(MODEL_DIR / "xg_buckets.csv", index=False)

# Save all
calibrated_pipeline = SkPipeline(
    steps=[
        ("prep", preprocess),
        ("calibrated_clf", calibrator),
    ]
)
joblib.dump(calibrated_pipeline, MODEL_PATH)
FEATURES_PATH.write_text("\n".join(features), encoding="utf-8")
df.to_csv(DATA_PATH, index=False)

with open(METRIC_PATH, "w", encoding="utf-8") as f:
    f.write(
        f"Uncalibrated  ROC AUC: {auc_uncal:.4f}\n"
        f"Uncalibrated  Brier  : {brier_uncal:.4f}\n"
        f"Calibrated    ROC AUC: {auc_cal:.4f}\n"
        f"Calibrated    Brier  : {brier_cal:.4f}\n"
        f"Reliability CSV     : {RELIABILITY_PATH.name}\n"
        f"Calibration Plot    : {Path(CALIB_PNG).name}\n"
        f"xG Buckets          : xg_buckets.csv\n"
        f"scale_pos_weight    : {scale_pos_weight:.3f}\n"
    )

print(
    "Saved calibrated xG pipeline, metrics, calibration plot, buckets, "
    "and updated processed_shots.csv"
)
print(f"Model:        {MODEL_PATH}")
print(f"Metrics:      {METRIC_PATH}")
print(f"Reliability:  {RELIABILITY_PATH}")
print(f"Calib plot:   {CALIB_PNG}")
print(f"xG buckets:   {MODEL_DIR / 'xg_buckets.csv'}")
print(f"Features:     {FEATURES_PATH}")
