# src/train_xG_model.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
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
MODEL_PATH = MODEL_DIR / "xgb_xg_calibrated.joblib"
METRIC_PATH = MODEL_DIR / "xgboost_metrics.txt"
RELIABILITY_PATH = MODEL_DIR / "xgboost_reliability.csv"
FEATURES_PATH = MODEL_DIR / "features.txt"
CALIB_PNG = MODEL_DIR / "xg_calibration.png"
XG_BUCKETS_CSV = MODEL_DIR / "xg_buckets.csv"

# ---------- Load ----------
df = pd.read_csv(DATA_PATH).copy()
df = df.dropna(subset=["shot_distance", "shot_angle", "goal_scored"])

# Flags / time hygiene
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

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------- sklearn version guards ----------
major, minor = map(int, sklearn.__version__.split(".")[:2])
use_sparse_output = (major, minor) >= (1, 2)
use_estimator_kw = (major, minor) >= (1, 4)

numeric_transformer = MinMaxScaler()
if categorical_features:
    if use_sparse_output:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True
        )
    else:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse=True
        )
else:
    categorical_transformer = "drop"

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

# Fit only on train
X_train_t = preprocess.fit_transform(X_train)
X_test_t = preprocess.transform(X_test)
X_all_t = preprocess.transform(X)

# ---------- XGBoost ----------
pos = int(y_train.sum())
neg = int(y_train.shape[0] - pos)
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
    eval_metric="logloss",  # constructor param for older xgboost
)

# Early stopping robust across versions
try:
    base_model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_test_t, y_test)],
        verbose=False,
        early_stopping_rounds=75,
    )
except TypeError:
    base_model.fit(
        X_train_t, y_train, eval_set=[(X_test_t, y_test)], verbose=False
    )

# ---------- Eval (uncalibrated) ----------
proba_uncal = base_model.predict_proba(X_test_t)[:, 1]
try:
    auc_uncal = roc_auc_score(y_test, proba_uncal)
except ValueError:
    auc_uncal = float("nan")
brier_uncal = brier_score_loss(y_test, proba_uncal)

# ---------- Calibration ----------
CalCV = CalibratedClassifierCV
calibrator = (
    CalCV(estimator=base_model, cv="prefit", method="isotonic")
    if use_estimator_kw
    else CalCV(base_estimator=base_model, cv="prefit", method="isotonic")
)
calibrator.fit(X_train_t, y_train)

proba_cal = calibrator.predict_proba(X_test_t)[:, 1]
try:
    auc_cal = roc_auc_score(y_test, proba_cal)
except ValueError:
    auc_cal = float("nan")
brier_cal = brier_score_loss(y_test, proba_cal)

# ---------- xG for all ----------
df["xg"] = calibrator.predict_proba(X_all_t)[:, 1]

# ---------- Reliability table ----------
bins = np.linspace(0, 1, 11)
digitized = np.digitize(proba_cal, bins) - 1
rows = []
for b in range(len(bins) - 1):
    mask = digitized == b
    if mask.sum() == 0:
        continue
    rows.append(
        {
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "avg_pred": proba_cal[mask].mean(),
            "avg_true": y_test[mask].mean(),
            "n": int(mask.sum()),
        }
    )
reliability_df = pd.DataFrame(rows)
reliability_df.to_csv(RELIABILITY_PATH, index=False)

# ---------- Calibration plot ----------
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.scatter(reliability_df["avg_pred"], reliability_df["avg_true"])
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration (Isotonic)")
plt.tight_layout()
plt.savefig(CALIB_PNG, dpi=160)
plt.close()

# ---------- xG buckets ----------
df["xg_bucket"] = pd.cut(
    df["xg"],
    bins=[0, 0.05, 0.10, 0.20, 0.30, 1.0],
    labels=["0-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "0.30+"],
    include_lowest=True,
)
<<<<<<< HEAD

# IMPORTANT: keep the chain intact
(
    df.groupby("xg_bucket", observed=False)["goal_scored"]
    .agg(["mean", "count"])
    .reset_index()
    .to_csv(XG_BUCKETS_CSV, index=False)
=======
df.groupby(
    "xg_bucket"
    )
["goal_scored"].agg(["mean", "count"]).reset_index().to_csv(
    XG_BUCKETS_CSV, index=False
>>>>>>> parent of 3885182 (debugging)
)

# ---------- Save ----------
calibrated_pipeline = SkPipeline(
    steps=[("prep", preprocess), ("calibrated_clf", calibrator)]
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
        f"scale_pos_weight    : {scale_pos_weight:.3f}\n"
        f"sklearn version     : {sklearn.__version__}\n"
        f"xgboost version     : {xgb.__version__}\n"
        # f"Reliability CSV     : {RELIABILITY_PATH.name}\n"
        # f"Calibration Plot    : {CALIB_PNG.name}\n"
        # f"xG Buckets          : {XG_BUCKETS_CSV.name}\n"
    )

print("Training complete. Summary:")
print(f"Model:        {MODEL_PATH}")
print(f"Metrics:      {METRIC_PATH}")
print(f"Reliability:  {RELIABILITY_PATH}")
print(f"Calib plot:   {CALIB_PNG}")
print(f"xG buckets:   {XG_BUCKETS_CSV}")
print(f"Features:     {FEATURES_PATH}")
