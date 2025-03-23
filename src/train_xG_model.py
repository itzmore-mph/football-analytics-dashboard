import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Paths
DATA_PATH = os.path.join("data", "processed_shots.csv")
MODEL_PATH = os.path.join("models", "xgboost_xg_model.pkl")
METRIC_PATH = os.path.join("models", "xgboost_metrics.txt")

# Load Data
df = pd.read_csv(DATA_PATH)

# Feature Engineering
df["distance_squared"] = df["shot_distance"] ** 2
df["angle_squared"] = df["shot_angle"] ** 2

# Optional advanced features (if available)
if "under_pressure" in df.columns:
    df["under_pressure"] = df["under_pressure"].astype(int)
    features = ["shot_distance", "shot_angle", "distance_squared", "angle_squared", "under_pressure"]
else:
    features = ["shot_distance", "shot_angle", "distance_squared", "angle_squared"]

target = "goal_scored"

X = df[features]
y = df[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    colsample_bytree=0.7,
    subsample=0.8,
    random_state=42,
)

# Train Model
model.fit(X_train, y_train)

# Evaluate
preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
print(f"ROC AUC Score: {auc:.4f}")

# Save Model and Metric
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
with open(METRIC_PATH, "w") as f:
    f.write(f"ROC AUC Score: {auc:.4f}\n")

print("Model and metrics saved successfully!")
