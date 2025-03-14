import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load Data
DATA_PATH = "../data/processed_shots.csv"
df = pd.read_csv(DATA_PATH)

# Feature Engineering
df["distance_squared"] = df["shot_distance"] ** 2
df["angle_squared"] = df["shot_angle"] ** 2

# Selecting Features & Target
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

# Save Model
joblib.dump(model, "../models/xgboost_xg_model.pkl")
print("Model saved successfully!")
