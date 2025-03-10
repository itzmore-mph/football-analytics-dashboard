import os
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "../models/xgboost_xg_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

# Load processed data
df = pd.read_csv(DATA_PATH)

# Prepare features
X = df[["shot_distance", "shot_angle"]]

# Load trained model
if not os.path.exists(MODEL_PATH):
    print("Error: Trained model not found!")
    exit()

model = joblib.load(MODEL_PATH)

# Predict xG values
df["predicted_xG"] = model.predict_proba(X)[:, 1]

# Save updated data
df.to_csv(DATA_PATH, index=False)
print(f"Updated data with predicted xG saved to {DATA_PATH}")
