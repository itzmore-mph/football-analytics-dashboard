import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "../data/shots_data.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

# Ensure Data Exists
if not os.path.exists(RAW_DATA_PATH):
    print(f"Error: Missing raw data file: {RAW_DATA_PATH}")
    exit()

# Load Data
df = pd.read_csv(RAW_DATA_PATH)

# Validate Required Columns
required_columns = ["player", "team", "shot_distance", "shot_angle", "goal_scored"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"Error: Missing columns in dataset: {missing_cols}")
    exit()



for col in ["body_part", "technique"]:
    if col in df.columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

# Normalize & Scale Data
df["shot_distance"] /= df["shot_distance"].max()
df["shot_angle"] /= df["shot_angle"].max()

# Save Processed Data
df.to_csv(PROCESSED_DATA_PATH, index=False)
print("Shot data preprocessed and saved successfully!")
