import os
import pandas as pd

# Load raw shot data
RAW_DATA_PATH = "../data/shots_data.csv"
OUTPUT_PATH = "../data/processed_shots.csv"

def preprocess_shots():
    df = pd.read_csv(RAW_DATA_PATH)

    # Normalize shot coordinates
    df["x"] = df["x"] / 120  # Assuming pitch width is 120m
    df["y"] = df["y"] / 80   # Assuming pitch height is 80m

    # Save processed data
    df.to_csv(OUTPUT_PATH, index=False)
    print("Processed shot data saved!")

if __name__ == "__main__":
    preprocess_shots()

def preprocess_xg_data():
    # Define absolute paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "../data/shots_data.csv")
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

    # Ensure raw data file exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Missing raw data file: {RAW_DATA_PATH}")
        return

    # Load and preprocess data
    df = pd.read_csv(RAW_DATA_PATH)
    df["shot_distance"] /= df["shot_distance"].max()
    df["shot_angle"] /= df["shot_angle"].max()

    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Shot data preprocessed and saved!")

if __name__ == "__main__":
    preprocess_xg_data()
