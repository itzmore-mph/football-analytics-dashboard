import os
import pandas as pd

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
