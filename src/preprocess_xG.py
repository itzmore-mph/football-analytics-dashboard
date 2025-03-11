import os
import pandas as pd

# Define file paths dynamically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "../data/shots_data.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

def preprocess_shots():
    """Loads raw shot data, normalizes coordinates, and saves processed data."""
    
    # Ensure raw data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Missing raw data file: {DATA_PATH}")
        return

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Normalize shot coordinates (assuming pitch dimensions are 120m x 80m)
    if "x" in df.columns and "y" in df.columns:
        df["x"] = df["x"] / 120
        df["y"] = df["y"] / 80

    # Normalize shot distance & angle if available
    if "shot_distance" in df.columns:
        df["shot_distance"] /= df["shot_distance"].max()
    if "shot_angle" in df.columns:
        df["shot_angle"] /= df["shot_angle"].max()

    # Save processed data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Processed shot data saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_shots()
