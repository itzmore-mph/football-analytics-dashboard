import os
import json
import pandas as pd

# Define file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
RAW_JSON_PATH = os.path.join(DATA_DIR, "15946.json")  # Adjust as needed
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "shots_data.csv")

def extract_shots():
    """Extracts shots from StatsBomb JSON and saves them as a CSV."""
    if not os.path.exists(RAW_JSON_PATH):
        print(f"Missing raw match data file: {RAW_JSON_PATH}")
        return

    with open(RAW_JSON_PATH, "r") as file:
        events = json.load(file)

    shots = []
    for event in events:
        if event["type"]["name"] == "Shot":
            shot = {
                "player": event["player"]["name"],
                "team": event["team"]["name"],
                "minute": event["minute"],
                "second": event["second"],
                "x": event["location"][0] if "location" in event else None,
                "y": event["location"][1] if "location" in event else None,
                "shot_distance": None,  # Will calculate in preprocess_xG.py
                "shot_angle": None,     # Will calculate in preprocess_xG.py
                "shot_type": event["shot"]["technique"]["name"],
                "body_part": event["shot"]["body_part"]["name"],
                "goal_scored": 1 if event["shot"]["outcome"]["name"] == "Goal" else 0,
            }
            shots.append(shot)

    # Save as CSV
    df = pd.DataFrame(shots)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Shot data saved at: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    extract_shots()
