import os
import json
import pandas as pd

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
MATCH_DATA_PATH = os.path.join(DATA_DIR, "15946.json")
SHOTS_DATA_PATH = os.path.join(DATA_DIR, "shots_data.csv")

def extract_shot_data():
    if not os.path.exists(MATCH_DATA_PATH):
        print(f"Error: {MATCH_DATA_PATH} not found. Run fetch_statsbomb.py first.")
        return

    with open(MATCH_DATA_PATH, "r") as f:
        events = json.load(f)

    shot_data = []
    for event in events:
        if event["type"]["name"] == "Shot":
            shot_data.append({
                "player": event["player"]["name"],
                "team": event["team"]["name"],
                "x": event["location"][0],
                "y": event["location"][1],
                "shot_distance": event["shot"]["end_location"][0] - event["location"][0],
                "shot_angle": event["shot"]["end_location"][1] - event["location"][1],
                "goal_scored": int(event["shot"]["outcome"]["name"] == "Goal")
            })

    df = pd.DataFrame(shot_data)
    df.to_csv(SHOTS_DATA_PATH, index=False)
    print(f"Shot data saved to {SHOTS_DATA_PATH}")

if __name__ == "__main__":
    extract_shot_data()
    
"body_part": event["shot"]["body_part"]["name"],
"technique": event["shot"].get("technique", {}).get("name", "Unknown"),
"under_pressure": event.get("under_pressure", False),

