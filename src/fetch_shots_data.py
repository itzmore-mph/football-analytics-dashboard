import os
import json
import pandas as pd

def extract_shots(json_file, output_csv):
    # Load JSON data
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract shot events
    shot_data = []
    for event in data:
        if event["type"]["name"] == "Shot":
            shot_data.append({
                "player": event["player"]["name"],
                "team": event["team"]["name"],
                "minute": event["minute"],
                "second": event["second"],
                "shot_distance": event["location"][0] if "location" in event else None,
                "shot_angle": event["location"][1] if "location" in event else None,
                "shot_type": event["shot"].get("technique", {}).get("name", "Unknown"),
                "body_part": event["shot"].get("bodyPart", {}).get("name", "Unknown"),
                "goal_scored": 1 if event["shot"]["outcome"]["name"] == "Goal" else 0,
                "xG": event["shot"].get("xG", 0)  # Use 0 if xG is missing
            })

    # Convert to DataFrame
    df_shots = pd.DataFrame(shot_data)

    # Save as CSV
    df_shots.to_csv(output_csv, index=False)
    print(f"Shot data saved to {output_csv}")

# Define file paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(PROJECT_ROOT, "../data/15946.json")  # Update to your match file
CSV_PATH = os.path.join(PROJECT_ROOT, "../data/shots_data.csv")

# Run the function
extract_shots(JSON_PATH, CSV_PATH)
