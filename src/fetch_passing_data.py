import json
import pandas as pd
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_JSON_PATH = os.path.join(PROJECT_ROOT, "../data/15946.json")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "../data/passing_data.csv")

def extract_passing_data():
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    passing_data = []
    
    for event in data:
        if event["type"]["name"] == "Pass":
            pass_event = {
                "passer": event["player"]["name"],
                "receiver": event["pass"].get("recipient", {}).get("name", None),
                "team": event["team"]["name"],
                "minute": event["minute"],
                "second": event["second"],
                "start_x": event["location"][0],  # Starting x coordinate
                "start_y": event["location"][1],  # Starting y coordinate
                "end_x": event["pass"]["end_location"][0],
                "end_y": event["pass"]["end_location"][1],
                "pass_length": event["pass"].get("length", None),
                "pass_angle": event["pass"].get("angle", None),
                "pass_outcome": event["pass"].get("outcome", {}).get("name", "Complete"),
            }
            passing_data.append(pass_event)

    df = pd.DataFrame(passing_data)

    # Save the extracted data
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Passing data extracted and saved!")

if __name__ == "__main__":
    extract_passing_data()
