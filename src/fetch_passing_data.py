import json
import pandas as pd
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_JSON_PATH = os.path.join(PROJECT_ROOT, "../data/15946.json")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "../data/processed_passing_data.csv")

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
                "start_x": event["location"][0],
                "start_y": event["location"][1],
                "end_x": event["pass"]["end_location"][0],
                "end_y": event["pass"]["end_location"][1],
                "pass_length": event["pass"].get("length", None),
                "pass_angle": event["pass"].get("angle", None),
                "pass_outcome": event["pass"].get("outcome", {}).get("name", "Complete")
            }
            passing_data.append(pass_event)

    df = pd.DataFrame(passing_data)
    df["x"] = (df["start_x"] + df["end_x"]) / 2
    df["y"] = (df["start_y"] + df["end_y"]) / 2

    grouped = df.groupby(["passer", "receiver"]).agg({
        "x": "mean",
        "y": "mean",
        "pass_length": "mean",
        "pass_angle": "mean",
        "pass_outcome": lambda x: (x == "Complete").sum(),
        "team": "first",
        "minute": "count"
    }).rename(columns={"minute": "pass_count"}).reset_index()

    grouped.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Passing data extracted and saved to processed_passing_data.csv")

if __name__ == "__main__":
    extract_passing_data()
