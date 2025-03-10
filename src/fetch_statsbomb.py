import os
import requests
import pandas as pd

def fetch_statsbomb_xg():
    MATCH_ID = 7561  # Example match ID
    URL = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{MATCH_ID}.json"

    # Define absolute path
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(PROJECT_ROOT, "../data/shots_data.csv")

    # Fetch data
    df = pd.read_json(URL)
    shots = df[df["type"].apply(lambda x: x.get("name", "")) == "Shot"]


    shot_data = []
    for _, row in shots.iterrows():
        shot_data.append({
            "player": row["player"]["name"],
            "team": row["team"]["name"],
            "minute": row["minute"],
            "shot_distance": row["location"][0],
            "shot_angle": row["location"][1],
            "body_part": row["shot"].get("bodyPart", {}).get("name", "Unknown"),
            "shot_type": row["shot"]["technique"]["name"],
            "goal_scored": 1 if row["shot"]["outcome"]["name"] == "Goal" else 0,
            "xG": row["shot"].get("xG", 0)  # Default to 0 if missing
        })

    # Save data
    df_shots = pd.DataFrame(shot_data)
    df_shots.to_csv(DATA_PATH, index=False)
    print("Shot data saved!")

if __name__ == "__main__":
    fetch_statsbomb_xg()
