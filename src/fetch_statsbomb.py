import requests
import os
import json

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
MATCH_DATA_PATH = os.path.join(DATA_DIR, "15946.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_match_data():
    url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/15946.json"
    response = requests.get(url)

    if response.status_code == 200:
        with open(MATCH_DATA_PATH, "w") as f:
            json.dump(response.json(), f, indent=4)
        print(f"Match data saved to {MATCH_DATA_PATH}")
    else:
        print("Failed to fetch match data.")

if __name__ == "__main__":
    fetch_match_data()
