import os
import requests
import json

# Define the output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

# URL for a sample StatsBomb open data match
MATCH_ID = "15946"  # Replace with another ID if needed
URL = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{MATCH_ID}.json"

def fetch_match_data():
    """Fetch match data from StatsBomb and save it locally."""
    response = requests.get(URL)
    if response.status_code == 200:
        file_path = os.path.join(DATA_DIR, f"{MATCH_ID}.json")
        with open(file_path, "w") as file:
            json.dump(response.json(), file, indent=4)
        print(f"Match data saved at: {file_path}")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

if __name__ == "__main__":
    fetch_match_data()
