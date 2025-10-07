from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def list_competitions() -> pd.DataFrame:
    with open(DATA / "competitions.json", "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

def list_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    p = DATA / f"matches_{competition_id}_{season_id}.json"
    with p.open("r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

def ensure_open_data_files():
    """Ensure the data directory exists."""
    DATA.mkdir(parents=True, exist_ok=True)
