import json
import math
from pathlib import Path
import pandas as pd

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]  # repo root (../ from src/)
DATA_DIR = ROOT / "data"
MATCH_DATA_PATH = DATA_DIR / "15946.json"   # default example
SHOTS_DATA_PATH = DATA_DIR / "shots_data.csv"

GOAL_X, GOAL_Y = 120.0, 40.0     # StatsBomb coordinates (120 x 80)
POST_LOW_Y, POST_HIGH_Y = 36.0, 44.0


def _safe_name(obj, key_chain, default=None):
    cur = obj
    for k in key_chain:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_int_bool(v) -> int:
    return int(bool(v))


def _shot_geometry(loc_x: float, loc_y: float) -> tuple[float, float]:
    dx_center = GOAL_X - loc_x
    dy_center = GOAL_Y - loc_y
    dist = math.hypot(dx_center, dy_center)
    a_high = math.atan2(POST_HIGH_Y - loc_y, GOAL_X - loc_x)
    a_low = math.atan2(POST_LOW_Y - loc_y, GOAL_X - loc_x)
    angle = abs(a_high - a_low)  # radians
    return dist, angle


def extract_shot_data(match_id: int = 15946):
    events_path = DATA_DIR / f"{match_id}.json"
    if not events_path.exists():
        print(f"Error: {events_path} not found. Run fetch_statsbomb.py first.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with events_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    shot_rows = []
    for e in events:
        if _safe_name(e, ("type", "name")) != "Shot":
            continue

        loc = e.get("location")
        if not (isinstance(loc, list) and len(loc) >= 2):
            continue

        x, y = float(loc[0]), float(loc[1])
        shot_distance, shot_angle = _shot_geometry(x, y)

        outcome = _safe_name(e, ("shot", "outcome", "name"), default="Unknown")
        body_part = _safe_name(
            e, ("shot", "body_part", "name"), default="Unknown"
        )
        technique = _safe_name(
            e, ("shot", "technique", "name"), default="Unknown"
        )
        shot_type = _safe_name(e, ("shot", "type", "name"), default="Unknown")

        row = {
            "match_id": _safe_name(e, ("match_id",), default=match_id),
            "player": _safe_name(e, ("player", "name"), default="Unknown"),
            "team": _safe_name(e, ("team", "name"), default="Unknown"),
            "x": x,
            "y": y,
            "shot_distance": shot_distance,
            "shot_angle": shot_angle,
            "goal_scored": int(outcome == "Goal"),
            "body_part": body_part,
            "technique": technique,
            "shot_type": shot_type,
            "is_penalty": int(shot_type == "Penalty"),
            "is_header": int(body_part == "Head"),
            "under_pressure": _to_int_bool(e.get("under_pressure", False)),
            "minute": e.get("minute"),
            "second": e.get("second"),
        }
        shot_rows.append(row)

    if not shot_rows:
        print("No shots parsed; CSV not written.")
        return

    df = pd.DataFrame(shot_rows)
    df = df.query("0 <= x <= 120 and 0 <= y <= 80").copy()

    SHOTS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SHOTS_DATA_PATH, index=False)
    print(f"Wrote {len(df)} shots: {SHOTS_DATA_PATH}")


if __name__ == "__main__":
    extract_shot_data()
