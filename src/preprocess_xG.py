# Preprocess raw shot data for xG modeling

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN_CSV = DATA / "shots_data.csv"
OUT_CSV = DATA / "processed_shots.csv"


def main():
    if not IN_CSV.exists():
        print(f"Error: {IN_CSV} not found. Run fetch_shots_data.py first.")
        return
    df = pd.read_csv(IN_CSV)

    # Minimal cleaning / schema guarantee
    required = [
        "shot_distance", "shot_angle", "goal_scored",
        "x", "y", "team", "player"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: missing columns {missing} in {IN_CSV}")

    # Ensure dtypes
    for c in ["shot_distance", "shot_angle", "x", "y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "goal_scored" in df.columns:
        df["goal_scored"] = df["goal_scored"].astype(int)

    # Optional: drop clear invalids
    df = df.dropna(subset=["shot_distance", "shot_angle", "goal_scored"])

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote processed shots: {OUT_CSV}")


if __name__ == "__main__":
    main()
