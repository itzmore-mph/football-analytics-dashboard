# src/preprocess_xG.py

# Preprocess raw shot data for xG modeling

from pathlib import Path
import pandas as pd

from .validation import validate_shots

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IN_CSV = DATA / "shots_data.csv"
OUT_CSV = DATA / "processed_shots.csv"


def main():
    if not IN_CSV.exists():
        print(f"Error: {IN_CSV} not found. Run fetch_shots_data.py first.")
        return
    raw = pd.read_csv(IN_CSV)
    validated = validate_shots(raw)
    for issue in validated.issues:
        print(f"Warning: {issue}")
    df = validated.frame

    keep_cols = [
        "shot_distance",
        "shot_angle",
        "goal_scored",
        "x",
        "y",
        "team",
        "player",
        "xG",
        "minute",
        "second",
        "under_pressure",
        "is_penalty",
        "is_header",
        "body_part",
        "technique",
        "shot_type",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].copy()

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote processed shots: {OUT_CSV}")


if __name__ == "__main__":
    main()
