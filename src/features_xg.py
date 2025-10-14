from __future__ import annotations

import math
import pandas as pd

# Goal location on a 120x80 pitch (center at y=40)
GOAL_X, GOAL_Y = 120.0, 40.0
GOAL_Y_TOP, GOAL_Y_BOTTOM = 44.0, 36.0  # posts, used for the angle geometry


def shot_distance(x: float, y: float) -> float:
    """Euclidean distance from (x, y) to the goal center (120, 40)."""
    return float(math.hypot(GOAL_X - x, GOAL_Y - y))


def shot_angle(x: float, y: float) -> float:
    """
    Angle subtended by the goal posts at the shot location.
    Uses a numerically stable formulation and clamps to [0, pi].
    """
    xg = max(1e-6, GOAL_X - x)  # avoid division by zero when on goal line
    y1, y2 = GOAL_Y_BOTTOM - y, GOAL_Y_TOP - y
    num = abs(math.atan2(y2, xg) - math.atan2(y1, xg))
    return float(max(0.0, min(num, math.pi)))


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build basic xG features from a shots DataFrame that already contains:
      - location.x, location.y
      - minute, under_pressure
      - body_part.name, play_pattern.name, shot.outcome.name
    Returns a new DataFrame with engineered numeric features.
    """
    df = df.copy()

    # Geometry-based features
    df["shot_distance"] = df.apply(
        lambda r: shot_distance(r["location.x"], r["location.y"]), axis=1
    )
    df["shot_angle"] = df.apply(
        lambda r: shot_angle(r["location.x"], r["location.y"]), axis=1
    )

    # Body-part flags
    body = df["body_part.name"].fillna("").str.lower()
    df["is_header"] = (body == "head").astype(int)
    df["is_left_foot"] = body.str.contains("left").astype(int)
    df["is_right_foot"] = body.str.contains("right").astype(int)

    # Context features
    df["under_pressure"] = (
        df["under_pressure"]
        .fillna(False)
        .infer_objects(copy=False)
        .astype("int64")
    )
    df["minute"] = df["minute"].fillna(0).astype(int)

    # Set-piece indicator (robust to variations in wording/case)
    df["is_set_piece"] = (
        df["play_pattern.name"]
        .fillna("")
        .str.lower()
        .str.contains(r"free kick|corner|throw|kick[- ]off|goal kick|penalty")
        .astype("int64")
    )

    # Target variable: 1 if goal, else 0
    df["is_goal"] = (
        df["shot.outcome.name"]
        .fillna("")
        .eq("Goal")
        .astype("int64")
    )

    return df


FEATURE_COLUMNS = [
    "shot_distance",
    "shot_angle",
    "is_header",
    "is_left_foot",
    "is_right_foot",
    "under_pressure",
    "minute",
    "is_set_piece",
]

# Training target column
TARGET_COLUMN = "is_goal"
