from __future__ import annotations

import math
import re
from typing import Final

import numpy as np
import pandas as pd

# Goal location on a 120x80 pitch (center at y=40)
GOAL_X: Final[float] = 120.0
GOAL_Y: Final[float] = 40.0
GOAL_Y_TOP: Final[float] = 44.0  # posts, used for the angle geometry
GOAL_Y_BOTTOM: Final[float] = 36.0

SET_PIECE_REGEX: Final[re.Pattern[str]] = re.compile(
    r"free\s?kick|corner|throw[- ]in|kick[- ]off|goal kick|penalty",
    flags=re.IGNORECASE,
)


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

    coords = df[["location.x", "location.y"]].astype(float)
    x = coords["location.x"].to_numpy()
    y = coords["location.y"].to_numpy()

    # Geometry-based features
    df["shot_distance"] = np.hypot(GOAL_X - x, GOAL_Y - y)
    xg = np.clip(GOAL_X - x, 1e-6, None)
    y_bottom = GOAL_Y_BOTTOM - y
    y_top = GOAL_Y_TOP - y
    angles = np.abs(np.arctan2(y_top, xg) - np.arctan2(y_bottom, xg))
    df["shot_angle"] = np.clip(angles, 0.0, math.pi)

    # Body-part flags
    body = df["body_part.name"].fillna("").str.casefold()
    df["is_header"] = (body == "head").astype("int64")
    df["is_left_foot"] = body.str.contains("left", na=False).astype("int64")
    df["is_right_foot"] = body.str.contains("right", na=False).astype("int64")

    # Context features
    under_pressure = df["under_pressure"].astype("boolean")
    df["under_pressure"] = under_pressure.fillna(False).astype("int64")
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce").fillna(0).round().astype("int64")

    # Set-piece indicator (robust to variations in wording/case)
    df["is_set_piece"] = (
        df["play_pattern.name"].fillna("").apply(SET_PIECE_REGEX.search).notnull()
    ).astype("int64")

    # Target variable: 1 if goal, else 0
    outcome = df.get("shot.outcome.name")
    if outcome is None:
        outcome = pd.Series("", index=df.index, dtype="object")
    df["is_goal"] = outcome.fillna("").astype(str).str.casefold().eq("goal").astype("int64")

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
