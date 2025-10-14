from __future__ import annotations
import math
import numpy as np
import pandas as pd


GOAL_X_LEFT, GOAL_X_RIGHT = 120.0, 120.0
GOAL_Y_TOP, GOAL_Y_BOTTOM = 44.0, 36.0  # goal posts (center at y=40) on 120x80


def shot_distance(x: float, y: float) -> float:
    # distance to goal center (120,40)
    return float(math.hypot(120.0 - x, 40.0 - y))


def shot_angle(x: float, y: float) -> float:
    # angle subtended by goal posts from shot location
    # Based on open-source geometry; clamp for numeric stability
    xg = max(1e-6, 120.0 - x)
    y1, y2 = 36.0 - y, 44.0 - y
    num = abs(math.atan2(y2, xg) - math.atan2(y1, xg))
    return float(max(0.0, min(num, math.pi)))


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["shot_distance"] = df.apply(
        lambda r: shot_distance(r["location.x"], r["location.y"]), axis=1
    )
    df["shot_angle"] = df.apply(
        lambda r: shot_angle(r["location.x"], r["location.y"]), axis=1
    )
    # simple flags
    df["is_header"] = (
        df["body_part.name"].fillna("").str.lower() == "head"
    ).astype(int)
    df["is_left_foot"] = (
        df["body_part.name"]
        .fillna("")
        .str.lower()
        .str.contains("left")
    ).astype(int)
    df["is_right_foot"] = (
        df["body_part.name"]
        .fillna("")
        .str.lower()
        .str.contains("right")
    ).astype(int)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["minute"] = df["minute"].fillna(0).astype(int)
    # play pattern (open play vs set piece)
    df["is_set_piece"] = df["play_pattern.name"].fillna("").str.contains(
        "Free Kick|Corner|Throw-in|From Kick-off|Penalty|Goal Kick", regex=True
    ).astype(int)
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
TARGET_COLUMN = "shot.outcome.name_goal"
