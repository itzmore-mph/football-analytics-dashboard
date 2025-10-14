from __future__ import annotations
import pandas as pd
from typing import Iterable

from .config import settings
from .open_data import events
from .utils_io import save_csv
from .features_xg import build_basic_features, FEATURE_COLUMNS, TARGET_COLUMN


def _extract_shots(ev: pd.DataFrame) -> pd.DataFrame:
    shots = ev[ev["type.name"] == "Shot"].copy()
    # outcome flag
    shots[TARGET_COLUMN] = (
        shots["shot.outcome.name"]
        .fillna("")
        .str.lower() == "goal"
    ).astype(int)
    # normalize locations
    loc = shots["location"].apply(
        lambda v: v if isinstance(v, list) else [None, None]
    )
    shots["location.x"] = loc.apply(
        lambda location: float(location[0])
        if location and location[0] is not None
        else None
    )
    shots["location.y"] = loc.apply(
        lambda location: float(location[1])
        if location and location[1] is not None
        else None
    )
    shots = shots.dropna(subset=[
        "location.x",
        "location.y"]).reset_index(drop=True)
    # keep useful columns
    keep = [
        "match_id",
        "team.name",
        "player.name",
        "minute",
        "body_part.name",
        "play_pattern.name",
        "under_pressure",
        "shot.outcome.name",
        "shot.type.name",
        "location.x",
        "location.y",
        ]
    shots = shots[keep]
    shots = build_basic_features(shots)
    return shots


def build_processed_shots(match_ids: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mid in match_ids:
        ev = events(mid)
        ev["match_id"] = mid
        frames.append(_extract_shots(ev))
    out = pd.concat(frames, ignore_index=True)
    save_csv(out, settings.processed_shots_csv)
    return out
