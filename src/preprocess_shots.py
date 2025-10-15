from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import settings
from .features_xg import build_basic_features
from .open_data import events
from .utils_io import save_csv


def _extract_shots(ev: pd.DataFrame) -> pd.DataFrame:
    shots = ev[ev["type.name"] == "Shot"].copy()

    # Keep StatsBomb event id if present (for stable dedup later)
    if "id" not in shots.columns:
        shots["id"] = np.nan

    # Ensure optional columns exist (StatsBomb sometimes omits them)
    optional_defaults: dict[str, object] = {
        "body_part.name": "Unknown",
        "play_pattern.name": "Unknown",
        "shot.type.name": "Unknown",
        "shot.outcome.name": "Unknown",
        "under_pressure": False,
    }
    for col, default in optional_defaults.items():
        if col not in shots.columns:
            shots[col] = default
        else:
            shots[col] = shots[col].where(shots[col].notna(), default)

    # Robust location extraction -> location.x / location.y
    if "location" not in shots.columns:
        shots["location"] = None
    loc = shots["location"].apply(
        lambda v: (
            v if isinstance(v, list) and len(v) >= 2 else [np.nan, np.nan]
        )
    )
    shots["location.x"] = loc.apply(
        lambda xy: float(xy[0]) if xy and xy[0] is not None else np.nan
    )
    shots["location.y"] = loc.apply(
        lambda xy: float(xy[1]) if xy and xy[1] is not None else np.nan
    )

    # Keep a stable set of columns for feature builder
    keep = [
        "id",
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
    # Use reindex to avoid KeyErrors; missing columns will be filled
    shots = shots.reindex(columns=keep)

    # Basic sanity: drop rows with completely missing coordinates
    shots = shots.dropna(subset=["location.x", "location.y"], how="any")

    # Downstream feature engineering
    shots = build_basic_features(shots)
    return shots


def build_processed_shots(match_ids: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mid in match_ids:
        ev = events(mid)
        ev["match_id"] = mid
        frames.append(_extract_shots(ev))

    if not frames:
        base_columns = [
            "id",
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
        empty_raw = pd.DataFrame(columns=base_columns)
        empty = build_basic_features(empty_raw)
        empty["match_id"] = pd.Series(dtype="int64")
        save_csv(empty, settings.processed_shots_csv)
        return empty

    out_new = pd.concat(frames, ignore_index=True)

    # Idempotent append: merge with existing CSV and drop duplicates
    csv_path: Path = settings.processed_shots_csv
    if csv_path.exists():
        try:
            out_old = pd.read_csv(csv_path)
            out = pd.concat([out_old, out_new], ignore_index=True)
            if "id" in out.columns:
                out = out.drop_duplicates(subset=["match_id", "id"])
            else:
                subset = [
                    c
                    for c in [
                        "match_id",
                        "minute",
                        "location.x",
                        "location.y",
                        "player.name",
                    ]
                    if c in out.columns
                ]
                out = (
                    out.drop_duplicates(subset=subset)
                    if subset
                    else out.drop_duplicates()
                )
        except Exception:
            out = out_new
    else:
        out = out_new

    save_csv(out, csv_path)
    return out
