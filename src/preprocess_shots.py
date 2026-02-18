from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import settings
from .features_xg import build_basic_features
from .open_data import events
from .utils_io import save_csv


def _first_nonempty(series_like: Sequence[Any]) -> Any:
    """
    Return the first non-null/nonnull value across a list-like of candidates;
    else None.
    """
    for v in series_like:
        if isinstance(v, pd.Series):
            v = v.dropna()
            if not v.empty:
                return v.iloc[0]
        elif v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None


def _extract_comp_season(ev: pd.DataFrame) -> tuple[str, str]:
    """
    Try multiple plausible column names for competition & season.
    StatsBomb open-data usually has them on the matches endpoint,
    but some pipelines propagate them to events. Be defensive.
    """
    comp_candidates = [
        ev.get("competition_name"),
        ev.get("competition.competition_name"),
        ev.get("competition.name"),
    ]
    season_candidates = [
        ev.get("season_name"),
        ev.get("season.season_name"),
        ev.get("season.name"),
        ev.get("season"),
    ]
    comp = _first_nonempty(comp_candidates)
    season = _first_nonempty(season_candidates)
    return (
        str(comp) if comp is not None else "",
        str(season) if season is not None else "",
    )


def _extract_shots(ev: pd.DataFrame) -> pd.DataFrame:
    shots = ev[ev["type.name"] == "Shot"].copy()

    # ---- Fixture metadata (persist for friendly UI labels) -----------------
    team_series = ev.get("team.name")
    team_list = (
        team_series.dropna().unique().tolist()
        if isinstance(team_series, pd.Series)
        else []
    )
    t1, t2 = (team_list + ["Unknown", "Unknown"])[:2]

    home_team = t1
    away_team = t2
    if (
        "home_team.home_team_name" in ev.columns
        and ev["home_team.home_team_name"].notna().any()
    ):
        home_team = str(ev["home_team.home_team_name"].dropna().iloc[0])
    if (
        "away_team.away_team_name" in ev.columns
        and ev["away_team.away_team_name"].notna().any()
    ):
        away_team = str(ev["away_team.away_team_name"].dropna().iloc[0])

    match_date = (
        pd.to_datetime(ev["match_date"].dropna().iloc[0])
        if "match_date" in ev.columns and ev["match_date"].notna().any()
        else pd.NaT
    )

    # ---- Competition / Season (for sidebar filters) ------------------------
    competition_name, season_name = _extract_comp_season(ev)

    # Fallback to authoritative match metadata if events lack comp/season
    # or if home/away/date are missing/Unknown.
    if (not competition_name) or (not season_name) or pd.isna(match_date) \
       or home_team == "Unknown" or away_team == "Unknown":
        try:
            # Lazy import avoids circular imports at module load time
            from .open_data import match_meta
            mid = (
                int(ev["match_id"].dropna().iloc[0])
                if "match_id" in ev.columns and ev["match_id"].notna().any()
                else None
            )
            if mid is not None:
                meta = match_meta(mid)
                competition_name = competition_name or str(
                    meta.get("competition_name", "") or ""
                )
                season_name = season_name or str(
                    meta.get("season_name", "") or ""
                )
                home_team = str(meta.get("home_team", home_team) or home_team)
                away_team = str(meta.get("away_team", away_team) or away_team)
                if meta.get("match_date"):
                    match_date = pd.to_datetime(meta["match_date"])
        except Exception:
            pass

    # Build fixture label after any overrides
    fixture_label = f"{home_team} vs {away_team}"

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
    shots = shots.reindex(columns=keep)

    # Basic sanity: drop rows with completely missing coordinates
    shots = shots.dropna(subset=["location.x", "location.y"], how="any")

    # Downstream feature engineering (adds distance, angle, etc.)
    shots = build_basic_features(shots)

    # Ensure columns used by the app exist (defensive)
    if "is_goal" not in shots.columns:
        shots["is_goal"] = (
            shots["shot.outcome.name"].astype(str) == "Goal"
        ).astype(int)
    if "shot_distance" not in shots.columns:
        # Fallback: Euclidean distance to center of goal in StatsBomb coords
        # (120x80; goal center at x=120, y=40)
        shots["shot_distance"] = np.sqrt(
            (120 - shots["location.x"]) ** 2 + (40 - shots["location.y"]) ** 2
        )

    # Attach fixture & meta (constant per match)
    shots["home_team"] = str(home_team)
    shots["away_team"] = str(away_team)
    shots["fixture"] = fixture_label
    shots["match_date"] = match_date

    shots["competition_name"] = competition_name  # new
    shots["season_name"] = season_name            # new

    # Stabilize dtypes to avoid mixed-type warnings later
    shots["minute"] = pd.to_numeric(
        shots["minute"], errors="coerce"
    ).astype("Int64")
    for col in [
        "team.name",
        "player.name",
        "body_part.name",
        "play_pattern.name",
        "shot.outcome.name",
        "shot.type.name",
    ]:
        if col in shots.columns:
            shots[col] = shots[col].astype("string")

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
        # required app columns (empty)
        empty["is_goal"] = pd.Series(dtype="Int64")
        empty["shot_distance"] = pd.Series(dtype="float")
        empty["home_team"] = pd.Series(dtype="string")
        empty["away_team"] = pd.Series(dtype="string")
        empty["fixture"] = pd.Series(dtype="string")
        empty["match_date"] = pd.Series(dtype="datetime64[ns]")
        empty["competition_name"] = pd.Series(dtype="string")
        empty["season_name"] = pd.Series(dtype="string")
        save_csv(empty, settings.processed_shots_csv)
        return empty

    out_new = pd.concat(frames, ignore_index=True)

    # Idempotent append: merge with existing CSV and drop duplicates
    csv_path: Path = settings.processed_shots_csv
    if csv_path.exists():
        try:
            out_old = pd.read_csv(csv_path, low_memory=False, dtype_backend="numpy_nullable")
            frames = []
            if out_old is not None and not out_old.empty:
                frames.append(out_old)
            if out_new is not None and not out_new.empty:
                frames.append(out_new)

            out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if "id" in out.columns:
                out = out.drop_duplicates(subset=["match_id", "id"])
            else:
                subset = [
                    c
                    for c in [
                        "match_id", "minute", "location.x",
                        "location.y", "player.name"
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
