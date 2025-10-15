# src/open_data.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from .config import settings
from .utils_io import read_remote_json

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


# ──────────────────────────────────────────────────────────────────────────────
# Low-level fetchers
# ──────────────────────────────────────────────────────────────────────────────


def competitions() -> pd.DataFrame:
    """
    Return the StatsBomb competitions catalog.
    Columns: competition_id, competition_name, season_id, season_name
    """
    data = read_remote_json(f"{RAW_BASE}/competitions.json")
    return pd.DataFrame(data)


def seasons_for(comp_id: int) -> pd.DataFrame:
    """
    Return seasons for a given competition id.
    """
    df = competitions()
    cols = ["competition_id", "season_id", "season_name"]
    return df[df["competition_id"] == comp_id][cols].copy()


def matches(comp_id: int, season_id: int) -> pd.DataFrame:
    """
    Return matches for a given competition/season.
    Adds convenience columns: home_team, away_team, match_date (if present).
    """
    data = read_remote_json(f"{RAW_BASE}/matches/{comp_id}/{season_id}.json")
    df = pd.DataFrame(data)

    # Normalize a few convenient columns if available
    rename_map = {
        "home_team.home_team_name": "home_team",
        "away_team.away_team_name": "away_team",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # Keep common fields present for downstream selectors
    expected = ["match_id", "home_team", "away_team", "match_date"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def events(match_id: int) -> pd.DataFrame:
    """
    Return a normalized events DataFrame for the given match_id.
    """
    data = read_remote_json(f"{RAW_BASE}/events/{match_id}.json")
    return pd.json_normalize(data, sep=".")


# Demo sampling (small, fast)


@dataclass(frozen=True)
class DemoPlan:
    competitions: tuple[int, ...] = (43, 49, 72)
    seasons_per_comp: int = 1
    matches_per_season: int = 2


def collect_demo_matches(plan: DemoPlan | None = None) -> list[int]:
    """
    Small, deterministic demo sampler used by your 'Build demo data' button.
    Respects settings.demo_matches as a final cap.
    """
    if plan is None:
        plan = DemoPlan()

    m_ids: list[int] = []
    for comp in plan.competitions:
        comp_df = seasons_for(comp)
        # Take up to N seasons per competition
        for _, row in comp_df.head(plan.seasons_per_comp).iterrows():
            ms = matches(int(row.competition_id), int(row.season_id)).head(
                plan.matches_per_season
            )
            m_ids.extend(ms["match_id"].astype(int).tolist())

    cap = getattr(settings, "demo_matches", None)
    if cap is not None:
        return m_ids[: int(cap)]
    return m_ids


# Scalable collectors


def collect_full_matches(limit: int | None = None) -> list[int]:
    """
    Collect up to `limit` match_ids across all competitions/seasons listed
    by StatsBomb.
    Order is the order provided by the competitions catalog (then matches).
    Useful for your 'Build more data' button.

    NOTE: This is a broad sweep.
    For more control, use collect_matches_by_name().
    """
    dfc = competitions()
    m_ids: list[int] = []

    for _, r in tqdm(dfc.iterrows(), total=len(dfc)):
        try:
            ms = matches(int(r.competition_id), int(r.season_id))
            ids = ms["match_id"].dropna().astype(int).tolist()
            m_ids.extend(ids)
            if limit and len(m_ids) >= limit:
                break
        except Exception:
            # Skip competition/season pairs that may not have matches available
            continue

    return m_ids if not limit else m_ids[:limit]


def collect_matches_by_name(
    competition_name: str,
    season_name: str | None = None,
    limit: int | None = 20,
    sort_by_date: bool = True,
    ascending: bool = True,
) -> list[int]:
    """
    Collect match_ids for a given competition name (e.g. "FIFA World Cup").
    If season_name is None, picks the latest season for that competition.
    """
    dfc = competitions()
    cdf = dfc[dfc["competition_name"] == competition_name].copy()
    if cdf.empty:
        raise ValueError(f"Competition not found: {competition_name!r}")

    if season_name:
        cdf = cdf[cdf["season_name"] == season_name]
        if cdf.empty:
            raise ValueError(
                f"Season {season_name!r} not found for competition "
                f"{competition_name!r}"
            )

    # Choose the latest season if not specified
    cdf = cdf.sort_values("season_name", ascending=False)
    comp_id = int(cdf.iloc[0]["competition_id"])
    season_id = int(cdf.iloc[0]["season_id"])

    mdf = matches(comp_id, season_id)
    if sort_by_date and "match_date" in mdf.columns:
        mdf = mdf.sort_values("match_date", ascending=ascending)

    mids = mdf["match_id"].dropna().astype(int).tolist()
    return mids if limit is None else mids[:limit]
