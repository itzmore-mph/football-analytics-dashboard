# src/open_data.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from tqdm import tqdm

from .config import settings  # noqa: F401  # kept for consistency/forward use
from .utils_io import read_remote_json

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


# ──────────────────────────────────────────────────────────────────────────────
# Low-level fetchers
# ──────────────────────────────────────────────────────────────────────────────

def competitions() -> pd.DataFrame:
    """
    Return the StatsBomb competitions catalog.
    Columns (typical): competition_id, competition_name, season_id, season_name
    """
    data = read_remote_json(f"{RAW_BASE}/competitions.json")
    return pd.DataFrame(data)


def seasons_for(comp_id: int) -> pd.DataFrame:
    """
    Return seasons for a given competition id.
    Columns: competition_id, season_id, season_name
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

    # Flatten nested dicts into columns like home_team.home_team_name
    df = pd.json_normalize(data, sep=".")

    # Convenience string columns
    if "home_team.home_team_name" in df.columns:
        df["home_team"] = df["home_team.home_team_name"]
    elif "home_team" in df.columns:
        # fallback if the API already gives a string
        df["home_team"] = df["home_team"].astype("string")

    if "away_team.away_team_name" in df.columns:
        df["away_team"] = df["away_team.away_team_name"]
    elif "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype("string")

    # Keep common fields for downstream selectors
    expected = ["match_id", "home_team", "away_team", "match_date"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    # Enforce nice dtypes
    df["home_team"] = df["home_team"].astype("string")
    df["away_team"] = df["away_team"].astype("string")
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    return df

def _extract_team_name(x):
    if isinstance(x, dict):
        # StatsBomb matches often store home_team_name/away_team_name inside the dict
        return x.get("home_team_name") or x.get("away_team_name") or x.get("name")
    return x

# after df is created (either via DataFrame or json_normalize)
if "home_team" in df.columns:
    df["home_team"] = df["home_team"].map(_extract_team_name)
if "away_team" in df.columns:
    df["away_team"] = df["away_team"].map(_extract_team_name)


def events(match_id: int) -> pd.DataFrame:
    """
    Return a normalized events DataFrame for the given match_id.
    """
    data = read_remote_json(f"{RAW_BASE}/events/{match_id}.json")
    return pd.json_normalize(data, sep=".")


# ──────────────────────────────────────────────────────────────────────────────
# Cached match index + metadata lookup
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _match_index() -> pd.DataFrame:
    """
    Build a cached index of all matches with their competition/season names.
    Columns:
      match_id, competition_id, season_id, competition_name, season_name,
      home_team, away_team, match_date
    """
    dfc = competitions()
    rows: list[pd.DataFrame] = []

    for _, r in dfc.iterrows():
        try:
            mdf = matches(int(r.competition_id), int(r.season_id))
        except Exception:
            # Skip pairs without available matches or transient fetch errors
            continue

        if mdf is None or mdf.empty:
            continue

        # Ensure essential columns exist
        keep = ["match_id", "home_team", "away_team", "match_date"]
        for c in keep:
            if c not in mdf.columns:
                mdf[c] = pd.NA

        tmp = mdf[keep].copy()
        tmp["competition_id"] = int(r.competition_id)
        tmp["season_id"] = int(r.season_id)
        tmp["competition_name"] = r.competition_name
        tmp["season_name"] = r.season_name
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "home_team",
                "away_team",
                "match_date",
                "competition_id",
                "season_id",
                "competition_name",
                "season_name",
            ]
        )

    # Concatenate and de-duplicate just in case
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["match_id"])
    return out


def match_meta(match_id: int) -> dict:
    """
    Return metadata for a match_id:

    {
      'competition_name', 'season_name', 'competition_id', 'season_id',
      'home_team', 'away_team', 'match_date', 'match_id'
    }

    Returns {} if not found.
    """
    idx = _match_index()
    row = idx[idx["match_id"] == match_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Demo sampling (small, fast)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Scalable collectors
# ──────────────────────────────────────────────────────────────────────────────

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
    Collect match_ids for a given competition name.
    If season_name is None, pick the latest season for that competition.
    Always returns a list[int], never None.
    """
    dfc = competitions()
    cdf = dfc[dfc["competition_name"] == competition_name].copy()
    if cdf.empty:
        raise ValueError(f"Competition not found: {competition_name!r}")

    # Choose season
    if season_name:
        cdf = cdf[cdf["season_name"] == season_name].copy()
        if cdf.empty:
            raise ValueError(
                f"Season not found for {competition_name!r}: {season_name!r}"
            )
    else:
        # "Latest" is safest by season_id if numeric and present, fallback to season_name sort
        if "season_id" in cdf.columns and cdf["season_id"].notna().any():
            cdf = cdf.sort_values("season_id", ascending=False)
        else:
            cdf = cdf.sort_values("season_name", ascending=False)

    row = cdf.iloc[0]
    comp_id = int(row["competition_id"])
    season_id = int(row["season_id"])

    ms = matches(comp_id, season_id)

    if ms is None or ms.empty or "match_id" not in ms.columns:
        raise RuntimeError(
            f"No matches available for competition_id={comp_id}, season_id={season_id}"
        )

    # Optional: sort by match_date if available
    if sort_by_date and "match_date" in ms.columns:
        ms = ms.copy()
        ms["match_date"] = pd.to_datetime(ms["match_date"], errors="coerce")
        ms = ms.sort_values("match_date", ascending=ascending)

    ids = ms["match_id"].dropna().astype(int).tolist()
    if not ids:
        raise RuntimeError(
            f"No match_ids found for competition_id={comp_id}, season_id={season_id}"
        )

    return ids[:limit] if limit else ids
