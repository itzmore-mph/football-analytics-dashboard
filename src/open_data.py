# src/open_data.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from tqdm import tqdm

from .config import settings
from .utils_io import read_remote_json

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


# ──────────────────────────────────────────────────────────────────────────────
# Low-level fetchers
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def competitions() -> pd.DataFrame:
    """
    Return the StatsBomb competitions catalog.

    Columns (typical):
      competition_id, competition_name, season_id, season_name, country_name, ...
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
    out = df[df["competition_id"] == comp_id][cols].copy()
    return out.reset_index(drop=True)


def _extract_team_name(x) -> str | None:
    """
    Defensive extractor for rare cases where a team might still be a dict.
    """
    if isinstance(x, dict):
        return x.get("home_team_name") or x.get("away_team_name") or x.get("name")
    if pd.isna(x):
        return None
    return str(x)


@lru_cache(maxsize=256)
def _matches_cached(comp_id: int, season_id: int) -> pd.DataFrame:
    """
    Cached raw match list for a (competition_id, season_id) pair.

    Uses json_normalize to flatten nested objects so downstream code receives
    clean columns like:
      home_team.home_team_name, away_team.away_team_name, ...
    """
    data = read_remote_json(f"{RAW_BASE}/matches/{comp_id}/{season_id}.json")
    df = pd.json_normalize(data, sep=".")
    return df


def matches(comp_id: int, season_id: int) -> pd.DataFrame:
    """
    Return matches for a given competition and season.

    Adds convenience columns:
      match_id (int), home_team (string), away_team (string), match_date (datetime)

    Returns a DataFrame (possibly empty), never None.
    """
    df = _matches_cached(int(comp_id), int(season_id)).copy()

    if df.empty:
        return pd.DataFrame(columns=["match_id", "home_team", "away_team", "match_date"])

    # Convenience columns (prefer normalized fields)
    if "home_team.home_team_name" in df.columns:
        df["home_team"] = df["home_team.home_team_name"]
    elif "home_team" in df.columns:
        df["home_team"] = df["home_team"].map(_extract_team_name)

    if "away_team.away_team_name" in df.columns:
        df["away_team"] = df["away_team.away_team_name"]
    elif "away_team" in df.columns:
        df["away_team"] = df["away_team"].map(_extract_team_name)

    # Ensure essentials exist
    expected = ["match_id", "home_team", "away_team", "match_date"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    # Dtypes
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    df["home_team"] = df["home_team"].astype("string")
    df["away_team"] = df["away_team"].astype("string")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Keep a clean, stable set for selectors and metadata
    out = df[expected].copy()
    out = out.dropna(subset=["match_id"])
    out["match_id"] = out["match_id"].astype(int)

    return out.reset_index(drop=True)


@lru_cache(maxsize=2048)
def events(match_id: int) -> pd.DataFrame:
    """
    Return a normalized events DataFrame for the given match_id.
    Cached by match_id to prevent repeated network calls on reruns.
    """
    data = read_remote_json(f"{RAW_BASE}/events/{int(match_id)}.json")
    return pd.json_normalize(data, sep=".")


# ──────────────────────────────────────────────────────────────────────────────
# Metadata lookup (avoid building a full global index unless you truly need it)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def match_meta_from_pair(comp_id: int, season_id: int, match_id: int) -> dict:
    """
    Fetch metadata for a match when you already know comp_id and season_id.
    This is much cheaper than indexing all competitions.

    Returns {} if not found.
    """
    ms = matches(int(comp_id), int(season_id))
    row = ms[ms["match_id"] == int(match_id)]
    if row.empty:
        return {}
    out = row.iloc[0].to_dict()
    out["competition_id"] = int(comp_id)
    out["season_id"] = int(season_id)
    return out


@lru_cache(maxsize=1)
def _match_index() -> pd.DataFrame:
    """
    Optional: Build a cached index of all matches with competition and season names.
    This is network-heavy the first time. Use only if your app truly needs it.

    Columns:
      match_id, competition_id, season_id, competition_name, season_name,
      home_team, away_team, match_date
    """
    dfc = competitions()
    rows: list[pd.DataFrame] = []

    for _, r in tqdm(dfc.iterrows(), total=len(dfc)):
        try:
            mdf = matches(int(r.competition_id), int(r.season_id))
        except Exception:
            continue

        if mdf.empty:
            continue

        tmp = mdf.copy()
        tmp["competition_id"] = int(r.competition_id)
        tmp["season_id"] = int(r.season_id)
        tmp["competition_name"] = r.get("competition_name")
        tmp["season_name"] = r.get("season_name")
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

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["match_id"])
    return out


def match_meta(match_id: int) -> dict:
    """
    Return metadata for a match_id from the global cached index.

    Returns {} if not found.

    Note: This will trigger a full index build the first time it is called.
    Prefer match_meta_from_pair() when you know competition_id and season_id.
    """
    idx = _match_index()
    row = idx[idx["match_id"] == int(match_id)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Demo sampling (Bundesliga fixed, deterministic)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DemoPlan:
    """
    Fixed demo plan: German Bundesliga, season 2023/2024.

    These IDs are from StatsBomb Open Data releases:
      competition_id=9, season_id=281
    """
    competition_id: int = 9
    season_id: int = 281
    matches_per_season: int = 2


def collect_demo_matches(plan: DemoPlan | None = None) -> list[int]:
    """
    Deterministic demo sampler used by your 'Build demo data' button.

    Picks the most recent matches by match_date from the Bundesliga season,
    then applies settings.demo_matches as an optional cap.

    Returns list[int], possibly empty.
    """
    if plan is None:
        plan = DemoPlan()

    ms = matches(plan.competition_id, plan.season_id)
    if ms.empty:
        return []

    # newest first
    ms = ms.sort_values("match_date", ascending=False, na_position="last")

    ids = (
        ms["match_id"]
        .dropna()
        .astype(int)
        .head(plan.matches_per_season)
        .tolist()
    )

    cap = getattr(settings, "demo_matches", None)
    if cap is not None:
        return ids[: int(cap)]
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# Scalable collectors
# ──────────────────────────────────────────────────────────────────────────────

def pick_competitions_with_matches(
    n: int = 3,
    max_probes: int = 25,
) -> list[tuple[int, int]]:
    """
    Returns [(competition_id, season_id), ...] for the first n that have matches.
    Probes up to max_probes unique competitions from competitions().

    Useful if you want a dynamic demo alternative.
    """
    df = competitions().copy()

    if "season_id" in df.columns:
        df = df.sort_values("season_id", ascending=False)

    picked: list[tuple[int, int]] = []
    seen_comp: set[int] = set()

    probes = 0
    for _, row in df.iterrows():
        if probes >= max_probes or len(picked) >= n:
            break

        comp_id = int(row["competition_id"])
        season_id = int(row["season_id"])

        if comp_id in seen_comp:
            continue
        seen_comp.add(comp_id)

        probes += 1

        try:
            ms = matches(comp_id, season_id)
        except Exception:
            continue

        if not ms.empty:
            picked.append((comp_id, season_id))

    return picked


def collect_full_matches(limit: int | None = None) -> list[int]:
    """
    Collect up to `limit` match_ids across all competition and season pairs.

    This can be large and network-heavy.
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

    if season_name:
        cdf = cdf[cdf["season_name"] == season_name].copy()
        if cdf.empty:
            raise ValueError(
                f"Season not found for {competition_name!r}: {season_name!r}"
            )
    else:
        if "season_id" in cdf.columns and cdf["season_id"].notna().any():
            cdf = cdf.sort_values("season_id", ascending=False)
        else:
            cdf = cdf.sort_values("season_name", ascending=False)

    row = cdf.iloc[0]
    comp_id = int(row["competition_id"])
    season_id = int(row["season_id"])

    ms = matches(comp_id, season_id)
    if ms.empty:
        raise RuntimeError(
            f"No matches available for competition_id={comp_id}, season_id={season_id}"
        )

    if sort_by_date:
        ms = ms.sort_values("match_date", ascending=ascending, na_position="last")

    ids = ms["match_id"].dropna().astype(int).tolist()
    if not ids:
        raise RuntimeError(
            f"No match_ids found for competition_id={comp_id}, season_id={season_id}"
        )

    return ids[:limit] if limit else ids
