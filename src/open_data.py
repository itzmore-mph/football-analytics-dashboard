from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List


import pandas as pd
from tqdm import tqdm


from .config import settings
from .utils_io import read_remote_json


RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def competitions() -> pd.DataFrame:
    data = read_remote_json(f"{RAW_BASE}/competitions.json")
    df = pd.DataFrame(data)
    return df


def seasons_for(comp_id: int) -> pd.DataFrame:
    df = competitions()
    return df[df["competition_id"] == comp_id][[
        "competition_id",
        "season_id",
        "season_name"]
        ]


def matches(comp_id: int, season_id: int) -> pd.DataFrame:
    data = read_remote_json(f"{RAW_BASE}/matches/{comp_id}/{season_id}.json")
    return pd.DataFrame(data)


def events(match_id: int) -> pd.DataFrame:
    data = read_remote_json(f"{RAW_BASE}/events/{match_id}.json")
    return pd.json_normalize(data, sep=".")


@dataclass(frozen=True)
class DemoPlan:
    competitions: tuple[int, ...] = (43, 49, 72)
    seasons_per_comp: int = 1
    matches_per_season: int = 2


def collect_demo_matches(plan: DemoPlan = DemoPlan()) -> list[int]:
    m_ids: list[int] = []
    for comp in plan.competitions[:2]:
        comp_df = seasons_for(comp)
        for _, row in comp_df.head(plan.seasons_per_comp).iterrows():
            m = matches(
                int(row.competition_id), int(row.season_id)
            ).head(plan.matches_per_season)
            m_ids.extend(m["match_id"].astype(int).tolist())
    return m_ids[: settings.demo_matches]


def collect_full_matches(limit: int | None = None) -> list[int]:
    df = competitions()
    m_ids: list[int] = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        try:
            ms = matches(int(r.competition_id), int(r.season_id))
            m_ids.extend(ms["match_id"].astype(int).tolist())
            if limit and len(m_ids) >= limit:
                break
        except Exception:
            continue
    return m_ids if not limit else m_ids[:limit]
