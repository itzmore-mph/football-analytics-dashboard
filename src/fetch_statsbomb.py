# src/fetch_statsbomb.py
from __future__ import annotations

from pathlib import Path
import json
from typing import Final

import requests
from requests import RequestException


class StatsBombFetchError(RuntimeError):
    """Raised when StatsBomb open-data files cannot be downloaded."""


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA: Final[Path] = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

RAW_BASE: Final[str] = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "football-analytics-dashboard"},
        )
        r.raise_for_status()
    except RequestException as exc:  # pragma: no cover - network failures vary
        raise StatsBombFetchError(
            f"Could not download StatsBomb open data: {url} ({exc})"
        ) from exc

    out.write_bytes(r.content)
    # Validate JSON to catch HTML error pages early
    try:
        json.loads(out.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        out.unlink(missing_ok=True)
        raise StatsBombFetchError(
            f"Downloaded file is not valid JSON: {url}"
        ) from exc
    return out


def fetch_competitions(force: bool = False) -> Path:
    out = DATA / "competitions.json"
    if out.exists() and not force:
        return out
    url = f"{RAW_BASE}/competitions.json"
    return _download(url, out)


def fetch_matches(
    competition_id: int, season_id: int, force: bool = False
) -> Path:
    """Lädt matches/<comp>/<season>.json."""
    out = DATA / "matches" / str(competition_id) / f"{season_id}.json"
    if out.exists() and not force:
        return out
    url = f"{RAW_BASE}/matches/{int(competition_id)}/{int(season_id)}.json"
    return _download(url, out)


def fetch_events(match_id: int, force: bool = False) -> Path:
    out = DATA / f"{int(match_id)}.json"
    if out.exists() and not force:
        return out
    url = f"{RAW_BASE}/events/{int(match_id)}.json"
    return _download(url, out)


def load_competitions_df():
    import pandas as pd
    p = fetch_competitions()
    return pd.read_json(p)


def load_matches_df(competition_id: int, season_id: int):
    import pandas as pd
    p = fetch_matches(competition_id, season_id)
    return pd.read_json(p)


def load_events_json(match_id: int) -> list[dict]:
    p = fetch_events(match_id)
    return json.loads(p.read_text(encoding="utf-8"))
