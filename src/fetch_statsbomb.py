# src/fetch_statsbomb.py
from pathlib import Path
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "football-analytics-dashboard"}
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} für {url}")
    out.write_bytes(r.content)
    # Validieren
    json.loads(out.read_text(encoding="utf-8"))
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
