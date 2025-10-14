from __future__ import annotations
import json
from os import path
from pathlib import Path
from typing import Any, Iterable


import pandas as pd
from functools import lru_cache


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _cached_read_remote(url: str) -> Any:
    import requests

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def read_remote_json(url: str) -> Any:
    return _cached_read_remote(url)


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df