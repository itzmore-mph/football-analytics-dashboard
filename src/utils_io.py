from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pandas as pd


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _cached_read_remote(url: str) -> Any:
    requests = cast(Any, importlib.import_module("requests"))

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def read_remote_json(url: str) -> Any:
    try:
        return _cached_read_remote(url)
    except Exception as exc:  # pragma: no cover - network fallback
        fallback = _local_sample_path(url)
        if fallback and fallback.exists():
            return read_json(fallback)
        msg = f"Failed to fetch {url}: {exc}"
        raise RuntimeError(msg) from exc


def _local_sample_path(url: str) -> Path | None:
    from .config import settings

    marker = "/open-data/master/data/"
    if marker not in url:
        return None
    relative = url.split(marker, 1)[1]
    return settings.sample_data_dir / relative


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df
