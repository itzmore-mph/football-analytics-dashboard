from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"
    models_dir: Path = root / "models"
    plots_dir: Path = models_dir / "plots"
    demo_matches: int = 4  # number of matches to fetch in demo
    cache_seconds: int = 3600


@property
def processed_shots_csv(self) -> Path:
    return self.data_dir / "processed_shots.csv"


@property
def passing_events_csv(self) -> Path:
    return self.data_dir / "passing_events.csv"


@property
def model_path(self) -> Path:
    return self.models_dir / "xg_calibrated.joblib"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.plots_dir.mkdir(parents=True, exist_ok=True)
