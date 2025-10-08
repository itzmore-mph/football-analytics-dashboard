"""Light-weight validation helpers for StatsBomb open-data tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    frame: pd.DataFrame
    issues: list[str]

    def warn_on(self, warn: callable) -> pd.DataFrame:
        for issue in self.issues:
            warn(issue)
        return self.frame


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_shots(df: pd.DataFrame) -> ValidationResult:
    required = {"x", "y", "shot_distance", "shot_angle", "goal_scored"}
    issues: list[str] = []
    missing = required - set(df.columns)
    if missing:
        issues.append(
            "Shot data missing required columns: " + ", ".join(sorted(missing))
        )

    dfc = df.copy()
    dfc = _coerce_numeric(
        dfc,
        ["x", "y", "shot_distance", "shot_angle", "xG", "minute", "second"],
    )

    # Drop impossible coordinates/distances
    coord_mask = (
        dfc["x"].between(0, 120, inclusive="both")
        & dfc["y"].between(0, 80, inclusive="both")
    )
    dropped = int(len(dfc) - coord_mask.sum())
    if dropped:
        issues.append(f"Dropped {dropped} shots outside the pitch bounds.")
    dfc = dfc[coord_mask]

    if "goal_scored" in dfc.columns:
        dfc["goal_scored"] = dfc["goal_scored"].fillna(0).astype(int)

    # Minutes should be non-negative
    if "minute" in dfc.columns:
        neg_minutes = dfc["minute"].lt(0).sum()
        if neg_minutes:
            issues.append(
                f"Found {neg_minutes} shots with negative minute; set to 0."
            )
            dfc.loc[dfc["minute"].lt(0), "minute"] = 0

    return ValidationResult(dfc, issues)


def validate_passes(df: pd.DataFrame) -> ValidationResult:
    required = {"passer", "receiver", "x", "y"}
    issues: list[str] = []
    missing = required - set(df.columns)
    if missing:
        issues.append(
            "Passing data missing required columns: "
            + ", ".join(sorted(missing))
        )

    dfc = df.copy()
    dfc = _coerce_numeric(
        dfc,
        ["x", "y", "start_x", "start_y", "end_x", "end_y", "minute", "second"],
    )

    mask = (
        dfc["x"].between(0, 120, inclusive="both")
        & dfc["y"].between(0, 80, inclusive="both")
    )
    dropped = int(len(dfc) - mask.sum())
    if dropped:
        issues.append(f"Dropped {dropped} passes outside the pitch bounds.")
    dfc = dfc[mask]

    if "minute" in dfc.columns:
        neg = dfc["minute"].lt(0).sum()
        if neg:
            issues.append(
                f"Found {neg} passes with negative minute; set to 0."
            )
            dfc.loc[dfc["minute"].lt(0), "minute"] = 0

    return ValidationResult(dfc, issues)


def sample_size_warning(df: pd.DataFrame, threshold: int = 10) -> str | None:
    if len(df) < threshold:
        return (
            f"Only {len(df)} rows available. Insights may not be reliable; "
            "consider selecting a different match or timeframe."
        )
    return None
