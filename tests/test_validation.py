# tests/test_validation.py

from __future__ import annotations

import pandas as pd

from src.validation import validate_shots, validate_passes, sample_size_warning


def test_validate_shots_filters_out_of_bounds():
    df = pd.DataFrame(
        {
            "x": [10, 130],
            "y": [20, -5],
            "shot_distance": [12.0, 5.0],
            "shot_angle": [0.4, 0.2],
            "goal_scored": [1, 0],
        }
    )

    result = validate_shots(df)

    assert len(result.frame) == 1
    assert result.frame.iloc[0]["x"] == 10
    assert any("outside the pitch" in msg for msg in result.issues)


def test_validate_passes_negative_minutes_clamped():
    df = pd.DataFrame(
        {
            "passer": ["A"],
            "receiver": ["B"],
            "x": [30],
            "y": [40],
            "minute": [-2],
        }
    )

    result = validate_passes(df)

    assert result.frame.iloc[0]["minute"] == 0
    assert any("negative minute" in msg for msg in result.issues)


def test_sample_size_warning_triggers():
    small_df = pd.DataFrame({"x": [1, 2]})
    msg = sample_size_warning(small_df, threshold=5)
    assert msg is not None
    assert "Only 2 rows" in msg
