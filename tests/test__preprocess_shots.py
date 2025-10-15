from __future__ import annotations

import pandas as pd

from src.features_xg import FEATURE_COLUMNS, build_basic_features


def test_build_basic_features_columns():
    df = pd.DataFrame(
        {
            "location.x": [100.0],
            "location.y": [40.0],
            "body_part.name": ["Right Foot"],
            "play_pattern.name": ["Open Play"],
            "under_pressure": [False],
            "minute": [12],
        }
    )
    out = build_basic_features(df)
    for c in FEATURE_COLUMNS:
        assert c in out.columns
