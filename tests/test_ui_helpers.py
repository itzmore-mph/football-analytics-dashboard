# tests/test_ui_helpers.py

from __future__ import annotations

import pandas as pd

from src.ui import _minute_range


def test_minute_range_caps_upper_bound():
    df = pd.DataFrame({"minute": [5, 30, 150]})
    start, end = _minute_range(df)
    assert start == 0
    assert end == 105  # capped at extra time window


def test_minute_range_defaults_when_missing():
    df = pd.DataFrame({"x": [1, 2]})
    assert _minute_range(df) == (0, 105)
