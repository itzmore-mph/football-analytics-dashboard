import pandas as pd

from src.ui import _minute_range, _comp_label, _match_label, _read_json


def test_minute_range_no_minute_column():
    df = pd.DataFrame({"x": [1, 2]})
    assert _minute_range(df) == (0, 105)


def test_minute_range_with_minutes():
    df = pd.DataFrame({"minute": [10, 50, 90]})
    mn_from, mn_to = _minute_range(df)
    assert mn_from == 0
    assert 45 <= mn_to <= 105


def test_comp_and_match_label_roundtrip():
    row = type("R", (), {
        "competition_name": "Champions",
        "season_name": "2024",
        "competition_id": 1,
        "season_id": 2
    })()
    assert "Champions" in _comp_label(row)

    m = {
        "home_team": {"home_team_name": "A"},
        "away_team": {"away_team_name": "B"},
        "match_date": "2024-01-01",
        "match_id": 42
    }
    assert "A vs B" in _match_label(m)


def test_read_json_missing(tmp_path):
    p = tmp_path / "nope.json"
    df = _read_json(p)
    assert df.empty
