from src.ui import _load_shots_df


def test_load_shots_missing(tmp_path):
    root = tmp_path
    df, issues = _load_shots_df(root)
    assert df.empty
    assert isinstance(issues, list)
