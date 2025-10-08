# tests/test_plots.py

from __future__ import annotations

import pandas as pd

from src.dashboard.plots import make_shot_map_plotly, build_network_tables


def _sample_shots():
    return pd.DataFrame(
        {
            "x": [10, 100],
            "y": [20, 60],
            "xG": [0.1, 0.4],
            "goal_scored": [0, 1],
            "player": ["A", "B"],
            "team": ["Team 1", "Team 2"],
            "minute": [12, 78],
            "body_part": ["Left Foot", "Head"],
            "technique": ["Normal", "Volley"],
        }
    )


def test_make_shot_map_plotly_returns_pitch():
    fig = make_shot_map_plotly(_sample_shots(), fig_size=(600, 400))
    assert len(fig.data) == 1
    assert fig.layout.xaxis.range == (-1, 121)
    assert fig.layout.yaxis.range == (-1, 81)
    assert fig.data[0]["mode"] == "markers"


def test_build_network_tables_filters_min_passes():
    df = pd.DataFrame(
        {
            "passer": ["A", "A", "B"],
            "receiver": ["B", "B", "C"],
            "x": [10, 15, 40],
            "y": [10, 12, 30],
            "end_x": [20, 22, 45],
            "end_y": [18, 22, 32],
        }
    )
    nodes, edges = build_network_tables(df, min_passes=2)
    assert len(nodes) == 3
    # Only A->B should remain with min_passes=2
    assert len(edges) == 1
    assert "A" in edges.iloc[0]["label"]
