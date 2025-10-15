"""Tests for dashboard components and plots."""

import pandas as pd

from src.dashboard.plots import cumulative_xg_plot


def test_cumulative_xg_plot_with_teams():
    """Test that cumulative xG plot works with team data."""
    df = pd.DataFrame(
        {
            "minute": [10, 20, 30, 40, 50],
            "xg": [0.1, 0.3, 0.2, 0.5, 0.1],
            "team.name": ["Team A", "Team B", "Team A", "Team B", "Team A"],
        }
    )
    fig = cumulative_xg_plot(df)
    assert fig is not None
    assert len(fig.data) == 2  # Two teams
    assert fig.layout.xaxis.title.text == "Minute"
    assert fig.layout.yaxis.title.text == "Cumulative xG"


def test_cumulative_xg_plot_without_teams():
    """Test that cumulative xG plot works without team data."""
    df = pd.DataFrame(
        {
            "minute": [10, 20, 30, 40, 50],
            "xg": [0.1, 0.3, 0.2, 0.5, 0.1],
        }
    )
    fig = cumulative_xg_plot(df)
    assert fig is not None
    assert len(fig.data) == 1  # Single line
    assert fig.layout.xaxis.title.text == "Minute"
    assert fig.layout.yaxis.title.text == "Cumulative xG"


def test_cumulative_xg_plot_empty():
    """Test that cumulative xG plot handles empty dataframe."""
    df = pd.DataFrame({"minute": [], "xg": [], "team.name": []})
    fig = cumulative_xg_plot(df)
    assert fig is not None
    assert len(fig.data) == 0  # No data to plot
