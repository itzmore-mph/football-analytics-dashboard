from __future__ import annotations
import plotly.graph_objects as go
import pandas as pd


def cumulative_xg_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for team, sub in df.groupby("team.name"):
        sub = sub.sort_values("minute").copy()
        sub["cum_xg"] = sub["xg"].cumsum()
    fig.add_trace(
        go.Scatter(
            x=sub["minute"],
            y=sub["cum_xg"],
            mode="lines",
            name=team,
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    fig.update_layout(legend_title_text="Team")
    fig.update_layout(width="stretch")
    return fig
