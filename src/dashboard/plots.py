from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def cumulative_xg_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for team, sub in df.groupby("team.name"):
        ordered = sub.sort_values("minute").copy()
        ordered["cum_xg"] = ordered["xg"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=ordered["minute"],
                y=ordered["cum_xg"],
                mode="lines",
                name=team,
            )
        )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    fig.update_layout(legend_title_text="Team")
    fig.update_layout(width="stretch")
    return fig
