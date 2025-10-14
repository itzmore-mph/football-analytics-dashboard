from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def cumulative_xg_plot(df: pd.DataFrame) -> go.Figure:
    s = (
        df.sort_values("minute").assign(cum_xg=df["xg"].cumsum())
    )
    fig = go.Figure()
    fig.add_scatter(
        x=s["minute"],
        y=s["cum_xg"],
        mode="lines+markers",
        name="Cumulative xG",
    )
    fig.update_layout(
        autosize=True,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Minute",
        yaxis_title="Cumulative xG",
        hovermode="x unified",
    )
    return fig
