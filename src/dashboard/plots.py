from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def cumulative_xg_plot(df: pd.DataFrame) -> go.Figure:
    """Create cumulative xG plot with separate lines for each team."""
    fig = go.Figure()

    # Check if we have team data
    if "team.name" in df.columns:
        teams = df["team.name"].unique()
        colors = ["#60a5fa", "#ef4444", "#22c55e", "#fbbf24"]

        for i, team in enumerate(teams):
            team_df = df[df["team.name"] == team].sort_values("minute")
            team_df = team_df.assign(cum_xg=team_df["xg"].cumsum())

            fig.add_scatter(
                x=team_df["minute"],
                y=team_df["cum_xg"],
                mode="lines+markers",
                name=team,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
            )
    else:
        # Fallback to single line if no team data
        s = df.sort_values("minute").assign(cum_xg=df["xg"].cumsum())
        fig.add_scatter(
            x=s["minute"],
            y=s["cum_xg"],
            mode="lines+markers",
            name="Cumulative xG",
        )

    fig.update_layout(
        autosize=True,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Minute",
        yaxis_title="Cumulative xG",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    return fig
