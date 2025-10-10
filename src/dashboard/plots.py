# src/plots.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# Pitch primitives (thicker, crisp StatsBomb 120x80)
# ============================================================


def _pitch_shapes(
    line_color: str = "#111111",
    line_width: float = 2.2,
) -> list[dict]:
    """Plotly shapes for a 120x80 StatsBomb pitch with crisp, thick lines."""
    L, W = 120, 80
    shapes: list[dict] = []

    # Outer frame
    shapes.append(dict(
        type="rect", x0=0, y0=0, x1=L, y1=W,
        line=dict(color=line_color, width=line_width),
        fillcolor="rgba(0,0,0,0)",
    ))

    # Halfway line
    shapes.append(dict(
        type="line", x0=L/2, y0=0, x1=L/2, y1=W,
        line=dict(color=line_color, width=line_width),
    ))

    # Center circle (r ≈ 10)
    shapes.append(dict(
        type="circle",
        x0=L/2 - 10, y0=W/2 - 10,
        x1=L/2 + 10, y1=W/2 + 10,
        line=dict(color=line_color, width=line_width),
        fillcolor="rgba(0,0,0,0)",
    ))

    # Penalty boxes (18 deep, 44 wide centered)
    shapes += [
        dict(type="rect", x0=L-18, y0=W/2-22, x1=L, y1=W/2+22,
             line=dict(color=line_color, width=line_width),
             fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=0, y0=W/2-22, x1=18, y1=W/2+22,
             line=dict(color=line_color, width=line_width),
             fillcolor="rgba(0,0,0,0)"),
    ]

    # Six-yard boxes (6 deep, 20 wide centered)
    shapes += [
        dict(type="rect", x0=L-6, y0=W/2-10, x1=L, y1=W/2+10,
             line=dict(color=line_color, width=line_width),
             fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=0, y0=W/2-10, x1=6, y1=W/2+10,
             line=dict(color=line_color, width=line_width),
             fillcolor="rgba(0,0,0,0)"),
    ]

    # Goals (thin)
    shapes += [
        dict(type="rect", x0=L, y0=W/2-3.66, x1=L+2, y1=W/2+3.66,
             line=dict(color=line_color, width=1.2),
             fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=-2, y0=W/2-3.66, x1=0, y1=W/2+3.66,
             line=dict(color=line_color, width=1.2),
             fillcolor="rgba(0,0,0,0)"),
    ]

    return shapes


def _pitch_figure(
    *,
    fig_size: Tuple[int, int] = (900, 520),
    line_color: str = "#111111",
    line_width: float = 2.2,
    bg_color: str = "#FFFFFF",
) -> go.Figure:
    """StatsBomb 120x80 pitch with equal aspect & thick lines."""
    L, W = 120, 80
    w, h = fig_size
    fig = go.Figure()
    fig.update_layout(
        width=w,
        height=h,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        xaxis=dict(visible=False, range=[-2, L + 2]),
        yaxis=dict(
            visible=False,
            range=[0, W],
            scaleanchor="x",
            scaleratio=1,
        ),
        shapes=_pitch_shapes(line_color=line_color, line_width=line_width),
    )

    # Penalty spots & center spot
    fig.add_trace(go.Scatter(
        x=[108, 12, L/2], y=[W/2, W/2, W/2],
        mode="markers",
        marker=dict(size=6, color=line_color),
        hoverinfo="skip",
        showlegend=False,
    ))
    return fig


# ============================================================
# Shot Map (interactive)
# ============================================================

def make_shot_map_plotly(
    df: pd.DataFrame,
    *,
    title: str = "Shot Map (bubble = xG, red = goal)",
    fig_size: Tuple[int, int] = (900, 520),
) -> go.Figure:
    """
    Interactive shot map for StatsBomb 120x80 coordinates.
    Rich hover; thicker pitch lines for readability.
    """
    if df.empty:
        return _pitch_figure(fig_size=fig_size)

    # Outcome colors
    goal = df.get("goal_scored", pd.Series([0] * len(df))).astype(int)
    colors = np.where(goal.eq(1), "#e63946", "#1d3557")

    # Bubble size from xG (fallback 0)
    xg = df.get("xG", pd.Series([0.0] * len(df))).clip(0, 0.95)
    sizes = (xg * 520.0) + 16.0

    # Hover fields
    custom = np.stack([
        df.get("player", pd.Series([""] * len(df))).astype(str).fillna(""),
        df.get("team", pd.Series([""] * len(df))).astype(str).fillna(""),
        df.get("minute", pd.Series([np.nan] * len(df))).fillna("").astype(str),
        xg.values,
        np.where(goal.eq(1), "Goal", "Shot"),
        df.get("body_part", pd.Series([""] * len(df))).astype(str).fillna(""),
        df.get("technique", pd.Series([""] * len(df))).astype(str).fillna(""),
    ], axis=1)

    hover = (
        "Player: %{customdata[0]}<br>"
        "Team: %{customdata[1]}<br>"
        "Minute: %{customdata[2]}<br>"
        "xG: %{customdata[3]:.2f}<br>"
        "Outcome: %{customdata[4]}<br>"
        "Body part: %{customdata[5]}<br>"
        "Technique: %{customdata[6]}"
    )

    fig = _pitch_figure(fig_size=fig_size)
    fig.update_layout(title=dict(text=title, x=0.02, y=0.95,
                                 font=dict(size=16, color="#111")))

    fig.add_trace(go.Scattergl(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.75,
            line=dict(width=0),
        ),
        customdata=custom,
        hovertemplate=hover,
        showlegend=False,
    ))
    return fig


# ============================================================
# Passing Network (interactive)
# ============================================================

def make_passing_network_plotly(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    title: str = "Passing Network",
    fig_size: Tuple[int, int] = (900, 520),
) -> go.Figure:
    """
    Interactive passing network from pre-computed:
    nodes: (player, x, y, size, label, hover)
    edges: (x0, y0, x1, y1, w, label)
    Renders variable-width edges (one trace per edge) for clarity.
    """
    fig = _pitch_figure(fig_size=fig_size)

    # Edges (one trace per edge to support variable width)
    if not edges.empty:
        for row in edges.itertuples(index=False):
            fig.add_trace(go.Scatter(
                x=[row.x0, row.x1],
                y=[row.y0, row.y1],
                mode="lines",
                line=dict(width=float(row.w), color="#111111"),
                hovertext=row.label,
                hoverinfo="text",
                opacity=0.65,
                showlegend=False,
            ))

    # Nodes on top
    if not nodes.empty:
        fig.add_trace(go.Scatter(
            x=nodes["x"],
            y=nodes["y"],
            mode="markers+text",
            text=nodes["label"],
            textposition="top center",
            marker=dict(
                size=nodes["size"].clip(lower=8, upper=40),
                color="royalblue",
                opacity=0.9,
                line=dict(color="white", width=1.2),
            ),
            hovertext=nodes["hover"],
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(title=dict(
        text=title,
        x=0.02,
        y=0.95,
        font=dict(size=16, color="#111")))
    return fig


def build_network_tables(
    df: pd.DataFrame,
    min_passes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From raw pass dataframe, compute:
    - nodes: x,y per player + size by betweenness; hover text
    - edges: segments x0,y0 -> x1,y1, width by weight; hover text
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Mean positions
    sx = "start_x" if "start_x" in df.columns else "x"
    sy = "start_y" if "start_y" in df.columns else "y"
    start = (
        df.groupby("passer")[[sx, sy]]
        .mean(numeric_only=True)
        .rename(columns={sx: "sx", sy: "sy"})
    )
    end = (
        df.groupby("receiver")[["end_x", "end_y"]]
        .mean(numeric_only=True)
        if {"end_x", "end_y"}.issubset(df.columns) else None
    )
    players = sorted(set(df["passer"]) | set(df["receiver"]))
    pos = pd.DataFrame(index=players).join(start, how="left")
    if end is not None:
        pos = pos.join(
            end.rename(columns={"end_x": "rx", "end_y": "ry"}),
            how="left"
        )
        pos["x"] = pos[["sx", "rx"]].mean(axis=1, skipna=True)
        pos["y"] = pos[["sy", "ry"]].mean(axis=1, skipna=True)
    else:
        pos["x"], pos["y"] = pos["sx"], pos["sy"]
    pos["x"] = pos["x"].fillna(60.0).clip(0, 120)
    pos["y"] = pos["y"].fillna(40.0).clip(0, 80)

    # Edge weights
    edges_raw = (
        df.groupby(["passer", "receiver"], dropna=True)
        .size()
        .reset_index(name="weight")
    )
    edges_raw = edges_raw[
        (edges_raw["passer"] != edges_raw["receiver"]) &
        (edges_raw["receiver"].notna())
    ]
    if min_passes > 1:
        edges_raw = edges_raw[edges_raw["weight"] >= min_passes]

    # Graph + betweenness
    import networkx as nx
    G = nx.DiGraph()
    for p, r, w in edges_raw[
        ["passer", "receiver", "weight"]
        ].itertuples(
        index=False
    ):
        G.add_edge(p, r, weight=float(w))
    cent = (
        nx.betweenness_centrality(G, weight="weight", normalized=True)
        if G.number_of_edges() else {}
    )

    # Nodes table
    nodes = pd.DataFrame({
        "player": players,
        "x": [float(pos.at[p, "x"]) for p in players],
        "y": [float(pos.at[p, "y"]) for p in players],
        "size": [max(cent.get(p, 0.0), 0.01) * 36.0 for p in players],
        "label": players,
        "hover": [
            f"<b>{p}</b><br>Betweenness: {cent.get(p, 0.0):.3f}"
            for p in players
        ],
    })

    # Edge table (each row becomes one line trace with width mapping)
    def width_map(w: float) -> float:
        # Slightly compress range but keep contrast
        return max(0.8, 0.8 * np.log1p(max(w, 1.0)))

    rows = []
    for passer, receiver, w in edges_raw[
        ["passer", "receiver", "weight"]
        ].itertuples(
        index=False
    ):
        if passer not in pos.index or receiver not in pos.index:
            continue
        rows.append({
            "x0": float(pos.at[passer, "x"]),
            "y0": float(pos.at[passer, "y"]),
            "x1": float(pos.at[receiver, "x"]),
            "y1": float(pos.at[receiver, "y"]),
            "w": width_map(float(w)),
            "label": f"{passer} → {receiver}<br>Passes: {int(w)}",
        })

    edges_tbl = pd.DataFrame(rows)
    return nodes, edges_tbl
