# src/plots.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# ---------- SHOT MAP (interactive) ----------
def make_shot_map_plotly(
    df: pd.DataFrame,
    *,
    title: str = "Shot Map (bubble = xG, red = goal)",
    fig_size: Tuple[int, int] = (880, 480),
) -> go.Figure:
    if df.empty:
        return _pitch_figure(fig_size=fig_size)

    # Colors by outcome
    goal_mask = (df.get("goal_scored", 0).astype(int) == 1)
    colors = np.where(goal_mask, "crimson", "steelblue")

    # Bubble size from xG (fallback small if missing)
    xg = df.get("xG", pd.Series(0.05, index=df.index)).clip(0, 0.9)
    sizes = (xg * 36.0) + 6.0

    hovertext = [
        "<b>{player}</b> — {team}<br>"
        "xG: {xg:.2f} • Minute: {minute}<br>"
        "Outcome: {outcome}<br>"
        "Body: {body} • Tech: {tech}<br>"
        "(x,y)=({x:.1f},{y:.1f})".format(
            player=str(row.get("player", "")),
            team=str(row.get("team", "")),
            xg=float(row.get("xG", 0.0)),
            minute=int(row.get("minute", 0)) if pd.notna(row.get("minute", None)) else 0,
            outcome="Goal" if int(row.get("goal_scored", 0)) == 1 else "No goal",
            body=str(row.get("body_part", "")),
            tech=str(row.get("technique", "")),
            x=float(row.get("x", np.nan)),
            y=float(row.get("y", np.nan)),
        )
        for _, row in df.iterrows()
    ]

    fig = _pitch_figure(fig_size=fig_size)
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.75, line=dict(width=0)),
            hovertext=hovertext,
            hoverinfo="text",
            name="Shots",
        )
    )
    fig.update_layout(
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _pitch_figure(*, fig_size: Tuple[int, int] = (880, 480)) -> go.Figure:
    """StatsBomb 120x80 pitch as Plotly shapes with equal aspect."""
    w, h = fig_size
    fig = go.Figure()
    line = dict(color="black", width=2)  # thicker lines

    shapes = [
        # Outer boundaries
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=line, fillcolor="white"),
        # Halfway line
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=line),
        # Penalty boxes
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=line),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=line),
        # Six-yard boxes
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=line),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=line),
        # Goals
        dict(type="rect", x0=-1, y0=36, x1=0, y1=44, line=line),
        dict(type="rect", x0=120, y0=36, x1=121, y1=44, line=line),
        # Center circle (approx 10m)
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=line),
        # Spots
        dict(type="circle", x0=59.6, y0=39.6, x1=60.4, y1=40.4, line=line),
        dict(type="circle", x0=11.6, y0=39.6, x1=12.4, y1=40.4, line=line),
        dict(type="circle", x0=107.6, y0=39.6, x1=108.4, y1=40.4, line=line),
    ]
    fig.update_layout(
        width=w,
        height=h,
        xaxis=dict(
            range=[-1, 121],
            showgrid=False,
            zeroline=False,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            fixedrange=True,
        ),
        yaxis=dict(range=[-1, 81], showgrid=False, zeroline=False, fixedrange=True),
        shapes=shapes,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------- PASSING NETWORK (interactive) ----------
def make_passing_network_plotly(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    title: str = "Passing Network",
    fig_size: Tuple[int, int] = (880, 480),
) -> go.Figure:
    w, h = fig_size
    fig = _pitch_figure(fig_size=(w, h))

    # Edges first
    if not edges.empty:
        n = len(edges)
        xs = np.empty(n * 3, dtype=float)
        ys = np.empty(n * 3, dtype=float)
        xs[0::3] = edges["x0"].to_numpy(dtype=float)
        xs[1::3] = edges["x1"].to_numpy(dtype=float)
        xs[2::3] = np.nan
        ys[0::3] = edges["y0"].to_numpy(dtype=float)
        ys[1::3] = edges["y1"].to_numpy(dtype=float)
        ys[2::3] = np.nan

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(
                    width=edges["w"].clip(lower=0.8, upper=8).tolist(),
                    color="black",
                ),
                hovertext=edges["label"],
                hoverinfo="text",
                opacity=0.70,
                showlegend=False,
            )
        )

    # Nodes on top
    if not nodes.empty:
        fig.add_trace(
            go.Scatter(
                x=nodes["x"],
                y=nodes["y"],
                mode="markers+text",
                text=nodes["label"],
                textposition="top center",
                marker=dict(
                    size=nodes["size"].clip(lower=6, upper=40),
                    color="royalblue",
                    opacity=0.90,
                    line=dict(color="white", width=1),
                ),
                hovertext=nodes["hover"],
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.update_layout(title=title)
    return fig


def build_network_tables(
    df: pd.DataFrame,
    min_passes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sx = "start_x" if "start_x" in df.columns else "x"
    sy = "start_y" if "start_y" in df.columns else "y"

    start = (
        df.groupby("passer")[[sx, sy]]
        .mean(numeric_only=True)
        .rename(columns={sx: "sx", sy: "sy"})
    )

    if {"end_x", "end_y"}.issubset(df.columns):
        end = df.groupby("receiver")[["end_x", "end_y"]].mean(numeric_only=True)
        end = end.rename(columns={"end_x": "rx", "end_y": "ry"})
    else:
        end = None

    players = sorted(set(df["passer"]) | set(df["receiver"].dropna()))
    pos = pd.DataFrame(index=players).join(start, how="left")

    if end is not None:
        pos = pos.join(end, how="left")
        pos["x"] = pos[["sx", "rx"]].mean(axis=1, skipna=True)
        pos["y"] = pos[["sy", "ry"]].mean(axis=1, skipna=True)
    else:
        pos["x"] = pos["sx"]
        pos["y"] = pos["sy"]

    pos["x"] = pos["x"].fillna(60.0).clip(0, 120)
    pos["y"] = pos["y"].fillna(40.0).clip(0, 80)

    edges = (
        df.groupby(["passer", "receiver"], dropna=True)
        .size()
        .reset_index(name="weight")
    )
    edges = edges[(edges["passer"] != edges["receiver"]) & edges["receiver"].notna()]
    if min_passes > 1:
        edges = edges[edges["weight"] >= min_passes]

    G = nx.DiGraph()
    for passer, receiver, w in edges[["passer", "receiver", "weight"]].itertuples(index=False):
        G.add_edge(passer, receiver, weight=float(w))
    cent = nx.betweenness_centrality(G, weight="weight", normalized=True) if G.number_of_edges() else {}

    nodes = pd.DataFrame({
        "player": players,
        "x": [float(pos.at[p, "x"]) for p in players],
        "y": [float(pos.at[p, "y"]) for p in players],
        "size": [max(cent.get(p, 0.0), 0.01) * 36.0 for p in players],
        "label": players,
        "hover": [f"<b>{p}</b><br>Betweenness: {cent.get(p, 0.0):.3f}" for p in players],
    })

    def width_map(w: float) -> float:
        return 0.8 + 0.7 * np.log1p(max(w, 1.0))

    rows = []
    for passer, receiver, w in edges[["passer", "receiver", "weight"]].itertuples(index=False):
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
