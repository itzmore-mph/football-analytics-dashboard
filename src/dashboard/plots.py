# src/dashboard/plots.py
from typing import Optional, Literal, Dict, Tuple
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
from mplsoccer import Pitch


# Shot map plot
def plot_shot_map(df: Optional[pd.DataFrame]) -> plt.Figure:
    """Return a pitch shot map colored by xG.

    Args:
        df: DataFrame with columns ['x','y','xG'], or None.
    Raises:
        ValueError: if df is None or missing required columns.
    """
    required = {"x", "y", "xG"}
    if df is None or not required.issubset(df.columns):
        raise ValueError("Invalid or missing shot data")

    pitch = Pitch(
        pitch_type="statsbomb", pitch_color="white", line_color="black"
    )
    fig, ax = pitch.draw(figsize=(8, 5))
    scatter = ax.scatter(
        df["x"], df["y"], c=df["xG"], cmap="Reds", edgecolors="black", s=80
    )
    fig.colorbar(scatter, ax=ax, label="Expected Goals (xG)")
    return fig


# Passing network plot
Layout = Literal["statsbomb", "spring", "circular"]

def plot_passing_network(
    df: Optional[pd.DataFrame],
    *,
    min_pass: int = 2,
    layout: Layout = "statsbomb",
    show_labels: bool = False
) -> plt.Figure:
    """Return a cleaned-up passing network.

    Args:
        df: DataFrame with ['passer','receiver','x','y','pass_count'].
        min_pass: only include edges where pass_count >= this.
        layout: "statsbomb" (use on-pitch coords), "spring", or "circular".
        show_labels: draw every node label if True, otherwise only top-5.
    Raises:
        ValueError: if data is missing or no edges survive filtering.
    """
    req = {"passer", "receiver", "x", "y", "pass_count"}
    if df is None or not req.issubset(df.columns):
        raise ValueError("No valid passing data to plot.")

    # filter low-frequency links
    df = df[df.pass_count >= min_pass]
    if df.empty:
        raise ValueError(f"No links ≥ {min_pass} passes.")

    # build directed graph
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(r.passer, r.receiver, weight=r.pass_count)

    # prepare temporary pitch for axis limits
    temp_pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    _, temp_ax = temp_pitch.draw(figsize=(8, 5))
    x0, x1 = temp_ax.get_xlim()
    y0, y1 = temp_ax.get_ylim()

    # choose node positions
    if layout == "statsbomb":
        pos: Dict[str, Tuple[float, float]] = {}
        for _, r in df.iterrows():
            jitter_x = r.x + random.uniform(-1, 1)
            jitter_y = r.y + random.uniform(-1, 1)
            pos.setdefault(r.passer, (jitter_x, jitter_y))
            pos.setdefault(r.receiver, (jitter_x, jitter_y))
    elif layout == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=42)
        # rescale spring coords into pitch space
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        for node, (x, y) in pos.items():
            norm_x = (x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
            norm_y = (y - min_y) / (max_y - min_y) if max_y > min_y else 0.5
            pos[node] = (x0 + norm_x * (x1 - x0), y0 + norm_y * (y1 - y0))
    else:  # circular
        pos = nx.circular_layout(G)
        # rescale circular coords into pitch space
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        for node, (x, y) in pos.items():
            norm_x = (x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
            norm_y = (y - min_y) / (max_y - min_y) if max_y > min_y else 0.5
            pos[node] = (x0 + norm_x * (x1 - x0), y0 + norm_y * (y1 - y0))

    # sizing
    deg = dict(G.degree(weight="weight"))
    min_size, max_size = 50, 500
    sizes = {n: deg[n] for n in G.nodes()}
    max_deg = max(sizes.values(), default=1)
    node_sizes = [
        min_size + (max_size - min_size) * (sizes[n]**0.5 / max_deg**0.5)
        for n in G.nodes()
    ]
    edge_widths = [d["weight"] * 0.2 for (_, _, d) in G.edges(data=True)]

    # draw on a pitch
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black").draw(figsize=(8, 5))
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="gray", alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="steelblue", alpha=0.6)

    # labels
    if show_labels:
        for node, (x, y) in pos.items():
            ax.text(x, y + 1.5, node, fontsize=8, ha="center", va="bottom", color="black")
    else:
        # annotate only the top-5 players by betweenness
        bet = nx.betweenness_centrality(G, weight="weight")
        top5 = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]
        for player, _ in top5:
            x, y = pos[player]
            ax.text(
                x, y + 0.02, player,
                ha="center", fontsize=9,
                color="darkred", weight="bold"
            )
    ax.set_title("Passing Network", pad=10)
    return fig
