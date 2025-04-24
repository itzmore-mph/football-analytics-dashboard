# src/dashboard/plots.py
from typing import Optional, Literal
import pandas as pd
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

    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
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
        show_labels: draw every node label if True, otherwise only top‐5.
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

    # choose node positions
    if layout == "statsbomb":
        pos: dict = {}
        for _, r in df.iterrows():
            pos.setdefault(r.passer, (r.x, r.y))
            pos.setdefault(r.receiver, (r.x, r.y))
    elif layout == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=42)
    else:  # circular
        pos = nx.circular_layout(G)

    # sizing
    deg = dict(G.degree(weight="weight"))
    node_sizes = [deg[n] * 200 for n in G.nodes()]
    edge_widths = [d["weight"] * 0.3 for (_, _, d) in G.edges(data=True)]

    # draw on a pitch
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = pitch.draw(figsize=(8, 5))
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths, edge_color="black", alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes, node_color="steelblue", alpha=0.8)

    # labels
    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    else:
        # annotate only the top-5 players by betweenness
        bet = nx.betweenness_centrality(G, weight="weight")
        top5 = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]
        for player, _ in top5:
            x, y = pos[player]
            ax.text(x, y + 0.02, player,
                    ha="center", fontsize=9,
                    color="darkred", weight="bold")

    ax.set_title("Passing Network", pad=10)
    return fig