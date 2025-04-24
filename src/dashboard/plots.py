# src/dashboard/plots.py
from typing import Optional
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
def plot_passing_network(
    df: Optional[pd.DataFrame], 
    min_pass: int = 2, 
    layout: str = "spring", 
    show_labels: bool = False
) -> plt.Figure:
    """Return a cleaned-up passing network plot.

    Args:
        df: DataFrame with ['passer','receiver','x','y','pass_count'].
        min_pass: minimum number of passes to include an edge.
        layout: one of "statsbomb_coords", "spring", or "circular".
        show_labels: whether to draw text labels on every node.
    Raises:
        ValueError: if df is None or missing required columns.
    """
    req = {"passer","receiver","x","y","pass_count"}
    if df is None or not req.issubset(df.columns):
        raise ValueError("Invalid or missing passing data")

    # filter low-frequency edges
    df = df[df.pass_count >= min_pass]
    if df.empty:
        raise ValueError(f"No passing links ≥ {min_pass} passes")

    # build graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row.passer, row.receiver, weight=row.pass_count)

    # choose layout
    if layout == "statsbomb_coords":
        # use the mean pitch coordinates from your pipeline
        pos = {r.passer: (r.x, r.y) for _, r in df.iterrows()}
        pos.update({r.receiver: (r.x, r.y) for _, r in df.iterrows()})
    elif layout == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=42)
    else:  # circular
        pos = nx.circular_layout(G)

    # node sizes by degree
    deg = dict(G.degree(weight="weight"))
    node_sizes = [deg[n] * 200 for n in G.nodes()]

    # edge widths by pass_count
    edge_widths = [d["weight"] * 0.3 for (_, _, d) in G.edges(data=True)]

    # draw on a pitch
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = pitch.draw(figsize=(8, 5))

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths, alpha=0.6, edge_color="black")
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes, alpha=0.8, node_color="steelblue")

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax,
                               font_size=8, font_color="black")

    # annotate top‐5 by betweenness
    bet = nx.betweenness_centrality(G, weight="weight")
    top5 = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]
    for player, _ in top5:
        x, y = pos[player]
        ax.text(x, y + 0.02, player, fontsize=9,
                ha="center", color="darkred", weight="bold")

    ax.set_title("Passing Network", pad=12)
    return fig