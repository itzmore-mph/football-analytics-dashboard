# Passing Network Visualization

from __future__ import annotations

from pathlib import Path

import pandas as pd
import networkx as nx
from mplsoccer import Pitch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "passing_data.csv"


def _candidate_paths(path: Path) -> list[Path]:
    candidates = [path]
    fallback = path.with_name("processed_passing_data.csv")
    if fallback not in candidates:
        candidates.append(fallback)
    return candidates


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data(path: Path = DATA_PATH) -> pd.DataFrame | None:
    """
    Load passing CSV and return a clean DataFrame with at least:
    ['passer','receiver','x','y']
    and, if available, ['start_x','start_y','end_x','end_y','minute'].
    """
    for candidate in _candidate_paths(path):
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate)

        # Required columns
        required = {"passer", "receiver", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            print(
                f"Error: Passing data is missing required columns: {missing}"
                  )
            continue

        # Basic hygiene
        # Normalize player/team names
        for col in [
            "passer",
            "receiver",
            "team",
            "team_passer",
            "team_receiver"
                ]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Coerce numerics
        num_cols = [
            "x",
            "y",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "minute",
            "second"
        ]
        df = _coerce_numeric(df, num_cols)

        # Drop obvious invalids and limit to pitch
        df = df.dropna(subset=["passer", "receiver", "x", "y"]).copy()
        df = df.query("0 <= x <= 120 and 0 <= y <= 80").copy()

        # If end_x/y missing, synthesize from x/y so downstream code works
        if "end_x" not in df.columns:
            df["end_x"] = df["x"]
        if "end_y" not in df.columns:
            df["end_y"] = df["y"]

        print(f"Loaded passing data from {candidate}")
        return df

    print(f"Error: Data file {path} not found.")
    return None


def _mean_positions(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Player anchor positions = mean of start pos (as passer)
    and end pos (as receiver).
    Falls back to start positions
    only when end positions are unavailable.
    """
    start_x_col = "start_x" if "start_x" in df.columns else "x"
    start_y_col = "start_y" if "start_y" in df.columns else "y"

    start_pos = df.groupby(
        "passer"
        )
    [[start_x_col, start_y_col]].mean(numeric_only=True)
    start_pos.columns = ["sx", "sy"]

    has_end = {"end_x", "end_y"}.issubset(df.columns)
    players = pd.Index(sorted(set(df["passer"]) | set(df["receiver"])))

    if has_end:
        recv_pos = df.groupby(
            "receiver"
            )
        [["end_x", "end_y"]].mean(numeric_only=True)
        recv_pos.columns = ["rx", "ry"]
        pos = pd.DataFrame(
            index=players
            ).join(start_pos, how="left").join(recv_pos, how="left")
        pos["x"] = pos[["sx", "rx"]].mean(axis=1, skipna=True)
        pos["y"] = pos[["sy", "ry"]].mean(axis=1, skipna=True)
    else:
        pos = pd.DataFrame(index=players).join(start_pos, how="left")
        pos["x"] = pos["sx"]
        pos["y"] = pos["sy"]

    # Fallback to center if still NaN
    pos["x"] = pos["x"].fillna(60.0)
    pos["y"] = pos["y"].fillna(40.0)

    return {
        p: (float(pos.at[p, "x"]),
            float(pos.at[p, "y"])) for p in pos.index
        }


def create_passing_network(
    df: pd.DataFrame,
    *,
    same_team_only: bool = True,
    min_passes: int = 2,
) -> nx.DiGraph:
    """
    Build a directed graph of passes.
    """
    team_p_col = "team_passer" if "team_passer" in df.columns else ("team" if "team" in df.columns else None)
    team_r_col = "team_receiver" if "team_receiver" in df.columns else ("team" if "team" in df.columns else None)

    df_use = df.copy()
    if same_team_only and team_p_col and team_r_col:
        # Only teammate-to-teammate passes (should be all SB passes, but guard anyway)
        df_use = df_use[df_use[team_p_col] == df_use[team_r_col]].copy()

    # Edge table
    edges = (
        df_use.groupby(["passer", "receiver"], dropna=True)
        .size()
        .reset_index(name="weight")
    )
    # Remove self-passes and NaN receivers
    edges = edges[(edges["passer"] != edges["receiver"]) & edges["receiver"].notna()]

    if min_passes > 1:
        edges = edges[edges["weight"] >= min_passes]

    G = nx.DiGraph()
    positions = _mean_positions(df_use)

    # Nodes
    for player, (px, py) in positions.items():
        G.add_node(player, pos=(px, py))

    # Edges
    for _, row in edges.iterrows():
        passer = str(row["passer"])
        receiver = str(row["receiver"])
        w = float(row["weight"])
        if passer in G and receiver in G:
            G.add_edge(passer, receiver, weight=w)

    return G


def _scaled_widths(G: nx.DiGraph, base: float = 0.6, scale: float = 1.6) -> list[float]:
    """Scale edge widths relative to min/max weights for readability."""
    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    if not weights:
        return []
    w_min, w_max = min(weights), max(weights)
    if w_max == w_min:
        return [base + scale * 0.5 for _ in weights]
    return [base + scale * ((w - w_min) / (w_max - w_min)) for w in weights]


def plot_passing_network(
    G: nx.DiGraph,
    *,
    title: str = "Passing Network",
    edge_scale: float = 0.25,  # kept for API compatibility (unused in new width scaling)
    node_scale: float = 2200.0,
    figsize: tuple[float, float] = (7, 4.5),
):
    """
    Render the network on a StatsBomb pitch. Returns a matplotlib Figure.
    """
    pitch = Pitch(pitch_type="statsbomb", line_color="black")
    fig, ax = pitch.draw(figsize=figsize)
    pos = nx.get_node_attributes(G, "pos")

    # Betweenness for node size
    centrality = (
        nx.betweenness_centrality(G, weight="weight", normalized=True)
        if G.number_of_edges() > 0
        else {n: 0.0 for n in G.nodes()}
    )
    node_sizes = [max(centrality.get(n, 0.0), 0.01) * node_scale for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")

    # Edge widths scaled relative to min/max for readability
    widths = _scaled_widths(G, base=0.6, scale=1.8)
    if widths:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=widths,
            alpha=0.6,
            arrows=False,
            edge_color="black",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig


def main():
    df = load_data()
    if df is None or df.empty:
        print("Passing network not available.")
        return

    G = create_passing_network(df, same_team_only=True, min_passes=2)
    fig = plot_passing_network(G, title="Passing Network (min 2 passes)")
    out_path = DATA_PATH.with_suffix(".png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved network figure -> {out_path}")


if __name__ == "__main__":
    main()
