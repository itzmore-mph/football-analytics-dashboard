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


def load_data(path: Path = DATA_PATH) -> pd.DataFrame | None:
    for candidate in _candidate_paths(path):
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate)
        required = {"passer", "receiver", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            print(f"Error: Passing data is missing required columns: {missing}")
            return None
        df = df.dropna(subset=["passer", "receiver", "x", "y"]).copy()
        df = df.query("0 <= x <= 120 and 0 <= y <= 80").copy()
        print(f"Loaded passing data from {candidate}")
        return df
    print(f"Error: Data file {path} not found.")
    return None


def _mean_positions(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    has_end = {"end_x", "end_y"}.issubset(df.columns)
    start_x_col = "start_x" if "start_x" in df.columns else "x"
    start_y_col = "start_y" if "start_y" in df.columns else "y"
    start_pos = df.groupby("passer")[[start_x_col, start_y_col]].mean()
    start_pos.columns = ["sx", "sy"]
    if has_end:
        recv_pos = df.groupby("receiver")[["end_x", "end_y"]].mean()
        recv_pos.columns = ["rx", "ry"]
        players = pd.Index(sorted(set(df["passer"]) | set(df["receiver"])))
        pos = pd.DataFrame(index=players).join(start_pos, how="left").join(recv_pos, how="left")
        pos["x"] = pos[["sx", "rx"]].mean(axis=1, skipna=True)
        pos["y"] = pos[["sy", "ry"]].mean(axis=1, skipna=True)
    else:
        players = pd.Index(sorted(set(df["passer"]) | set(df["receiver"])))
        pos = pd.DataFrame(index=players).join(start_pos, how="left")
        pos["x"] = pos["sx"]
        pos["y"] = pos["sy"]
    pos["x"] = pos["x"].fillna(60.0)
    pos["y"] = pos["y"].fillna(40.0)
    return {p: (float(pos.at[p, "x"]), float(pos.at[p, "y"])) for p in pos.index}


def create_passing_network(df: pd.DataFrame, *, same_team_only: bool = True, min_passes: int = 2) -> nx.DiGraph:
    team_p_col = "team_passer" if "team_passer" in df.columns else ("team" if "team" in df.columns else None)
    team_r_col = "team_receiver" if "team_receiver" in df.columns else ("team" if "team" in df.columns else None)
    if same_team_only and team_p_col and team_r_col:
        df = df[df[team_p_col] == df[team_r_col]].copy()

    edges = (
        df.groupby(["passer", "receiver"])
        .size()
        .reset_index(name="weight")
        .query("receiver == receiver")
    )
    if min_passes > 1:
        edges = edges[edges["weight"] >= min_passes]

    G = nx.DiGraph()
    positions = _mean_positions(df)
    for player, (px, py) in positions.items():
        G.add_node(player, pos=(px, py))

    for _, row in edges.iterrows():
        passer = row["passer"]
        receiver = row["receiver"]
        w = float(row["weight"])
        if passer == receiver:
            continue
        if not G.has_node(passer) or not G.has_node(receiver):
            continue
        G.add_edge(passer, receiver, weight=w)

    return G


def plot_passing_network(
    G: nx.DiGraph,
    *,
    title: str = "Passing Network",
    edge_scale: float = 0.25,
    node_scale: float = 2200.0,
    figsize: tuple[float, float] = (8, 5)   # <— smaller
):
    pitch = Pitch(pitch_type="statsbomb", line_color="black")
    fig, ax = pitch.draw(figsize=figsize)    # <— use new size
    pos = nx.get_node_attributes(G, "pos")

    centrality = (
        nx.betweenness_centrality(G, weight="weight", normalized=True)
        if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}
    )

    node_sizes = [max(centrality.get(n, 0.0), 0.01) * node_scale for n in G.nodes()]
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, alpha=0.8)

    widths = [0.4 + edge_scale * float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.6, arrows=False)

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
    print(f"Saved network figure → {out_path}")


if __name__ == "__main__":
    main()
