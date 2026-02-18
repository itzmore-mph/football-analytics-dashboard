from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from .config import settings
from .open_data import events
from .utils_io import save_csv


@dataclass
class NetworkResult:
    nodes: pd.DataFrame
    edges: pd.DataFrame


def _extract_passes(ev: pd.DataFrame) -> pd.DataFrame:
    df = ev[ev["type.name"] == "Pass"].copy()
    # locations
    loc = df["location"].apply(
        lambda v: v if isinstance(v, list) else [None, None]
    )
    df["x"] = loc.apply(
        lambda location: (
            float(location[0]) if location and location[0] is not None
            else None
            # noqa: E501
        )
    )
    df["y"] = loc.apply(
        lambda location: (
            float(location[1]) if location and location[1] is not None
            else None
        )
    )
    end = df["pass.end_location"].apply(
        lambda v: v if isinstance(v, list) else [None, None]
    )
    df["x_end"] = end.apply(
        lambda location: (
            float(location[0]) if location and location[0] is not None
            else None
            # noqa: E501
        )
    )
    df["y_end"] = end.apply(
        lambda location: (
            float(location[1]) if location and location[1] is not None
            else None
        )
    )
    # receiver
    df["receiver"] = df["pass.recipient.name"].fillna("")
    df = df.dropna(subset=["x", "y", "x_end", "y_end"])  # basic sanity
    return df


def build_team_network(
    match_id: int, team_name: str, min_edge: int = 2
) -> NetworkResult:
    ev = events(match_id)
    df = _extract_passes(ev)
    df = df[df["team.name"] == team_name].copy()

    # nodes: average positions per player (passes & receptions)
    starts = (
        df.groupby("player.name")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            touches=("player.name", "count"),
        )
        .reset_index()
    )
    recvs = (
        df.groupby("receiver")
        .agg(
            x_recv=("x_end", "mean"),
            y_recv=("y_end", "mean"),
            received=("receiver", "count"),
        )
        .reset_index()
    )
    nodes = starts.merge(
        recvs,
        left_on="player.name",
        right_on="receiver",
        how="outer",
    )
    nodes["x_mean"] = nodes[["x", "x_recv"]].mean(axis=1)
    nodes["y_mean"] = nodes[["y", "y_recv"]].mean(axis=1)
    nodes["touches"] = nodes[["touches", "received"]].fillna(0).sum(axis=1)
    nodes = nodes.rename(columns={"player.name": "player"})
    # Guarantee a human-readable label column for UI
    nodes["player_name"] = nodes["player"].astype(str)
    nodes = nodes[
        ["player", "player_name", "x_mean", "y_mean", "touches"]
    ].fillna(0)

    # edges: completed passes player -> receiver
    # (completed when outcome is NaN in SB data)
    comp = df[df["pass.outcome.name"].isna()].copy()
    edges = (
        comp.groupby(["player.name", "receiver"])
        .size()
        .reset_index(name="count")
        .rename(columns={"player.name": "source", "receiver": "target"})
    )
    edges = edges[edges["count"] >= max(1, int(min_edge))]
    edges = edges.reset_index(drop=True)

    # OPTIONAL graph metrics: degree centrality
    G = nx.DiGraph()
    for _, n in nodes.iterrows():
        G.add_node(n["player"])
    for _, e in edges.iterrows():
        G.add_edge(e["source"], e["target"], weight=int(e["count"]))
    deg = nx.degree_centrality(G)
    nodes["centrality"] = nodes["player"].map(deg).fillna(0.0)

    return NetworkResult(nodes=nodes, edges=edges)


def build_and_save_passing_events(match_ids: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mid in match_ids:
        ev = events(mid)
        df = _extract_passes(ev)
        df["match_id"] = mid
        frames.append(df)
    if not frames:
        out = pd.DataFrame(
            columns=[
                "match_id",
                "player.name",
                "receiver",
                "x",
                "y",
                "x_end",
                "y_end",
            ]
        )
        save_csv(out, settings.passing_events_csv)
        return out

    out_new = pd.concat(frames, ignore_index=True)
    csv_path: Path = settings.passing_events_csv
    if csv_path.exists():
        try:
            out_old = pd.read_csv(csv_path, low_memory=False, dtype_backend="numpy_nullable")
            out = pd.concat([out_old, out_new], ignore_index=True)
            subset = [
                c
                for c in [
                    "match_id",
                    "player.name",
                    "receiver",
                    "x",
                    "y",
                    "x_end",
                    "y_end",
                ]
                if c in out.columns
            ]
            out = out.drop_duplicates(subset=subset)
        except Exception:
            out = out_new
    else:
        out = out_new

    save_csv(out, csv_path)
    return out
