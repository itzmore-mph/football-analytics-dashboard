from __future__ import annotations
import json
import typer

from .config import settings
from .evaluate_xg import evaluate_and_plot
from .open_data import collect_demo_matches, collect_full_matches
from .passing_network import build_and_save_passing_events
from .preprocess_shots import build_processed_shots
from .train_xg_model import train

app = typer.Typer(add_completion=False, help="Football analytics CLI")


@app.command()
def fetch(
    demo: bool = typer.Option(
        False, "--demo", help="Fetch a small demo subset"
    ),
    full: bool = typer.Option(
        False, "--full", help="Fetch many matches (internet required)"
    ),
):
    if demo and full:
        raise typer.BadParameter("Choose either --demo or --full")
    if demo:
        mids = collect_demo_matches()
    else:
        mids = collect_full_matches(limit=100 if not full else None)
    # Also persist passing events to CSV for dashboard speed
    build_and_save_passing_events(mids)
    typer.echo(json.dumps({"matches": mids}, indent=2))


@app.command()
def preprocess():
    # Use the match ids from last fetch (derived from passing_events)
    import pandas as pd

    pe = pd.read_csv(settings.passing_events_csv)
    mids = sorted(pe["match_id"].unique().tolist())
    build_processed_shots(mids)
    typer.echo(f"Processed shots saved to {settings.processed_shots_csv}")


def train_cmd(
    model: str = typer.Option("xgb", "--model", help="lr|xgb"),
    calibration: str = typer.Option(
        "isotonic",
        "--calibration",
        help="isotonic|platt"),
        ):
    # If called programmatically (e.g. demo()), coerce OptionInfo -> defaults
    if not isinstance(model, str):
        model = "xgb"
    if not isinstance(calibration, str):
        calibration = "isotonic"

    metrics = train(model, calibration)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def evaluate():
    out = evaluate_and_plot()
    typer.echo(json.dumps(out, indent=2))


@app.command()
def build_passing(match_id: int):

    # find teams from events
    from .open_data import events
    from .passing_network import build_team_network

    ev = events(match_id)
    teams = ev["team.name"].dropna().unique().tolist()
    results = {}
    for t in teams:
        res = build_team_network(match_id, t)
        results[t] = {"nodes": len(res.nodes), "edges": len(res.edges)}
    typer.echo(json.dumps(results, indent=2))


@app.command()
def demo():
    mids = collect_demo_matches()
    build_and_save_passing_events(mids)
    preprocess()
    # safe call — works both via CLI and programmatically
    train_cmd()
    evaluate()


if __name__ == "__main__":
    app()
