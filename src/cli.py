"""Typer command line interface for the analytics pipeline."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import pandas as pd
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
    demo: Annotated[
        bool, typer.Option("--demo", help="Fetch a small demo subset")
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Fetch many matches (internet required)"),
    ] = False,
) -> None:
    """Download match ids and cache passing events."""

    if demo and full:
        raise typer.BadParameter("Choose either --demo or --full")

    if demo:
        mids = collect_demo_matches()
    else:
        mids = collect_full_matches(limit=100 if not full else None)

    build_and_save_passing_events(mids)
    typer.echo(json.dumps({"matches": mids}, indent=2))


@app.command()
def preprocess() -> None:
    """Build the processed shots CSV from cached passing events."""

    if not settings.passing_events_csv.exists():
        typer.secho(
            "Run `python -m src.cli fetch --demo` before preprocessing.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    pe = pd.read_csv(settings.passing_events_csv)
    mids = sorted(pe["match_id"].unique().tolist())
    build_processed_shots(mids)
    typer.echo(f"Processed shots saved to {settings.processed_shots_csv}")


@app.command(name="train")
def train_cmd(
    model: Annotated[
        Literal["lr", "xgb"],
        typer.Option("--model", help="Choose between 'lr' and 'xgb'"),
    ] = "xgb",
    calibration: Annotated[
        Literal["isotonic", "platt", "sigmoid"],
        typer.Option(
            "--calibration",
            help="Calibration method: isotonic|platt"),
    ] = "isotonic",
) -> None:
    try:
        metrics = train(model, calibration)
    except FileNotFoundError as exc:  # bubble up as CLI-friendly error
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def evaluate() -> None:
    """Evaluate the trained model and export calibration plots."""

    try:
        out = evaluate_and_plot()
    except FileNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(out, indent=2))


@app.command()
def build_passing(
    match_id: Annotated[int, typer.Argument(help="StatsBomb match id")]
) -> None:
    """Build a passing network for a single match and team."""

    # Lazy imports to keep CLI startup fast
    from .open_data import events
    from .passing_network import build_team_network

    ev = events(match_id)
    teams = ev["team.name"].dropna().unique().tolist()
    results: dict[str, dict[str, int]] = {}
    for team in teams:
        res = build_team_network(match_id, team)
        results[team] = {"nodes": len(res.nodes), "edges": len(res.edges)}
    typer.echo(json.dumps(results, indent=2))


@app.command()
def demo() -> None:
    """Run the full demo pipeline end-to-end."""

    mids = collect_demo_matches()
    build_and_save_passing_events(mids)
    preprocess()
    train_cmd()
    evaluate()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
