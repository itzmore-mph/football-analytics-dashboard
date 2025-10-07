# src/ui.py
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import streamlit as st

from .fetch_statsbomb import (
    fetch_competitions,
    fetch_matches,
    fetch_events,
)


def _run_py(script: Path) -> None:
    """
    Run a Python script and surface stdout/stderr nicely in Streamlit.
    Forces UTF-8 on Windows to avoid cp1252 errors when printing unicode.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # important on Windows

    res = subprocess.run(  # noqa: S603,S607 (local script execution)
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if res.returncode != 0:
        st.error(res.stderr or "Unknown error")
    elif res.stdout:
        st.code(res.stdout.strip(), language="bash")


def _comp_label(r: pd.Series) -> str:
    return (
        f"{r.competition_name} — {r.season_name} "
        f"({int(r.competition_id)}/{int(r.season_id)})"
    )


def _match_label(r: pd.Series) -> str:
    home = r["home_team"]["home_team_name"]
    away = r["away_team"]["away_team_name"]
    return f"{r['match_date']}  {home} vs {away}  (id={int(r['match_id'])})"


def _read_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


def render_dashboard(root: Path) -> None:
    st.title("Football Analytics Dashboard")

    with st.expander("Match wählen (Open Data) & Pipeline neu bauen", expanded=False):
        # 1) Competitions laden
        fetch_competitions()
        comps_path = root / "data" / "competitions.json"
        comps = _read_json(comps_path)

        if comps.empty:
            st.warning("Konnte competitions.json nicht laden.")
            return

        comps = comps.sort_values(["competition_name", "season_name"])
        comp_labels = comps.apply(_comp_label, axis=1).tolist()
        comp_choice = st.selectbox("Competition / Season", comp_labels)

        idx = comp_labels.index(comp_choice)
        sel = comps.iloc[idx]

        # 2) Matches laden
        comp_id = int(sel.competition_id)
        season_id = int(sel.season_id)
        fetch_matches(comp_id, season_id)

        matches_path = root / "data" / "matches" / str(comp_id) / f"{season_id}.json"
        matches = _read_json(matches_path)
        if matches.empty:
            st.warning("Konnte Matches nicht laden.")
            return

        matches["label"] = matches.apply(_match_label, axis=1)
        match_choice = st.selectbox("Match", matches["label"].tolist())
        match_id = int(match_choice.split("id=")[-1].rstrip(")"))

        if st.button("Update Dashboard für dieses Match", use_container_width=True):
            with st.spinner(f"Lade Events & baue Pipeline für match_id={match_id} ..."):
                # Events holen/erneuern
                fetch_events(match_id, force=True)

                # Shots & Passes für diese ID erzeugen
                # relative Imports, weil ui.py im Paket src liegt
                from .fetch_shots_data import extract_shot_data
                from .fetch_passing_data import main as fetch_passes

                extract_shot_data(match_id=match_id)
                fetch_passes(match_id=match_id)

                # Preprocess + Train
                _run_py(root / "src" / "preprocess_xG.py")
                _run_py(root / "src" / "train_xG_model.py")

            st.success("Pipeline aktualisiert")
            st.rerun()
