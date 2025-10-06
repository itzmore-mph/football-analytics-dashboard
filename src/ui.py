from .fetch_statsbomb import fetch_competitions, fetch_matches, fetch_events
import pandas as pd
import streamlit as st
from pathlib import Path
import os
import subprocess, sys

def _run_py(script: Path):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"   # <<— wichtig auf Windows
    res = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )
    if res.returncode != 0:
        st.error(res.stderr or "Unknown error")
    elif res.stdout:
        st.code(res.stdout, language="bash")

def render_dashboard(root: Path):
    st.title("Football Analytics Dashboard")

    with st.expander("Match wählen (Open Data) & Pipeline neu bauen"):
        # 1) Competitions laden
        fetch_competitions()
        comps = pd.read_json(root / "data" / "competitions.json")
        comps = comps.sort_values(["competition_name", "season_name"])
        comp = st.selectbox(
            "Competition / Season",
            comps.apply(lambda r: f"{r.competition_name} — {r.season_name} ({r.competition_id}/{r.season_id})", axis=1)
        )
        # Auswahl parsen
        sel = comps.iloc[
            [comps.apply(lambda r: f"{r.competition_name} — {r.season_name} ({r.competition_id}/{r.season_id})", axis=1).tolist().index(comp)]
        ].iloc[0]

        # 2) Matches laden
        fetch_matches(int(sel.competition_id), int(sel.season_id))
        matches_path = root / "data" / "matches" / str(int(sel.competition_id)) / f"{int(sel.season_id)}.json"
        matches = pd.read_json(matches_path)

        matches["label"] = matches.apply(
            lambda r: f"{r['match_date']}  {r['home_team']['home_team_name']} vs {r['away_team']['away_team_name']}  (id={r['match_id']})",
            axis=1
        )
        mlabel = st.selectbox("Match", matches["label"].tolist())
        match_id = int(mlabel.split("id=")[-1].rstrip(")"))

        if st.button("Update Dashboard für dieses Match"):
            with st.spinner(f"Lade Events & baue Pipeline für match_id={match_id} …"):
                fetch_events(match_id, force=True)  # Events holen/erneuern
                # Shots & Passes für diese ID erzeugen
                from .fetch_shots_data import extract_shot_data
                from .fetch_passing_data import main as fetch_passes
                extract_shot_data(match_id=match_id)
                fetch_passes(match_id=match_id)
                # Preprocess + Train
                _run_py(root / "src" / "preprocess_xG.py")
                _run_py(root / "src" / "train_xG_model.py")
            st.success("Pipeline aktualisiert")
            st.rerun()
