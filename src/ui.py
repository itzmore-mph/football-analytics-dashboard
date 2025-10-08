# src/ui.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from .fetch_statsbomb import fetch_competitions, fetch_matches, fetch_events
from .passing_network import (
    load_data as load_passes_csv,
    create_passing_network,
    plot_passing_network,
)


# ---------- helpers ----------

def _run_py(script: Path) -> None:
    """Run a Python script and show stdout/stderr in Streamlit."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    res = subprocess.run(  # noqa: S603,S607 (local script)
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


def _comp_label(row: pd.Series) -> str:
    return (
        f"{row.competition_name} — {row.season_name} "
        f"({int(row.competition_id)}/{int(row.season_id)})"
    )


def _match_label(row: pd.Series) -> str:
    home = row["home_team"]["home_team_name"]
    away = row["away_team"]["away_team_name"]
    return f"{row['match_date']} {home} vs {away} (id={int(row['match_id'])})"


def _read_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


# add helper
def _file_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data
def _load_shots_df(path: Path, mtime: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    needed = {"x", "y", "goal_scored"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df


def _minute_range(df: pd.DataFrame) -> Tuple[int, int]:
    if "minute" in df.columns and df["minute"].notna().any():
        max_minute = int(min(120, df["minute"].max()))
        return 0, max(45, min(105, max_minute))
    return 0, 105


# ---------- main UI ----------

def render_dashboard(root: Path) -> None:
    st.title("Football Analytics Dashboard")

    with st.expander("Pick a Match", expanded=False):
        # 1) Competitions
        fetch_competitions()
        comps = _read_json(root / "data" / "competitions.json")
        if comps.empty:
            st.warning("Konnte competitions.json nicht laden.")
            return

        comps = comps.sort_values(["competition_name", "season_name"])
        comp_labels = comps.apply(_comp_label, axis=1).tolist()
        comp_choice = st.selectbox("Competition / Season", comp_labels)
        sel = comps.iloc[comp_labels.index(comp_choice)]
        comp_id = int(sel.competition_id)
        season_id = int(sel.season_id)

        # 2) Matches
        fetch_matches(comp_id, season_id)
        matches = _read_json(
            root / "data" / "matches" / str(comp_id) / f"{season_id}.json")
        if matches.empty:
            st.warning("Konnte Matches nicht laden.")
            return

        matches["label"] = matches.apply(_match_label, axis=1)
        match_choice = st.selectbox("Match", matches["label"].tolist())
        match_id = int(match_choice.split("id=")[-1].rstrip(")"))

        # 3) Trigger pipeline
        if st.button("Update Dashboard", use_container_width=True):
            with st.spinner(f"Load events for match_id={match_id} ..."):
                fetch_events(match_id, force=True)

                # Late imports
                from .fetch_shots_data import extract_shot_data
                from .fetch_passing_data import main as fetch_passes

                extract_shot_data(match_id=match_id)
                fetch_passes(match_id=match_id)

                _run_py(root / "src" / "preprocess_xG.py")
                _run_py(root / "src" / "train_xG_model.py")

            st.success("Pipeline updated.")
            st.cache_data.clear()
            st.rerun()

    # --------- TABS ---------
    tabs = st.tabs(["Shot Map (xG)", "Passing Network", "Model", "Data"])

    # Tab 1: Shot Map
    with tabs[0]:
        shots_path = root / "data" / "processed_shots.csv"
        df_shots = _load_shots_df(shots_path, _file_mtime(shots_path))
        if df_shots.empty:
            st.info("No shots available yet. Please run the pipeline.")
        else:
            try:
                from mplsoccer import Pitch  # heavy import; inside try

                pitch = Pitch(pitch_type="statsbomb", line_color="black")
                fig, ax = pitch.draw(figsize=(10, 6))
                size = (df_shots.get("xG", pd.Series([0]*len(df_shots)))
                        .clip(0, 0.8) * 500) + 18
                ax.scatter(df_shots["x"], df_shots["y"], s=size, alpha=0.65)
                ax.set_title("Shot Map (Bubble size = xG)")
                st.pyplot(fig, clear_figure=True)
            except Exception as exc:  # pragma: no cover
                st.error(f"Plot-Fehler (Shot Map): {exc}")

    # Tab 2: Passing Network
    with tabs[1]:
        df_pass = load_passes_csv()
        if df_pass is None or df_pass.empty:
            st.info("Noch keine Passing-Daten. Bitte Pipeline ausführen.")
        else:
            c1, c2, c3 = st.columns(3)
            min_passes = int(
                c1.slider("Min passes zwischen Spielern", 1, 8, 2)
                )
            half = c2.selectbox(
                "Half", ["Both", "1st half", "2nd half"], index=0
                )
            mn_from, mn_to = _minute_range(df_pass)
            m1, m2 = c3.slider(
                "Minute range", mn_from, mn_to, (mn_from, mn_to)
                )
            dfp = df_pass.copy()
            if "minute" in dfp.columns:
                if half == "1st half":
                    dfp = dfp.query("minute <= 45")
                elif half == "2nd half":
                    dfp = dfp.query("minute > 45")
                dfp = dfp[(dfp["minute"] >= m1) & (dfp["minute"] <= m2)]
            G = create_passing_network(
                dfp, same_team_only=True, min_passes=min_passes
                )
            fig = plot_passing_network(
                G, title=f"Passing Network (≥{min_passes} passes)"
                )
            st.pyplot(fig, clear_figure=True)

    # Tab 3: Model
    with tabs[2]:
        model_dir = root / "models"
        calib_png = model_dir / "xg_calibration.png"
        buckets_csv = model_dir / "xg_buckets.csv"
        feats_txt = model_dir / "features.txt"
        metrics_txt = model_dir / "xgboost_metrics.txt"

        cols = st.columns(2)
        if calib_png.exists():
            cols[0].image(str(calib_png), caption="Calibration curve")
        if buckets_csv.exists():
            st.subheader("xG Buckets")
            st.dataframe(pd.read_csv(buckets_csv))
        if feats_txt.exists():
            st.subheader("Features (train/infer)")
            st.code(feats_txt.read_text(encoding="utf-8"))
        if metrics_txt.exists():
            st.subheader("Metrics")
            st.code(metrics_txt.read_text(encoding="utf-8"))

    # Tab 4: Data
    with tabs[3]:
        df_shots = _load_shots_df(root)
        if df_shots.empty:
            st.info("Noch keine processed_shots.csv.")
        else:
            st.dataframe(df_shots.head(500))
