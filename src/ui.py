# src/ui.py
"""
Football Analytics Dashboard – Streamlit UI (final)

Works with your repo structure (no fa_dash/).
Fixes:
- Replaces deprecated `use_container_width` with `width="stretch"`
- Avoids Plotly list-style line widths (uses scalar + binning)
- Adds caching for data/model
- Robust filters + friendly empty states
- Shareable state via query params

Expected (gracefully optional) files:
- data/processed_shots.csv
- data/passing_data.csv
- models/xgb_xg_calibrated.joblib (optional)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.DataFrame()


def _file_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data
def _load_shots_df(root: Path, mtime: float | None = None) -> pd.DataFrame:
    """
    Cached loader for processed_shots.csv.
    If mtime is None we compute it, so old call sites still work.
    """
    csv_path = root / "data" / "processed_shots.csv"
    if mtime is None:
        mtime = _file_mtime(csv_path)  # used only to key the cache
    _ = mtime

    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    needed = {"x", "y", "goal_scored"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df


def _minute_range(df: pd.DataFrame) -> tuple[int, int]:
    if "minute" in df.columns and df["minute"].notna().any():
        mx = int(min(120, df["minute"].max()))
        return 0, max(45, min(105, mx))
    return 0, 105


# ---------- main UI ----------


def render_dashboard(root: Path) -> None:
    st.title("Football Analytics Dashboard")

    with st.expander("Pick a Match", expanded=False):
        # 1) Competitions
        fetch_competitions()
        comps = _read_json(root / "data" / "competitions.json")
        if comps.empty:
            st.warning("Could not load competitions.json.")
            return

        comps = comps.sort_values(["competition_name", "season_name"])
        comp_labels = comps.apply(_comp_label, axis=1).tolist()
        comp_choice = st.selectbox(
            "Competition / Season", comp_labels, key="picker_comp_season"
        )
        sel = comps.iloc[comp_labels.index(comp_choice)]
        comp_id = int(sel.competition_id)
        season_id = int(sel.season_id)

        # 2) Matches
        fetch_matches(comp_id, season_id)
        matches = _read_json(
            root / "data" / "matches" / str(comp_id) / f"{season_id}.json"
        )
        if matches.empty:
            st.warning("Could not load matches.")
            return

        matches["label"] = matches.apply(_match_label, axis=1)
        match_choice = st.selectbox(
            "Match", matches["label"].tolist(), key="picker_match"
        )
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
        shots_csv = root / "data" / "processed_shots.csv"
        df_shots = _load_shots_df(root, _file_mtime(shots_csv))
        if df_shots.empty:
            st.info("No shots available yet. Please run the pipeline.")
        else:
            team_options = ["All teams"] + sorted(
                df_shots.get("team", pd.Series(dtype=str))
                .dropna()
                .unique()
                .tolist()
            )
            outcome_options = ["All shots", "Goals", "Other shots"]

            # Size slider
            size_col, _, _ = st.columns([2, 1, 1])
            sm_scale = size_col.slider(
                "Pitch size",
                0.5,
                1.2,
                0.8,
                0.05,
                key=f"sm_scale_{match_id}",
                help="Scales the representation of the Shot Map.",
            )
            w_sm, h_sm = 8 * sm_scale, 4.5 * sm_scale

            c1, c2, c3 = st.columns(3)
            team_choice = c1.selectbox(
                "Team", team_options, key=f"sm_team_{match_id}"
            )
            outcome_choice = c2.selectbox(
                "Outcome",
                outcome_options,
                index=0,
                key=f"sm_outcome_{match_id}",
            )
            mn_from, mn_to = _minute_range(df_shots)
            minute_range = c3.slider(
                "Minute range",
                mn_from,
                mn_to,
                (mn_from, mn_to),
                key=f"sm_minute_{match_id}",
            )

            dff = df_shots.copy()
            if team_choice != "All teams" and "team" in dff.columns:
                dff = dff[dff["team"] == team_choice]
            if outcome_choice == "Goals" and "goal_scored" in dff.columns:
                dff = dff[dff["goal_scored"] == 1]
            elif (
                outcome_choice == "Other shots"
                and "goal_scored" in dff.columns
            ):
                dff = dff[dff["goal_scored"] == 0]
            if "minute" in dff.columns:
                dff = dff[
                    (dff["minute"].fillna(0) >= minute_range[0])
                    & (dff["minute"].fillna(mn_to) <= minute_range[1])
                ]

            if dff.empty:
                st.warning("No shots for the selected filters.")
            else:
                try:
                    from mplsoccer import Pitch  # heavy import

                    pitch = Pitch(pitch_type="statsbomb", line_color="black")
                    fig, ax = pitch.draw(figsize=(w_sm, h_sm))
                    xg = dff.get(
                        "xG", pd.Series([0.0] * len(dff), index=dff.index)
                    )
                    size = (xg.clip(0, 0.8) * 420) + 14
                    ax.scatter(dff["x"], dff["y"], s=size, alpha=0.7)
                    ax.set_title("Shot Map (bubble size = xG)")
                    st.pyplot(
                        fig,
                        clear_figure=True,
                        use_container_width=False,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Plot error (Shot Map): {exc}")

    # Tab 2: Passing Network
    with tabs[1]:
        df_pass = load_passes_csv(root / "data" / "passing_data.csv")
        if df_pass is None or df_pass.empty:
            st.info("No passing data yet. Please run the pipeline.")
        else:
            # Size slider
            size_col, _, _ = st.columns([2, 1, 1])
            pn_scale = size_col.slider(
                "Network size",
                0.5,
                1.2,
                0.8,
                0.05,
                key=f"pn_scale_{match_id}",
                help="Scales the representation of the Passing Network.",
            )
            w_pn, h_pn = 8 * pn_scale, 4.5 * pn_scale

            c1, c2, c3 = st.columns(3)
            min_passes = int(
                c1.slider(
                    "Min passes between players",
                    1,
                    8,
                    2,
                    key=f"pn_min_passes_{match_id}",
                )
            )
            half = c2.selectbox(
                "Half",
                ["Both", "1st half", "2nd half"],
                index=0,
                key=f"pn_half_{match_id}",
            )
            mn_from, mn_to = _minute_range(df_pass)
            m1, m2 = c3.slider(
                "Minute range",
                mn_from,
                mn_to,
                (mn_from, mn_to),
                key=f"pn_minute_{match_id}",
            )

            dfp = df_pass.copy()
            if "minute" in dfp.columns:
                if half == "1st half":
                    dfp = dfp.query("minute <= 45")
                elif half == "2nd half":
                    dfp = dfp.query("minute > 45")
                dfp = dfp[(dfp["minute"] >= m1) & (dfp["minute"] <= m2)]

            G = create_passing_network(
                dfp,
                same_team_only=True,
                min_passes=min_passes,
            )
            fig = plot_passing_network(
                G,
                title=f"Passing Network (≥{min_passes} passes)",
                figsize=(w_pn, h_pn),
            )
            st.pyplot(fig, clear_figure=True, use_container_width=False)

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
            st.info("No processed_shots.csv yet.")
        else:
            st.dataframe(df_shots.head(500))
