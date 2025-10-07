# src/ui.py
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, Tuple

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
def _load_shots_df(root: Path, mtime: float) -> pd.DataFrame:
    """Cached loader keyed by file mtime (mtime arg ensures cache invalidation)."""
    _ = mtime  # used only for cache keying
    p = root / "data" / "processed_shots.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    needed = {"x", "y", "goal_scored"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df


def _safe_fetch(
    callable_: Callable,
    *,
    description: str,
    args: Iterable | None = None,
    kwargs: Mapping | None = None,
) -> None:
    try:
        callable_(*([] if args is None else list(args)), **({} if kwargs is None else dict(kwargs)))
    except Exception as exc:  # pragma: no cover - UI level safeguard
        st.error(f"{description} failed: {exc}")
        st.stop()


def _format_timestamp(ts: float) -> str:
    if ts <= 0:
        return "unknown"
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


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
        _safe_fetch(fetch_competitions, description="Fetching competitions")
        comps = _read_json(root / "data" / "competitions.json")
        if comps.empty:
            st.warning("Not able to load competitions.json.")
            return

        comps = comps.sort_values(["competition_name", "season_name"])
        comp_labels = comps.apply(_comp_label, axis=1).tolist()
        comp_choice = st.selectbox("Competition / Season", comp_labels, key="picker_comp_season")
        sel = comps.iloc[comp_labels.index(comp_choice)]
        comp_id = int(sel.competition_id)
        season_id = int(sel.season_id)

        # 2) Matches
        _safe_fetch(
            fetch_matches,
            description="Fetching matches",
            args=(comp_id, season_id),
        )
        matches = _read_json(root / "data" / "matches" / str(comp_id) / f"{season_id}.json")
        if matches.empty:
            st.warning("Konnte Matches nicht laden.")
            return

        matches["label"] = matches.apply(_match_label, axis=1)
        match_choice = st.selectbox("Match", matches["label"].tolist(), key="picker_match")
        match_id = int(match_choice.split("id=")[-1].rstrip(")"))

        # 3) Trigger pipeline
        if st.button("Update Dashboard", use_container_width=True):
            with st.spinner(f"Load events for match_id={match_id} ..."):
                _safe_fetch(
                    fetch_events,
                    description="Fetching events",
                    args=(match_id,),
                    kwargs={"force": True},
                )

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
        shots_mtime = _file_mtime(shots_path)
        df_shots = _load_shots_df(root, shots_mtime)
        if df_shots.empty:
            st.info("No shots available yet. Please run the pipeline.")
        else:
            df_shots = df_shots.copy()
            if "goal_scored" in df_shots.columns:
                df_shots["goal_scored"] = df_shots["goal_scored"].fillna(0).astype(int)
            else:
                df_shots["goal_scored"] = 0

            team_options = ["All teams"] + sorted(
                [t for t in df_shots.get("team", pd.Series(dtype=str)).dropna().unique()]
            )
            outcome_options = ["All shots", "Goals", "Other shots"]

            filters = st.columns(3)
            team_choice = filters[0].selectbox("Team", team_options, key=f"sm_team_{match_id}")
            outcome_choice = filters[1].selectbox(
                "Outcome", outcome_options, index=0, key=f"sm_outcome_{match_id}"
            )
            mn_from, mn_to = _minute_range(df_shots)
            minute_range = filters[2].slider(
                "Minute range", mn_from, mn_to, (mn_from, mn_to), key=f"sm_minute_{match_id}"
            )

            df_filtered = df_shots.copy()
            if team_choice != "All teams" and "team" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["team"] == team_choice]
            if outcome_choice == "Goals" and "goal_scored" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["goal_scored"] == 1]
            elif outcome_choice == "Other shots" and "goal_scored" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["goal_scored"] == 0]
            if "minute" in df_filtered.columns:
                df_filtered = df_filtered[
                    (df_filtered["minute"].fillna(0) >= minute_range[0])
                    & (df_filtered["minute"].fillna(mn_to) <= minute_range[1])
                ]

            if df_filtered.empty:
                st.warning("No shots for the selected filters.")
                st.caption(f"Data last updated: {_format_timestamp(shots_mtime)}")
            else:
                total_shots = len(df_filtered)
                goal_series = (
                    df_filtered["goal_scored"]
                    if "goal_scored" in df_filtered
                    else pd.Series([0] * len(df_filtered))
                )
                goals = int(goal_series.sum())
                xg_series = (
                    df_filtered["xG"]
                    if "xG" in df_filtered
                    else pd.Series([0.0] * len(df_filtered))
                )
                xg_total = float(xg_series.sum())
                avg_xg = xg_total / total_shots if total_shots > 0 else 0.0
                conversion = goals / total_shots if total_shots else 0.0

                metric_cols = st.columns(5)
                metric_cols[0].metric("Shots", f"{total_shots}")
                metric_cols[1].metric("Goals", f"{goals}")
                metric_cols[2].metric("Total xG", f"{xg_total:.2f}")
                metric_cols[3].metric("xG / shot", f"{avg_xg:.2f}")
                metric_cols[4].metric("Conversion", f"{conversion:.1%}")
                st.caption(f"Data last updated: {_format_timestamp(shots_mtime)}")

                try:
                    from mplsoccer import Pitch  # heavy import; inside try

                    pitch = Pitch(pitch_type="statsbomb", line_color="black")
                    fig, ax = pitch.draw(figsize=(10, 6))
                    size = (xg_series.clip(0, 0.8) * 500) + 18
                    goal_colors = (
                        df_filtered["goal_scored"]
                        if "goal_scored" in df_filtered
                        else pd.Series([0] * len(df_filtered))
                    )
                    colors = ["#e63946" if bool(val) else "#1d3557" for val in goal_colors]
                    ax.scatter(df_filtered["x"], df_filtered["y"], s=size, alpha=0.7, c=colors)
                    ax.set_title("Shot Map (bubble size = xG, red = goal)")
                    st.pyplot(fig, clear_figure=True)
                except Exception as exc:  # pragma: no cover
                    st.error(f"Plot-Fehler (Shot Map): {exc}")

                if {"player", "xG"}.issubset(df_filtered.columns):
                    st.subheader("Top shooters by xG")
                    top_df = df_filtered.copy()
                    if "goal_scored" not in top_df.columns:
                        top_df["goal_scored"] = 0
                    top_shooters = (
                        top_df.groupby("player")
                        .agg(
                            shots=("player", "size"),
                            goals=("goal_scored", "sum"),
                            xg=("xG", "sum"),
                        )
                        .sort_values("xg", ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    top_shooters["goals"] = top_shooters["goals"].astype(int)
                    top_shooters["shots"] = top_shooters["shots"].astype(int)
                    top_shooters["xg"] = top_shooters["xg"].round(2)
                    st.dataframe(top_shooters, use_container_width=True)

    # Tab 2: Passing Network
    with tabs[1]:
        passes_path = root / "data" / "passing_data.csv"
        fallback_passes = passes_path.with_name("processed_passing_data.csv")
        df_pass = load_passes_csv(passes_path)

        if df_pass is None or df_pass.empty:
            st.info("No Passing Data loaded yet. Please execute Pipeline.")
        else:
            df_pass = df_pass.copy()
            c1, c2, c3, _ = st.columns(4)

            # --- unique, keyed widgets (no duplicates) ---
            min_passes = int(
                c1.slider(
                    "Min passes zwischen Spielern", 1, 8, 2, key=f"pn_min_passes_{match_id}"
                )
            )
            half = c2.selectbox(
                "Half", ["Both", "1st half", "2nd half"], index=0, key=f"pn_half_{match_id}"
            )

            teams_available = sorted({
                *df_pass.get("team_passer", pd.Series(dtype=str)).dropna().unique(),
                *df_pass.get("team", pd.Series(dtype=str)).dropna().unique(),
                *df_pass.get("team_receiver", pd.Series(dtype=str)).dropna().unique(),
            })
            options = ["All teams"] + (teams_available or [])
            team_filter = c3.selectbox("Team", options, index=0, key=f"pn_team_{match_id}")

            mn_from, mn_to = _minute_range(df_pass)
            m1, m2 = c3.slider(
                "Minute range", mn_from, mn_to, (mn_from, mn_to), key=f"pn_minute_{match_id}"
            )

            # --- filtering ---
            dfp = df_pass.copy()
            if "minute" in dfp.columns:
                if half == "1st half":
                    dfp = dfp.query("minute <= 45")
                elif half == "2nd half":
                    dfp = dfp.query("minute > 45")
                dfp = dfp[(dfp["minute"] >= m1) & (dfp["minute"] <= m2)]

            if team_filter != "All teams":
                team_mask = pd.Series(False, index=dfp.index)
                for col in ["team_passer", "team", "team_receiver"]:
                    if col in dfp.columns:
                        team_mask = team_mask | (dfp[col] == team_filter)
                dfp = dfp[team_mask]

            # --- build / plot ---
            G = create_passing_network(dfp, same_team_only=True, min_passes=min_passes)
            fig = plot_passing_network(G, title=f"Passing Network (≥{min_passes} passes)")
            st.pyplot(fig, clear_figure=True)
            passes_mtime = max(_file_mtime(passes_path), _file_mtime(fallback_passes))
            st.caption(f"Passing data last updated: {_format_timestamp(passes_mtime)}")

            if not dfp.empty:
                st.subheader("Most frequent passing links")
                edge_table = (
                    dfp.groupby(["passer", "receiver"])
                    .size()
                    .reset_index(name="passes")
                    .sort_values("passes", ascending=False)
                    .head(15)
                )
                st.dataframe(edge_table, use_container_width=True)
            else:
                st.info("No passes meet the current filters. Please adjust.")

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
        shots_path = root / "data" / "processed_shots.csv"
        df_shots = _load_shots_df(root, _file_mtime(shots_path))
        if df_shots.empty:
            st.info("No processed_shots.csv loaded yet.")
        else:
            st.dataframe(df_shots.head(500))
