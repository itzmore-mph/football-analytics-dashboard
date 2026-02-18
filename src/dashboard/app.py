# src/dashboard/app.py
from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable, BEFORE any src.* imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import joblib
import pandas as pd
import streamlit as st
from mplsoccer import Pitch

from src.config import settings
from src.dashboard.components import sidebar_filters
from src.dashboard.plots import cumulative_xg_plot
from src.dashboard.theming import set_theme
from src.features_xg import FEATURE_COLUMNS

cache_data = st.cache_data
cache_resource = st.cache_resource


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit compatibility helpers (work across versions)
# ──────────────────────────────────────────────────────────────────────────────
def _df_full_width(df: pd.DataFrame, **kwargs):
    """
    Prefer new API (width='stretch'), fall back to legacy (use_container_width=True).
    """
    try:
        return st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(df, use_container_width=True, **kwargs)


def _plotly_full_width(fig, **kwargs):
    """
    Ensure Plotly charts size to container across versions.
    """
    cfg = kwargs.pop("config", {})
    try:
        return st.plotly_chart(fig, width="stretch", config=cfg, **kwargs)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, config=cfg, **kwargs)


def plot(fig) -> None:
    _plotly_full_width(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Cached lists for settings UI
# ──────────────────────────────────────────────────────────────────────────────
@cache_data(show_spinner=False)
def _list_competitions_df() -> pd.DataFrame:
    from src.open_data import competitions
    return competitions()


@cache_data(show_spinner=False)
def _list_seasons_for_comp(comp_name: str) -> list[str]:
    df = _list_competitions_df()
    if df.empty or "competition_name" not in df.columns or "season_name" not in df.columns:
        return []
    return (
        df.loc[df["competition_name"] == comp_name, "season_name"]
        .dropna()
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap helpers
# ──────────────────────────────────────────────────────────────────────────────
def _artifacts_exist() -> bool:
    return all(
        [
            settings.passing_events_csv.exists(),
            settings.processed_shots_csv.exists(),
            settings.model_path.exists(),
        ]
    )


@cache_resource(show_spinner=False)
def _build_demo_artifacts() -> dict:
    """
    One-time builder for a tiny demo dataset:
    fetch -> passing CSV -> shots CSV -> train model.

    Cached as a resource so it runs only when you click the button.
    """
    from src.open_data import collect_demo_matches
    from src.passing_network import build_and_save_passing_events
    from src.preprocess_shots import build_processed_shots
    from src.train_xg_model import train

    mids = collect_demo_matches()
    build_and_save_passing_events(mids)
    build_processed_shots(mids)
    metrics = train(kind="xgb", calibration="isotonic")
    return {"matches": mids, "metrics": metrics}


# ──────────────────────────────────────────────────────────────────────────────
# Data/model loaders
# ──────────────────────────────────────────────────────────────────────────────
@cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = (
        pd.read_csv(settings.processed_shots_csv, low_memory=False)
        if settings.processed_shots_csv.exists()
        else pd.DataFrame()
    )
    passes = (
        pd.read_csv(settings.passing_events_csv, low_memory=False)
        if settings.passing_events_csv.exists()
        else pd.DataFrame()
    )
    return shots, passes


@cache_resource(show_spinner=False)
def load_model():
    if settings.model_path.exists():
        return joblib.load(settings.model_path)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # IMPORTANT: set_page_config must be the first Streamlit command
    # Make sure set_theme() only calls st.set_page_config() and nothing else.
    set_theme()

    # Title + short description
    st.title("⚽ Football Analytics Dashboard")
    st.markdown(
        "Interactive match analytics with **xG modeling**, **shot maps**, and **passing networks**. "
        "Use the sidebar to filter by competition, season, match, and minute range."
    )

    st.caption("*Statistical analysis powered by StatsBomb Open Data*")

    # Tighter page width & padding (CSS)
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1200px;
                padding-top: 2.5rem !important;
                padding-bottom: 2rem;
                overflow: visible;
            }
            .block-container h1 {
                margin-top: 0.25rem !important;
                line-height: 1.2 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Bootstrap demo artifacts if missing
    if not _artifacts_exist():
        try:
            ctx = st.container(border=True)
        except TypeError:
            ctx = st.container()

        with ctx:
            st.info(
                "No data/model artifacts found yet.\n\n"
                "Click the button below to build a small demo dataset "
                "(fetch -> feature -> train)."
            )
            if st.button("Build demo data now"):
                with st.spinner("Building demo data..."):
                    _ = _build_demo_artifacts()

                # Clear only what we need, avoid calling non-existent clear() methods
                try:
                    load_data.clear()
                except Exception:
                    pass
                try:
                    load_model.clear()
                except Exception:
                    pass
                try:
                    st.cache_data.clear()
                    st.cache_resource.clear()
                except Exception:
                    pass

                st.success("Demo data built. Reloading...")
                st.rerun()
        st.stop()

    # Sidebar filters and compact layout toggle
    filters = sidebar_filters()

    try:
        compact = st.sidebar.toggle(
            "Compact layout",
            value=True,
            help="Smaller plots and tighter spacing",
        )
    except AttributeError:
        compact = st.sidebar.checkbox(
            "Compact layout",
            value=True,
            help="Smaller plots and tighter spacing",
        )

    shots, passes = load_data()
    model = load_model()

    # Guardrails
    if shots.empty or passes.empty or model is None:
        st.warning("Artifacts missing or empty. Please rebuild demo data in Settings.")
        st.stop()

    # Predict xG for shots
    missing = [c for c in FEATURE_COLUMNS if c not in shots.columns]
    if missing:
        st.error(f"Missing feature columns: {missing}. Rebuild demo data in Settings.")
        st.stop()

    if not hasattr(model, "predict_proba"):
        st.error("Loaded model doesn’t support predict_proba(). Re-train in demo builder.")
        st.stop()

    shots = shots.copy()
    shots["xg"] = model.predict_proba(shots[FEATURE_COLUMNS].values)[:, 1]

    # Sidebar: Competition/Season filters (based on columns in processed shots)
    comp_col = "competition_name" if "competition_name" in shots.columns else None
    season_col = "season_name" if "season_name" in shots.columns else None

    # Ensure we do not double-render "Filters" if sidebar_filters already shows it
    st.sidebar.markdown("---")
    st.sidebar.subheader("Competition / Season")

    if comp_col:
        comp_options = ["(All)"] + sorted(shots[comp_col].dropna().unique().tolist())
        selected_comp = st.sidebar.selectbox(
            "Competition",
            comp_options,
            index=0,
            help="Filter matches by competition.",
        )
    else:
        selected_comp = "(All)"

    if season_col:
        season_df = shots.copy()
        if comp_col and selected_comp != "(All)":
            season_df = season_df[season_df[comp_col] == selected_comp]
        season_options = ["(All)"] + sorted(season_df[season_col].dropna().unique().tolist())
        selected_season = st.sidebar.selectbox(
            "Season",
            season_options,
            index=0,
            help="Filter matches by season.",
        )
    else:
        selected_season = "(All)"

    # Apply comp/season filters to available matches
    shots_filtered = shots.copy()
    if comp_col and selected_comp != "(All)":
        shots_filtered = shots_filtered[shots_filtered[comp_col] == selected_comp]
    if season_col and selected_season != "(All)":
        shots_filtered = shots_filtered[shots_filtered[season_col] == selected_season]

    # Sidebar — Match selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Match Selection")

    def _match_label(mid: int) -> str:
        # Prefer fixture if available (and format nicely)
        if "fixture" in shots_filtered.columns:
            lab = (
                shots_filtered.loc[shots_filtered["match_id"] == mid, "fixture"]
                .dropna()
                .astype(str)
                .head(1)
                .tolist()
            )
            if lab:
                if "match_date" in shots_filtered.columns:
                    md = (
                        pd.to_datetime(
                            shots_filtered.loc[shots_filtered["match_id"] == mid, "match_date"],
                            errors="coerce",
                        )
                        .dropna()
                        .dt.date
                        .astype(str)
                        .head(1)
                        .tolist()
                    )
                    return f"{lab[0]} , {md[0]}" if md else lab[0]
                return lab[0]

        # Fallback: build "Team A vs. Team B"
        teams = (
            shots_filtered.loc[shots_filtered["match_id"] == mid, "team.name"]
            .dropna()
            .unique()
            .tolist()
            if "team.name" in shots_filtered.columns
            else []
        )

        if len(teams) >= 2:
            vs = f"{teams[0]} vs. {teams[1]}"
        elif len(teams) == 1:
            vs = teams[0]
        else:
            vs = f"Match {mid}"

        # Optional: add date if present
        if "match_date" in shots_filtered.columns:
            md = (
                pd.to_datetime(
                    shots_filtered.loc[shots_filtered["match_id"] == mid, "match_date"],
                    errors="coerce",
                )
                .dropna()
                .dt.date
                .astype(str)
                .head(1)
                .tolist()
            )
            if md:
                return f"{vs} , {md[0]}"

        return vs

    if "match_id" not in shots_filtered.columns or shots_filtered["match_id"].dropna().empty:
        st.sidebar.warning("No matches available for current filters.")
        st.stop()

    match_ids = sorted(shots_filtered["match_id"].dropna().astype(int).unique().tolist())
    match_id = st.sidebar.selectbox(
        "Match",
        match_ids,
        format_func=_match_label,
        help="Pick a match from the filtered set.",
    )

    teams = (
        shots_filtered.loc[shots_filtered["match_id"] == match_id, "team.name"]
        .dropna()
        .unique()
        .tolist()
        if "team.name" in shots_filtered.columns
        else []
    )
    if not teams:
        st.warning("No team information available for this match.")
        st.stop()

    team = st.sidebar.selectbox("Team (for passing network)", teams, index=0)

    # Filter shots by match and minute range
    ms = shots_filtered[shots_filtered["match_id"] == match_id].copy()
    minute_min, minute_max = filters["minute_range"]
    if "minute" in ms.columns:
        ms = ms[(ms["minute"] >= minute_min) & (ms["minute"] <= minute_max)]

    if ms.empty:
        st.info("No shots in the selected minute range.")
        st.stop()

    # Player filter
    available_players = sorted(
        ms.get("player.name", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    selected_players = st.sidebar.multiselect(
        "Players (filter shots & stats)", available_players
    )
    if selected_players and "player.name" in ms.columns:
        ms = ms[ms["player.name"].isin(selected_players)]

    # Tabs
    tab_overview, tab_xg_pitch, tab_passing, tab_stats, tab_settings = st.tabs(
        [
            "📊 Overview",
            "🎯 xG Model & Pitch",
            "🔗 Passing Network",
            "📈 Statistics",
            "⚙️ Data Fetch & Settings",
        ]
    )


    # ───────────────────────── Overview ─────────────────────────
    with tab_overview:
        st.header("Match Overview")

        left_meta = []
        if comp_col and selected_comp != "(All)":
            left_meta.append(selected_comp)
        if season_col and selected_season != "(All)":
            left_meta.append(str(selected_season))
        meta_str = " • ".join(left_meta) if left_meta else "All competitions/seasons"

        # Teams/date
        if {"home_team", "away_team"} <= set(ms.columns):
            h_team = ms["home_team"].dropna().astype(str).head(1).tolist()
            a_team = ms["away_team"].dropna().astype(str).head(1).tolist()
            h_team = h_team[0] if h_team else None
            a_team = a_team[0] if a_team else None
        else:
            teams_order = ms["team.name"].dropna().unique().tolist() if "team.name" in ms.columns else []
            h_team, a_team = (teams_order + [None, None])[:2]

        match_dt = (
            pd.to_datetime(ms["match_date"].dropna().iloc[0]).date()
            if "match_date" in ms.columns and ms["match_date"].notna().any()
            else None
        )

        goals_by_team = (
            ms.groupby("team.name")["is_goal"].sum().astype(int).to_dict()
            if "is_goal" in ms.columns and "team.name" in ms.columns
            else {}
        )
        xg_by_team = (
            ms.groupby("team.name")["xg"].sum().round(2).to_dict()
            if "xg" in ms.columns and "team.name" in ms.columns
            else {}
        )

        h = h_team or next(iter(goals_by_team.keys()), "Home")
        a = a_team or (list(goals_by_team.keys())[1] if len(goals_by_team) > 1 else "Away")

        h_goals = int(goals_by_team.get(h, 0))
        a_goals = int(goals_by_team.get(a, 0))
        h_xg = float(xg_by_team.get(h, 0.0))
        a_xg = float(xg_by_team.get(a, 0.0))

        center = st.columns([1, 6, 1])[1]
        with center:
            caption = meta_str if match_dt is None else f"{meta_str} • {match_dt}"
            st.markdown(
                f"""
                <div style="text-align:center;">
                <div style="font-size:28px; font-weight:700; line-height:1.2; margin-bottom:2px;">
                    {h} , {a}
                </div>
                <div style="font-size:44px; font-weight:900; line-height:1.0; color:#e5e7eb; font-variant-numeric: tabular-nums;">
                    {h_goals} - {a_goals}
                </div>
                <div style="margin-top:6px; opacity:0.8; font-size:14px;">
                    {caption}
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"xG , {h}", f"{h_xg:.2f}")
        with c2:
            st.metric(f"xG , {a}", f"{a_xg:.2f}")
        with c3:
            st.metric(f"Goals , {h}", h_goals)
        with c4:
            st.metric(f"Goals , {a}", a_goals)

        total_shots = len(ms)
        avg_xg = float(ms["xg"].mean()) if "xg" in ms.columns and total_shots else 0.0
        conv = f"{(100 * ms['is_goal'].sum() / total_shots):.1f}%" if total_shots and "is_goal" in ms.columns else "0%"
        shooters = ms["player.name"].nunique() if "player.name" in ms.columns else 0

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Total Shots", total_shots)
        with g2:
            st.metric("Avg xG per Shot", f"{avg_xg:.2f}")
        with g3:
            st.metric("Conversion %", conv)
        with g4:
            st.metric("Players with shots", shooters)

        st.subheader("Cumulative xG Timeline")
        timeline = cumulative_xg_plot(ms)
        try:
            timeline.update_layout(
                height=340 if compact else 520,
                margin=dict(l=20, r=20, t=30, b=40),
                legend=dict(orientation="h", y=-0.2),
            )
        except Exception:
            pass
        plot(timeline)

        st.markdown("---")
        st.caption(f"Data loaded from {len(shots)} total shots across {len(match_ids)} matches")

    # ───────────────────── xG Model & Pitch ─────────────────────
    with tab_xg_pitch:
        st.header("Shot Map & xG Model")

        st.markdown(
            """
            <style>
            .xg-help {
                padding: 10px 14px;
                border-radius: 10px;
                font-size: 0.95rem;
                line-height: 1.35;
                margin-top: 4px;
                border: 1px solid rgba(0,0,0,0.08);
                background: rgba(0,0,0,0.035);
            }
            .xg-help strong { font-weight: 600; }
            .xg-help ul { margin: 6px 0 0 1.1rem; padding: 0; }
            .xg-help li { margin: 2px 0; }
            [data-theme="dark"] .xg-help {
                border: 1px solid rgba(255,255,255,0.10);
                background: rgba(255,255,255,0.05);
            }
            </style>

            <div class="xg-help">
            <strong>How to read this plot</strong>
            <ul>
                <li>Each dot is a shot. Dot size scales with predicted xG.</li>
                <li>Colors: green = goal, yellow = high xG, red = low xG.</li>
                <li>Use the xG Threshold slider to highlight high-probability shots.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_report_path = settings.models_dir / "model_report.json"
        if model_report_path.exists():
            with open(model_report_path) as f:
                metrics = json.load(f)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            with m2:
                st.metric("Brier Score", f"{metrics.get('brier', 0):.3f}")
            with m3:
                st.metric("Model", str(metrics.get("model", "xgb")).upper())

        xg_threshold = st.slider(
            "xG Threshold (highlight shots above this value)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )

        st.subheader("Shot Map")
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#0B132B", line_color="#E5E7EB")
        fig_w, fig_h = (9, 6) if compact else (12, 8)
        fig, ax = pitch.draw(figsize=(fig_w, fig_h))

        for _, r in ms.iterrows():
            base_color = "#22c55e" if int(r.get("is_goal", 0) or 0) == 1 else ("#fbbf24" if float(r.get("xg", 0.0) or 0.0) >= xg_threshold else "#ef4444")
            color = "#60a5fa" if (selected_players and r.get("player.name") in selected_players) else base_color

            pitch.scatter(
                r.get("location.x", None),
                r.get("location.y", None),
                s=max(20, 300 * float(r.get("xg", 0.0) or 0.0)),
                color=color,
                ax=ax,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#22c55e", label="Goal"),
            Patch(facecolor="#fbbf24", label=f"High xG (≥{xg_threshold})"),
            Patch(facecolor="#ef4444", label=f"Low xG (<{xg_threshold})"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.8)
        st.pyplot(fig, clear_figure=True)
        st.caption("Shot locations use StatsBomb coordinates (origin at top-left).")

        calibration_plot_path = settings.plots_dir / "calibration.png"
        if calibration_plot_path.exists():
            st.subheader("Model Calibration")
            try:
                st.image(str(calibration_plot_path), use_container_width=True)
            except TypeError:
                st.image(str(calibration_plot_path), use_column_width=True)

    # ───────────────────── Passing Network ─────────────────────
    with tab_passing:
        from src.passing_network import build_team_network

        st.header("Passing Network")
        st.markdown(f"Team: {team}")

        net = build_team_network(match_id=match_id, team_name=team, min_edge=filters["pass_threshold"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Players", len(net.nodes))
        with c2:
            st.metric("Passing Connections", len(net.edges))
        with c3:
            max_passes = int(net.edges["count"].max()) if hasattr(net, "edges") and not net.edges.empty else 0
            st.metric("Max Passes", max_passes)

        if not hasattr(net, "edges") or net.edges.empty:
            st.info(f"No passing connections with at least {filters['pass_threshold']} passes.")
        else:
            pitch = Pitch(pitch_type="statsbomb", pitch_color="#0B132B", line_color="#E5E7EB")
            pn_w, pn_h = (9, 6) if compact else (12, 8)
            fig2, ax2 = pitch.draw(figsize=(pn_w, pn_h))

            edge_scale = max(1.0, float(net.edges["count"].max() or 0))

            if "player_name" not in net.nodes.columns:
                if "player" in net.nodes.columns:
                    net.nodes["player_name"] = net.nodes["player"].astype(str)
                else:
                    net.nodes["player_name"] = ""

            def _display_last_name(value: str) -> str:
                if not isinstance(value, str) or not value:
                    return ""
                s = value.strip()
                if not any(ch.isalpha() for ch in s):
                    return s
                return s.split()[-1]

            for _, e in net.edges.iterrows():
                src = net.nodes[net.nodes["player"] == e["source"]][["x_mean", "y_mean"]].mean()
                dst = net.nodes[net.nodes["player"] == e["target"]][["x_mean", "y_mean"]].mean()
                if pd.isna(src["x_mean"]) or pd.isna(dst["x_mean"]):
                    continue
                lw = 0.5 + (float(e["count"]) / edge_scale) * 5
                pitch.lines(
                    src["x_mean"], src["y_mean"], dst["x_mean"], dst["y_mean"],
                    lw=lw, comet=False, ax=ax2, alpha=0.6, color="#94a3b8"
                )

            for _, n in net.nodes.iterrows():
                node_color = "#22c55e" if (selected_players and n.get("player_name") in selected_players) else "#60a5fa"
                pitch.scatter(
                    n["x_mean"], n["y_mean"],
                    s=80 + 3 * float(n.get("touches", 0) or 0),
                    ax=ax2, color=node_color, edgecolors="#1f2937", linewidth=2
                )
                label = _display_last_name(n.get("player_name", ""))
                ax2.text(
                    n["x_mean"], n["y_mean"] - 2,
                    label, ha="center", va="top",
                    fontsize=8, color="#E5E7EB", weight="bold"
                )

            st.pyplot(fig2, clear_figure=True)

    # ───────────────────── Statistics ─────────────────────
    with tab_stats:
        st.header("Statistics")

        st.subheader("Team Statistics")
        if {"team.name", "xg", "is_goal"} <= set(ms.columns):
            team_stats = (
                ms.groupby("team.name")
                .agg(
                    xg_sum=("xg", "sum"),
                    xg_mean=("xg", "mean"),
                    goals=("is_goal", "sum"),
                    shots=("is_goal", "count"),
                    avg_distance=("shot_distance", "mean") if "shot_distance" in ms.columns else ("xg", "mean"),
                )
                .round(2)
            )
            team_stats = team_stats.rename(
                columns={
                    "xg_sum": "Total xG",
                    "xg_mean": "Avg xG per Shot",
                    "goals": "Goals",
                    "shots": "Shots",
                    "avg_distance": "Avg Distance",
                }
            )
            _df_full_width(team_stats)

        st.subheader("Top Players by xG")
        if "player.name" in ms.columns:
            player_stats = (
                ms.groupby("player.name")
                .agg(total_xg=("xg", "sum"), goals=("is_goal", "sum"), avg_distance=("shot_distance", "mean") if "shot_distance" in ms.columns else ("xg", "mean"))
                .round(2)
                .sort_values("total_xg", ascending=False)
                .head(10)
            )
            player_stats = player_stats.rename(columns={"total_xg": "Total xG", "goals": "Goals", "avg_distance": "Avg Distance"})
            _df_full_width(player_stats)

        st.subheader("Export Data")
        csv = ms.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Shot Data (CSV)",
            data=csv,
            file_name=f"shots_match_{match_id}.csv",
            mime="text/csv",
        )

    # ───────────────────── Settings / Data Fetch ─────────────────────
    with tab_settings:
        st.header("Settings")

        st.subheader("Cache Management")
        st.markdown("Clear cached data to force a refresh from source.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear Data Cache"):
                load_data.clear()
                st.success("Data cache cleared!")
                st.rerun()
        with c2:
            if st.button("Clear Model Cache"):
                load_model.clear()
                st.success("Model cache cleared!")
                st.rerun()

        st.markdown("---")
        st.subheader("Data Source")
        st.markdown("StatsBomb Open Data")
        st.markdown("Data is fetched from: https://github.com/statsbomb/open-data")

        st.subheader("Add matches by competition / season")
        with st.form("add_by_comp_season"):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                comp_options = sorted(_list_competitions_df()["competition_name"].dropna().unique().tolist())
                default_idx = comp_options.index("Bundesliga") if "Bundesliga" in comp_options else 0
                comp_name = st.selectbox("Competition", comp_options, index=default_idx)
            with c2:
                season_opts = ["(latest)"] + _list_seasons_for_comp(comp_name)
                season_choice = st.selectbox("Season", season_opts, index=0)
                season_name = None if season_choice == "(latest)" else season_choice
            with c3:
                n_comp = st.number_input("How many matches", min_value=4, max_value=200, value=20, step=4)
            submitted = st.form_submit_button("Fetch & preprocess")

        if submitted:
            from src.open_data import collect_matches_by_name
            from src.passing_network import build_and_save_passing_events
            from src.preprocess_shots import build_processed_shots

            try:
                with st.spinner("Fetching matches & building artifacts..."):
                    mids = collect_matches_by_name(
                        competition_name=comp_name,
                        season_name=season_name or None,
                        limit=int(n_comp),
                    )
                    build_and_save_passing_events(mids)
                    build_processed_shots(mids)

                try:
                    load_data.clear()
                    st.cache_data.clear()
                except Exception:
                    pass

                st.success(f"Added/updated {len(mids)} matches from {comp_name}. Reloading...")
                st.rerun()

            except Exception as e:
                st.error(f"Could not fetch matches: {e}")

        st.markdown("---")
        st.subheader("About")
        st.markdown(
            """
            This dashboard provides football analytics using Expected Goals (xG)
            modeling and passing network analysis.

            Features
            - xG prediction using XGBoost with calibration
            - Shot maps with xG visualization
            - Passing networks with player positions
            - Team and player statistics
            """
        )

        st.markdown("---")
        st.info("Tip: Use the sidebar filters to adjust minute range and passing thresholds.")


# Streamlit entrypoint
run()
