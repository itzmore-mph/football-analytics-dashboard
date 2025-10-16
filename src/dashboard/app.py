# src/dashboard/app.py
from __future__ import annotations

from typing import Any, cast

import json
import joblib
import pandas as pd
from mplsoccer import Pitch

# Absolute imports (cloud-safe)
from src.config import settings
from src.dashboard.components import metric_badge, sidebar_filters
from src.dashboard.plots import cumulative_xg_plot
from src.dashboard.theming import set_theme
from src.features_xg import FEATURE_COLUMNS

# Streamlit safe imports / stubs
try:
    import streamlit as _st

    st = cast(Any, _st)
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except Exception:

    class _StreamlitStub:
        def __getattr__(self, name: str):
            raise RuntimeError("Streamlit is required to run the dashboard")

    st = _StreamlitStub()  # type: ignore[assignment]

    def _cache_data(*_a: Any, **_k: Any):
        def _wrap(func):
            return func

        return _wrap

    def _cache_resource(*_a: Any, **_k: Any):
        def _wrap(func):
            return func

        return _wrap

    cache_data = cast(Any, _cache_data)
    cache_resource = cast(Any, _cache_resource)


# Streamlit compatibility helpers (work across versions)
def _df_full_width(df: pd.DataFrame, **kwargs):
    """
    Prefer new API (width='stretch');
    fall back to legacy (use_container_width=True).
    """
    try:
        # Newer Streamlit (some local envs): accepts string widths
        return st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        # Older/Cloud images: only accept int/None and legacy flag
        return st.dataframe(df, use_container_width=True, **kwargs)


def _plotly_full_width(fig, **kwargs):
    """
    Ensure Plotly charts size to container across versions.
    Move any config kwargs into the config dict explicitly.
    """
    cfg = kwargs.pop("config", {})
    # never pass width strings to Plotly itself; let Streamlit handle sizing
    try:
        return st.plotly_chart(fig, width="stretch", config=cfg, **kwargs)
    except TypeError:
        return st.plotly_chart(
            fig,
            use_container_width=True,
            config=cfg,
            **kwargs,
        )


@cache_data(show_spinner=False)
def _list_competitions_df() -> pd.DataFrame:
    from src.open_data import competitions

    return competitions()


@cache_data(show_spinner=False)
def _list_seasons_for_comp(comp_name: str) -> list[str]:
    df = _list_competitions_df()
    return (
        df.loc[df["competition_name"] == comp_name, "season_name"]
        .dropna()
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )


def plot(fig) -> None:
    _plotly_full_width(fig)


# ---------------------------------------------------------------------
# Cloud/bootstrap helpers
# ---------------------------------------------------------------------
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
    try:
        from src.open_data import collect_demo_matches
        from src.passing_network import build_and_save_passing_events
        from src.preprocess_shots import build_processed_shots
        from src.train_xg_model import train
    except ModuleNotFoundError:
        from ..open_data import collect_demo_matches
        from ..passing_network import build_and_save_passing_events
        from ..preprocess_shots import build_processed_shots
        from ..train_xg_model import train

    mids = collect_demo_matches()
    build_and_save_passing_events(mids)
    build_processed_shots(mids)
    metrics = train(kind="xgb", calibration="isotonic")
    return {"matches": mids, "metrics": metrics}


def _scoreline_for_match(ms: pd.DataFrame) -> str:
    """
    Compute scoreline by summing is_goal per team for the selected match
    frame.
    """
    if (
        ms.empty
        or "team.name" not in ms.columns
        or "is_goal" not in ms.columns
    ):
        return ""
    goals = (
        ms.groupby("team.name")["is_goal"]
        .sum()
        .astype(int)
        .sort_index()
        .to_dict()
    )
    teams = list(goals.keys())
    if len(teams) < 2:
        return ""
    # Prefer a stable home/away ordering if present
    home_team, away_team = teams[0], teams[1]
    if "home_team" in ms.columns and "away_team" in ms.columns:
        h = ms["home_team"].dropna().astype(str).head(1).tolist()
        a = ms["away_team"].dropna().astype(str).head(1).tolist()
        if h and a:
            home_team, away_team = h[0], a[0]
    h_goals = int(goals.get(home_team, 0))
    a_goals = int(goals.get(away_team, 0))
    return f"{home_team} {h_goals} – {a_goals} {away_team}"


# ---------------------------------------------------------------------
# Data/model loaders
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
def run() -> None:
    set_theme()
    st.title("⚽ Football Analytics Dashboard")
    st.markdown("*Statistical analysis powered by StatsBomb Open Data*")

    if not _artifacts_exist():
        try:
            ctx = st.container(border=True)  # newer Streamlit
        except TypeError:
            ctx = st.container()  # older Streamlit fallback
        with ctx:
            st.info(
                "No data/model artifacts found yet.\n\n"
                "Click the button below to build a small demo dataset "
                "(fetch → feature → train)."
            )
            if st.button("Build demo data now"):
                with st.spinner("Building demo data… "):
                    _ = _build_demo_artifacts()
                # ⬇️ Clear cached frames so the new CSVs are read on rerun
                try:
                    load_data.clear()
                    st.cache_data.clear()  # safe on newer Streamlit
                except Exception:
                    pass
                st.success("Demo data built. Reloading…")
                st.rerun()
        st.stop()

    filters = sidebar_filters()

    shots, passes = load_data()
    model = load_model()

    # Sidebar: Competition/Season
    # Try to use columns from preprocessing; fallback to "(All)"
    comp_col = (
        "competition_name" if "competition_name" in shots.columns else None
    )
    season_col = "season_name" if "season_name" in shots.columns else None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    if comp_col:
        comp_options = ["(All)"] + sorted(
            shots[comp_col].dropna().unique().tolist()
        )
        selected_comp = st.sidebar.selectbox(
            "Competition",
            comp_options,
            index=0,
            help="Filter matches by competition (from StatsBomb Open Data).",
        )
    else:
        selected_comp = "(All)"

    if season_col:
        # season options depend on competition if one is selected
        season_df = shots.copy()
        if comp_col and selected_comp != "(All)":
            season_df = season_df[season_df[comp_col] == selected_comp]
        season_options = ["(All)"] + sorted(
            season_df[season_col].dropna().unique().tolist()
        )
        selected_season = st.sidebar.selectbox(
            "Season",
            season_options,
            index=0,
            help="Filter matches by season.",
        )
    else:
        selected_season = "(All)"

    if shots.empty or passes.empty or model is None:
        st.warning("Artifacts missing. Please click **Build demo data now**")
        st.stop()

    # Predict xG for shots
    shots = shots.copy()

    # Guard against missing features after preprocessing changes
    missing = [c for c in FEATURE_COLUMNS if c not in shots.columns]
    if missing:
        st.error(
            f"Missing feature columns: {missing}. "
            "Try rebuilding demo data in Settings."
        )
        st.stop()

    # Guard against unexpected model types
    if not hasattr(model, "predict_proba"):
        st.error(
            "Loaded model doesn’t support predict_proba(). "
            "Re-train the model in demo builder."
        )
        st.stop()

    shots["xg"] = model.predict_proba(shots[FEATURE_COLUMNS].values)[:, 1]

    # Sidebar — Match selection (with friendly labels)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Match Selection")

    def _match_label(mid: int) -> str:
        # Prefer persisted fixture + date from preprocessing
        if "fixture" in shots.columns:
            lab = (
                shots.loc[shots["match_id"] == mid, "fixture"]
                .dropna()
                .astype(str)
                .head(1)
                .tolist()
            )
            if lab:
                if "match_date" in shots.columns:
                    md = (
                        pd.to_datetime(
                            shots.loc[shots["match_id"] == mid, "match_date"]
                        )
                        .dropna()
                        .astype(str)
                        .head(1)
                        .tolist()
                    )
                    return f"{lab[0]} — {md[0]}" if md else lab[0]
                return lab[0]
        # Fallback: infer two team names from shots
        teams = (
            shots.loc[shots["match_id"] == mid, "team.name"]
            .dropna()
            .unique()
            .tolist()
        )
        vs = " vs ".join(teams[:2]) if teams else f"Match {mid}"
        return f"{vs} — {mid}"

    # Apply comp/season filters to available matches
    shots_filtered = shots.copy()
    if comp_col and selected_comp != "(All)":
        shots_filtered = shots_filtered[
            shots_filtered[comp_col] == selected_comp
        ]
    if season_col and selected_season != "(All)":
        shots_filtered = shots_filtered[
            shots_filtered[season_col] == selected_season
        ]

    matches = sorted(shots_filtered["match_id"].unique().tolist())
    if not matches:
        st.sidebar.warning(
            "No matches for current Competition/Season filters."
        )
        st.stop()

    match_id = st.sidebar.selectbox(
        "Match",
        matches,
        format_func=_match_label,
        help="Pick a match from the filtered set.",
    )

    # Team selector should reflect the filtered set
    teams = shots_filtered.loc[
        shots_filtered["match_id"] == match_id, "team.name"
    ].unique().tolist()
    team = st.sidebar.selectbox("Team (for passing network)", teams, index=0)

    # Filter shots by match and minute range (use filtered base)
    ms = shots_filtered[shots_filtered["match_id"] == match_id]
    minute_min, minute_max = filters["minute_range"]
    ms = ms[(ms["minute"] >= minute_min) & (ms["minute"] <= minute_max)]

    if ms.empty:
        st.info("No shots in the selected minute range.")
        return

    # Player filter
    available_players = sorted(
        ms.get("player.name", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    selected_players = st.sidebar.multiselect(
        "Players (filter shots & stats)", available_players
    )
    if selected_players:
        ms = ms[ms["player.name"].isin(selected_players)]

    # Tabs
    (
        tab_overview,
        tab_xg_pitch,
        tab_passing,
        tab_stats,
        tab_settings,
    ) = st.tabs(
        [
            "📊 Overview",
            "🎯 xG Model & Pitch",
            "🔗 Passing Network",
            "📈 Statistics",
            "⚙️ Data Fetch & Settings",
        ]
    )

    # Tab 1: Overview
    with tab_overview:
        st.header("Match Overview")
        left_meta = []
        if comp_col and selected_comp != "(All)":
            left_meta.append(selected_comp)
        if season_col and selected_season != "(All)":
            left_meta.append(str(selected_season))
        meta_str = (
            " • ".join(left_meta)
            if left_meta
            else "All competitions/seasons"
        )

        scoreline = _scoreline_for_match(ms)
        if scoreline:
            st.markdown(f"**{scoreline}**  \n_{meta_str}_")
        else:
            st.caption(meta_str)

        # Team xG metrics
        team_xg = ms.groupby("team.name")["xg"].sum().round(2)
        cols = st.columns(len(team_xg))
        for i, (t, v) in enumerate(team_xg.items()):
            with cols[i]:
                metric_badge(f"xG — {t}", v)

        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Shots", len(ms))
        with col2:
            st.metric("Goals", int(ms["is_goal"].sum()))
        with col3:
            shots_on_target = len(ms[ms["is_goal"] == 1])
            conv = (
                f"{(100 * shots_on_target / len(ms)):.1f}%"
                if len(ms)
                else "0%"
            )
            st.metric("Conversion %", conv)

        # Cumulative xG timeline
        st.subheader("Cumulative xG Timeline")
        timeline = cumulative_xg_plot(ms)
        plot(timeline)

        # Data freshness
        st.markdown("---")
        st.caption(
            (
                f"📅 Data loaded from {len(shots)} total shots across "
                f"{len(matches)} matches"
            )
        )

    # Tab 2: xG Model & Pitch
    with tab_xg_pitch:
        st.header("Shot Map & xG Model")
        st.caption(
            "• Each dot is a shot. Dot **size** scales with predicted xG "
            "(chance of scoring).  \n"
            "• **Colors**: green=goal, yellow=high xG (above slider), "
            "red=low xG.  \n"
            "• Use the **xG Threshold** slider to highlight high-probability "
            "shots."
        )

        # Model summary (if available)
        model_report_path = settings.models_dir / "model_report.json"
        if model_report_path.exists():
            with open(model_report_path) as f:
                metrics = json.load(f)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            with col2:
                st.metric("Brier Score", f"{metrics.get('brier', 0):.3f}")
            with col3:
                st.metric("Model", metrics.get("model", "xgb").upper())

        # xG threshold slider
        xg_threshold = st.slider(
            "xG Threshold (highlight shots above this value)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help=(
                "Shots with predicted xG ≥ this value are highlighted in "
                "yellow."
            ),
        )

        # Shot map
        st.subheader("Shot Map")
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#0B132B",
            line_color="#E5E7EB",
        )
        fig, ax = pitch.draw(figsize=(12, 8))

        for _, r in ms.iterrows():
            base_color = (
                "#22c55e"
                if r["is_goal"] == 1
                else ("#fbbf24" if r["xg"] >= xg_threshold else "#ef4444")
            )
            color = (
                "#60a5fa"
                if (
                    selected_players
                    and r.get("player.name") in selected_players
                )
                else base_color
            )
            pitch.scatter(
                r["location.x"],
                r["location.y"],
                s=max(20, 300 * r["xg"]),
                color=color,
                ax=ax,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#22c55e", label="Goal"),
            Patch(facecolor="#fbbf24", label=f"High xG (≥{xg_threshold})"),
            Patch(facecolor="#ef4444", label=f"Low xG (<{xg_threshold})"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.8)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Shot locations use StatsBomb coordinates "
            "(origin at top-left)."
        )

        # Calibration plot (if available)
        calibration_plot_path = settings.plots_dir / "calibration.png"
        if calibration_plot_path.exists():
            st.subheader("Model Calibration")
            st.image(str(calibration_plot_path), use_column_width=True)
            st.caption(
                "Calibration plot showing predicted xG "
                "vs actual goal rate"
            )

    # -----------------------------------------------------------------
    # Tab 3: Passing Network
    # -----------------------------------------------------------------
    with tab_passing:
        from src.passing_network import build_team_network

        st.header("Passing Network")
        st.markdown(f"**Team:** {team}")

        net = build_team_network(
            match_id=match_id,
            team_name=team,
            min_edge=filters["pass_threshold"],
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Players", len(net.nodes))
        with col2:
            st.metric("Passing Connections", len(net.edges))
        with col3:
            st.metric(
                "Max Passes",
                int(net.edges["count"].max()) if not net.edges.empty else 0,
            )

        if net.edges.empty:
            st.info(
                f"No passing connections with at least "
                f"{filters['pass_threshold']} passes."
            )
        else:
            pitch = Pitch(
                pitch_type="statsbomb",
                pitch_color="#0B132B",
                line_color="#E5E7EB",
            )
            fig2, ax2 = pitch.draw(figsize=(12, 8))

            edge_scale = max(1.0, float(net.edges["count"].max() or 0))

            # Ensure player_name column exists for labeling
            if "player_name" not in net.nodes.columns:
                if "player" in net.nodes.columns:
                    net.nodes["player_name"] = net.nodes["player"].astype(str)
                else:
                    net.nodes["player_name"] = ""

            def _display_last_name(value: str) -> str:
                if not isinstance(value, str) or not value:
                    return ""
                s = value.strip()
                if not any(ch.isalpha() for ch in s):  # numeric-like id
                    return s
                return s.split()[-1]

            # Draw edges
            for _, e in net.edges.iterrows():
                src = net.nodes[net.nodes["player"] == e["source"]][
                    ["x_mean", "y_mean"]
                ].mean()
                dst = net.nodes[net.nodes["player"] == e["target"]][
                    ["x_mean", "y_mean"]
                ].mean()
                if pd.isna(src["x_mean"]) or pd.isna(dst["x_mean"]):
                    continue
                lw = 0.5 + (e["count"] / edge_scale) * 5
                pitch.lines(
                    src["x_mean"],
                    src["y_mean"],
                    dst["x_mean"],
                    dst["y_mean"],
                    lw=lw,
                    comet=False,
                    ax=ax2,
                    alpha=0.6,
                    color="#94a3b8",
                )

            # Draw nodes (highlight selected players if any)
            for _, n in net.nodes.iterrows():
                node_color = (
                    "#22c55e"
                    if (
                        selected_players
                        and n.get("player_name") in selected_players
                    )
                    else "#60a5fa"
                )
                pitch.scatter(
                    n["x_mean"],
                    n["y_mean"],
                    s=80 + 3 * n["touches"],
                    ax=ax2,
                    color=node_color,
                    edgecolors="#1f2937",
                    linewidth=2,
                )
                label = _display_last_name(n.get("player_name", ""))
                ax2.text(
                    n["x_mean"],
                    n["y_mean"] - 2,
                    label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#E5E7EB",
                    weight="bold",
                )

            st.pyplot(fig2, clear_figure=True)

    # -----------------------------------------------------------------
    # Tab 4: Statistics
    # -----------------------------------------------------------------
    with tab_stats:
        st.header("Statistics")

        # Team-level stats
        st.subheader("Team Statistics")
        team_stats = (
            ms.groupby("team.name")
            .agg(
                {
                    "xg": ["sum", "mean"],
                    "is_goal": ["sum", "count"],
                    "shot_distance": "mean",
                }
            )
            .round(2)
        )
        team_stats.columns = [
            "Total xG",
            "Avg xG per Shot",
            "Goals",
            "Shots",
            "Avg Distance",
        ]
        _df_full_width(team_stats)

        # Player-level stats
        st.subheader("Top Players by xG")
        if "player.name" in ms.columns:
            player_stats = (
                ms.groupby("player.name")
                .agg({"xg": "sum", "is_goal": "sum", "shot_distance": "mean"})
                .round(2)
                .sort_values("xg", ascending=False)
                .head(10)
            )
            player_stats.columns = ["Total xG", "Goals", "Avg Distance"]
            _df_full_width(player_stats)

        # Export shots
        st.subheader("Export Data")
        csv = ms.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Shot Data (CSV)",
            data=csv,
            file_name=f"shots_match_{match_id}.csv",
            mime="text/csv",
        )

    # -----------------------------------------------------------------
    # Tab 5: Settings
    # -----------------------------------------------------------------
    with tab_settings:
        st.header("Settings")

        st.subheader("Cache Management")
        st.markdown("Clear cached data to force a refresh from source.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Data Cache"):
                load_data.clear()
                st.success("Data cache cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Clear Model Cache"):
                load_model.clear()
                st.success("Model cache cleared!")
                st.rerun()

        st.markdown("---")
        st.subheader("Data Source")
        st.markdown("**StatsBomb Open Data**")
        st.markdown(
            "Data is fetched from: "
            "https://github.com/statsbomb/open-data"
        )

        # Add matches by competition/season (targeted)
        st.subheader("Add matches by competition / season")
        with st.form("add_by_comp_season"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                comp_options = sorted(
                    _list_competitions_df()["competition_name"]
                    .dropna()
                    .unique()
                    .tolist()
                )
                default_idx = (
                    comp_options.index("NWSL")
                    if "NWSL" in comp_options
                    else 0
                )
                comp_name = st.selectbox(
                    "Competition", comp_options, index=default_idx
                )
            with col2:
                season_opts = ["(latest)"] + _list_seasons_for_comp(comp_name)
                season_choice = st.selectbox("Season", season_opts, index=0)
                season_name = (
                    None if season_choice == "(latest)" else season_choice
                )
            with col3:
                n_comp = st.number_input(
                    "How many matches",
                    min_value=4,
                    max_value=200,
                    value=20,
                    step=4,
                )
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
                st.success(
                    f"Added/updated {len(mids)} matches from {comp_name} "
                    f"{season_name or ''}. Reloading…"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not fetch matches: {e}")

        st.markdown("---")
        st.subheader("About")
        # Cleaned up stray parentheses/quotes so the copy renders nicely
        st.markdown(
            """
            This dashboard provides football analytics using
            **Expected Goals (xG)**
            modeling and passing network analysis.

            **Features**
            - xG prediction using XGBoost with calibration
            - Shot maps with xG visualization
            - Passing networks with player positions
            - Team and player statistics
            """
        )

        st.markdown("---")
        st.info(
            "💡 **Tip:** Use the sidebar filters to adjust minute range "
            "and passing thresholds."
        )
