from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import joblib
import pandas as pd
from mplsoccer import Pitch

# Absolute imports (cloud-safe)
from src.config import settings
from src.dashboard.components import metric_badge, sidebar_filters
from src.dashboard.plots import cumulative_xg_plot
from src.dashboard.theming import set_theme
from src.features_xg import FEATURE_COLUMNS

F = TypeVar("F", bound=Callable[..., Any])

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

    st = _StreamlitStub()

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


def plot(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# LEGACY (catch deprecated kwargs)
_real_plotly_chart = st.plotly_chart
_ALLOWED = {"width", "height", "config", "key"}


def _guard_plotly_chart(*args, **kwargs):
    bad = [k for k in kwargs if k not in _ALLOWED]
    if bad:
        raise RuntimeError(f"Deprecated kwargs to st.plotly_chart: {bad}")
    return _real_plotly_chart(*args, **kwargs)


st.plotly_chart = _guard_plotly_chart


# Cloud bootstrap
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
    One-time builder for a tiny demo dataset on Streamlit Cloud:
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


# Data/model loaders
@cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = (
        pd.read_csv(settings.processed_shots_csv)
        if settings.processed_shots_csv.exists()
        else pd.DataFrame()
    )
    passes = (
        pd.read_csv(settings.passing_events_csv)
        if settings.passing_events_csv.exists()
        else pd.DataFrame()
    )
    return shots, passes


@cache_resource(show_spinner=False)
def load_model():
    if settings.model_path.exists():
        return joblib.load(settings.model_path)
    return None


# App
def run():
    set_theme()
    st.title("⚽ Football Analytics Dashboard")
    st.markdown("*Statistical analysis powered by StatsBomb Open Data*")

    if not _artifacts_exist():
        with st.container(border=True):
            st.info(
                "No data/model artifacts found yet.\n\n"
                "Click the button below to build a small demo dataset "
                "(fetch → feature → train)."
            )
            if st.button("Build demo data now"):
                with st.spinner(
                    "Building demo data… this is a one-time step."
                ):
                    _ = _build_demo_artifacts()
                st.success("Demo data built. Reloading…")
                st.rerun()
        st.stop()

    filters = sidebar_filters()

    shots, passes = load_data()
    model = load_model()
    if shots.empty or passes.empty or model is None:
        st.warning(
            "Artifacts missing. Please click **Build demo data now** above."
        )
        st.stop()

    # Predict xG for shots
    shots = shots.copy()
    shots["xg"] = model.predict_proba(shots[FEATURE_COLUMNS].values)[:, 1]

    # Global selectors
    st.sidebar.markdown("---")
    st.sidebar.subheader("Match Selection")
    matches = sorted(shots["match_id"].unique().tolist())
    match_id = st.sidebar.selectbox("Match", matches)
    teams = (
        shots.loc[shots["match_id"] == match_id, "team.name"].unique().tolist()
    )
    team = st.sidebar.selectbox("Team (for passing network)", teams, index=0)

    # Filter shots by match and minute range
    ms = shots[shots["match_id"] == match_id]
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

    # Create tabs
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
            "⚙️ Settings",
        ]
    )

    # Tab 1: Overview
    with tab_overview:
        st.header("Match Overview")

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
            if len(ms) > 0:
                conversion_rate = 100 * shots_on_target / len(ms)
                st.metric("Conversion %", f"{conversion_rate:.1f}%")
            else:
                st.metric("Conversion %", "0%")

        # Cumulative xG timeline
        st.subheader("Cumulative xG Timeline")
        timeline = cumulative_xg_plot(ms)
        plot(timeline)

        # Data freshness
        st.markdown("---")
        st.caption(
            f"📅 Data loaded from {len(shots)} total shots across "
            f"{len(matches)} matches"
        )

    # Tab 2: xG Model & Pitch
    with tab_xg_pitch:
        import json

        st.header("Shot Map & xG Model")

        # Show model metrics if available
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
            color = base_color
            if selected_players and r.get("player.name") in selected_players:
                color = "#60a5fa"  # highlight selected players

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

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#22c55e", label="Goal"),
            Patch(facecolor="#fbbf24", label=f"High xG (≥{xg_threshold})"),
            Patch(facecolor="#ef4444", label=f"Low xG (<{xg_threshold})"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.8)

        st.pyplot(fig, clear_figure=True)

        # Display calibration plot if available
        calibration_plot_path = settings.plots_dir / "calibration.png"
        if calibration_plot_path.exists():
            st.subheader("Model Calibration")
            st.image(str(calibration_plot_path), use_container_width=True)
            st.caption(
                "Calibration plot showing predicted xG vs actual goal rate"
            )

    # Tab 3: Passing Network
    with tab_passing:
        from src.passing_network import build_team_network

        st.header("Passing Network")
        st.markdown(f"**Team:** {team}")

        net = build_team_network(
            match_id=match_id,
            team_name=team,
            min_edge=filters["pass_threshold"],
        )

        # Network stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Players", len(net.nodes))
        with col2:
            st.metric("Passing Connections", len(net.edges))
        with col3:
            if not net.edges.empty:
                st.metric("Max Passes", int(net.edges["count"].max()))
            else:
                st.metric("Max Passes", 0)

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
                # If it's numeric-like (no letters), don't split—display as-is
                if not any(ch.isalpha() for ch in s):
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
                node_color = "#60a5fa"
                is_selected = (
                    bool(selected_players)
                    and n.get("player_name") in selected_players
                )
                if is_selected:
                    node_color = "#22c55e"
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

    # Tab 4: Statistics
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
        st.dataframe(team_stats, width="stretch")

        # Player-level stats (top scorers by xG)
        st.subheader("Top Players by xG")
        if "player.name" in ms.columns:
            player_stats = (
                ms.groupby("player.name")
                .agg({"xg": "sum", "is_goal": "sum", "shot_distance": "mean"})
                .round(2)
            )
            player_stats.columns = ["Total xG", "Goals", "Avg Distance"]
            player_stats = player_stats.sort_values(
                "Total xG", ascending=False
            )
            player_stats = player_stats.head(10)
            st.dataframe(player_stats, width="stretch")

        # Export functionality
        st.subheader("Export Data")
        csv = ms.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Shot Data (CSV)",
            data=csv,
            file_name=f"shots_match_{match_id}.csv",
            mime="text/csv",
        )

    # Tab 5: Settings
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
            "Data is fetched from: " "https://github.com/statsbomb/open-data"
        )

        # Build more data (fast path via collect_full_matches)
        st.subheader("Build more data")
        colA, colB = st.columns([3, 1])
        with colA:
            n = st.slider(
                "How many additional matches to fetch & process", 4, 120, 20, 4
            )
        with colB:
            if st.button("Fetch & preprocess more matches"):
                with st.spinner("Fetching matches & building artifacts..."):
                    from src.open_data import collect_full_matches
                    from src.passing_network import (
                        build_and_save_passing_events,
                    )
                    from src.preprocess_shots import build_processed_shots

                    mids = collect_full_matches(limit=n)
                    build_and_save_passing_events(mids)
                    build_processed_shots(mids)
                st.success(f"Added/updated {len(mids)} matches. Reloading…")
                st.rerun()
        st.markdown("---")
        st.subheader("About")
        st.markdown(
            """
        This dashboard provides football analytics using Expected Goals (xG)
        modeling and passing network analysis.

        **Features:**
        - xG prediction using XGBoost with calibration
        - Shot maps with xG visualization
        - Passing networks with player positions
        - Team and player statistics
        """
        )

        st.markdown("---")
        st.info(
            "💡 **Tip:** Use the sidebar filters to adjust minute range"
            "and passing thresholds."
        )
