from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import joblib
import pandas as pd
from mplsoccer import Pitch

# Absolute imports (cloud-safe)
from src.config import settings
from src.features_xg import FEATURE_COLUMNS
from src.dashboard.components import metric_badge, sidebar_filters
from src.dashboard.plots import cumulative_xg_plot
from src.dashboard.theming import set_theme

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
        def _wrap(func): return func
        return _wrap

    def _cache_resource(*_a: Any, **_k: Any):
        def _wrap(func): return func
        return _wrap
    cache_data = cast(Any, _cache_data)
    cache_resource = cast(Any, _cache_resource)


# Cloud bootstrap
def _artifacts_exist() -> bool:
    return all([
        settings.passing_events_csv.exists(),
        settings.processed_shots_csv.exists(),
        settings.model_path.exists(),
    ])


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
    st.title("Football Analytics Dashboard")

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
    matches = sorted(shots["match_id"].unique().tolist())
    match_id = st.selectbox("Match", matches)
    teams = shots.loc[
        shots["match_id"] == match_id, "team.name"
    ].unique().tolist()
    team = st.selectbox("Team (for passing network)", teams, index=0)

    # Overview metrics
    ms = shots[shots["match_id"] == match_id]
    minute_min, minute_max = filters["minute_range"]
    ms = ms[(ms["minute"] >= minute_min) & (ms["minute"] <= minute_max)]
    if ms.empty:
        st.info("No shots in the selected minute range.")
        return
    team_xg = ms.groupby("team.name")["xg"].sum().round(2)
    cols = st.columns(len(team_xg))
    for i, (t, v) in enumerate(team_xg.items()):
        with cols[i]:
            metric_badge(f"xG — {t}", v)

    # Cumulative xG timeline
    timeline = cumulative_xg_plot(ms)
    st.plotly_chart(timeline, use_container_width=True)

    # Shot map
    st.subheader("Shot Map & xG")
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#0B132B",
        line_color="#E5E7EB"
    )
    fig, ax = pitch.draw(figsize=(10, 6))
    for _, r in ms.iterrows():
        color = "#22c55e" if r["is_goal"] == 1 else "#ef4444"
        pitch.scatter(
            r["location.x"], r["location.y"],
            s=max(20, 300 * r["xg"]), color=color,
            ax=ax, alpha=0.8
        )
    st.pyplot(fig, clear_figure=True)

    # Passing network
    st.subheader("Passing Network")
    from src.passing_network import build_team_network  # absolute again
    net = build_team_network(
        match_id=match_id, team_name=team, min_edge=filters["pass_threshold"]
    )
    fig2, ax2 = pitch.draw(figsize=(10, 6))
    edge_scale = max(1.0, float(net.edges["count"].max() or 0))
    for _, e in net.edges.iterrows():
        src = net.nodes[
            net.nodes["player"] == e["source"]
        ][["x_mean", "y_mean"]].mean()
        dst = net.nodes[
            net.nodes["player"] == e["target"]
        ][["x_mean", "y_mean"]].mean()
        if pd.isna(src["x_mean"]) or pd.isna(dst["x_mean"]):
            continue
        lw = 0.5 + (e["count"] / edge_scale) * 5
        pitch.lines(src["x_mean"], src["y_mean"], dst["x_mean"], dst["y_mean"],
                    lw=lw, comet=False, ax=ax2, alpha=0.6)
    for _, n in net.nodes.iterrows():
        pitch.scatter(n["x_mean"], n["y_mean"], s=80 + 3 * n["touches"],
                      ax=ax2, color="#60a5fa", edgecolors="#1f2937")
        ax2.text(n["x_mean"], n["y_mean"] - 2, n["player"].split(" ")[-1],
                 ha="center", va="top", fontsize=8, color="#E5E7EB")
    st.pyplot(fig2, clear_figure=True)
