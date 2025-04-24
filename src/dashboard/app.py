# src/dashboard/app.py
import streamlit as st
from dashboard.data import load_shot_data, load_passing_data
from dashboard.plots import plot_shot_map, plot_passing_network

# File paths
SHOT_CSV = "data/processed_shots.csv"
PASSING_CSV = "data/processed_passing_data.csv"


def main() -> None:
    """Entry point for the Football Analytics Dashboard app."""
    st.title("⚽ Football Analytics Dashboard")

    # Load data
    df_shots = st.cache_data(load_shot_data)(SHOT_CSV)
    df_passes = st.cache_data(load_passing_data)(PASSING_CSV)

    # Determine list of teams for filtering
    teams = []
    if df_shots is not None and "team" in df_shots.columns:
        teams = df_shots["team"].unique().tolist()

    # Sidebar: Team filter and stats
    if teams:
        selected_team = st.sidebar.selectbox("Select Team", teams)
        df_shots = df_shots[df_shots["team"] == selected_team]
        if df_passes is not None and "team" in df_passes.columns:
            df_passes = df_passes[df_passes["team"] == selected_team]
        st.sidebar.markdown(
            f"🎯 Shots: {len(df_shots)}  "
            f"🔄 Passes: {len(df_passes) if df_passes is not None else 0}"
        )
    else:
        st.sidebar.info("No shot data available to filter.")

    # ─── NEW SIDEBAR CONTROLS FOR PASSING NETWORK ─────────────────────────
    min_pass = st.sidebar.slider(
        "Min passes between players",
        1,
        int(df_passes.pass_count.max() if df_passes is not None else 1),
        2,
    )
    layout = st.sidebar.selectbox(
        "Network layout", ["statsbomb", "spring", "circular"]
    )
    show_labels = st.sidebar.checkbox("Show all labels", value=False)
    # ─────────────────────────────────────────────────────────────────────

    # Main tabs
    tab1, tab2 = st.tabs(["Shot Map (xG)", "Passing Network"])

    with tab1:
        st.subheader("Shot Map with xG")
        try:
            fig = plot_shot_map(df_shots)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(str(e))

    with tab2:
        st.subheader("Passing Network")
        try:
            # ─── PASS THE NEW ARGUMENTS INTO YOUR PLOTTING FUNCTION ─────────
            fig = plot_passing_network(
                df_passes,
                min_pass=min_pass,
                layout=layout,
                show_labels=show_labels,
            )
            st.pyplot(fig, use_container_width=True)
            # ────────────────────────────────────────────────────────────────
        except Exception as e:
            st.warning(str(e))