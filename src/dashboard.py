import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mplsoccer import Pitch

# Define paths
SHOT_DATA_PATH = "data/processed_shots.csv"
PASSING_DATA_PATH = "data/processed_passing_data.csv"

# Load shot data
@st.cache_data
def load_shot_data():
    if os.path.exists(SHOT_DATA_PATH):
        return pd.read_csv(SHOT_DATA_PATH)
    return None

# Load passing data
@st.cache_data
def load_passing_data():
    if os.path.exists(PASSING_DATA_PATH):
        return pd.read_csv(PASSING_DATA_PATH)
    return None

# Function to visualize shots
def plot_shot_map(df):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = pitch.draw(figsize=(8, 5))

    if df is not None and {"x", "y", "xG"}.issubset(df.columns) and len(df) > 0:
        scatter = ax.scatter(df["x"], df["y"], c=df["xG"], cmap="Reds", edgecolors="black", s=80)
        plt.colorbar(scatter, ax=ax, label="Expected Goals (xG)")
    else:
        st.warning("No shots available for selected team or missing xG data.")

    st.pyplot(fig)

# Function to visualize passing network
def plot_passing_network(df):
    if df is None or not {"passer", "receiver", "x", "y"}.issubset(df.columns) or len(df) == 0:
        st.warning("No passing data available for selected team or missing required columns.")
        return

    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black")
    fig, ax = pitch.draw(figsize=(8, 5))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["passer"], row["receiver"], weight=row.get("pass_count", 1))

    pos = {row["passer"]: (row["x"], row["y"]) for _, row in df.iterrows()}
    node_sizes = [G.degree(n) * 100 for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] / 2 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="blue", alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="black", alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="black", ax=ax)

    st.pyplot(fig)

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("⚽ Football Analytics Dashboard")

# Load data
shot_data = load_shot_data()
passing_data = load_passing_data()

# Sidebar filter
if shot_data is not None and "team" in shot_data.columns:
    team_options = shot_data["team"].unique().tolist()
    selected_team = st.sidebar.selectbox("Select Team", team_options)
    shot_data = shot_data[shot_data["team"] == selected_team]

    if passing_data is not None and "team" in passing_data.columns:
        passing_data = passing_data[passing_data["team"] == selected_team]

    st.sidebar.markdown(f"🔍 **{selected_team}**")
    st.sidebar.markdown(f"🎯 Shots: {len(shot_data)}")
    st.sidebar.markdown(f"🔄 Passes: {len(passing_data) if passing_data is not None else 0}")

# Tab layout
tab1, tab2 = st.tabs(["📈 Shot Map (xG)", "🔄 Passing Network"])

with tab1:
    st.subheader("Shot Map with xG")
    plot_shot_map(shot_data)

with tab2:
    st.subheader("Passing Network")
    plot_passing_network(passing_data)

st.success("Dashboard Loaded Successfully!")
