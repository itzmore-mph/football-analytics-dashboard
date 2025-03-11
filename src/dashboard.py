import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mplsoccer import Pitch

# Define Data Paths
XG_DATA_PATH = "data/processed_shots.csv"
PASSING_DATA_PATH = "data/passing_data.csv"

# Load Data
@st.cache_data
def load_xg_data():
    try:
        df = pd.read_csv(XG_DATA_PATH)
        if "x" not in df.columns or "y" not in df.columns or "xG" not in df.columns:
            st.error("xG data is missing required columns: 'x', 'y', 'xG'")
            return None
        return df
    except FileNotFoundError:
        st.error(f"File not found: {XG_DATA_PATH}")
        return None

@st.cache_data
def load_passing_data():
    try:
        df = pd.read_csv(PASSING_DATA_PATH)
        if not {"passer", "receiver", "x", "y"}.issubset(df.columns):
            st.error("Passing data is missing required columns: 'passer', 'receiver', 'x', 'y'")
            return None
        return df
    except FileNotFoundError:
        st.error(f"File not found: {PASSING_DATA_PATH}")
        return None

df_xg = load_xg_data()
df_passing = load_passing_data()

# Streamlit Layout
st.title("Football Analytics Dashboard")
st.sidebar.header("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["xG Shot Map", "xG Distribution", "xG Over Time", "Passing Network"])

# Function: xG Shot Map
def plot_xG_shotmap(df):
    st.subheader("Expected Goals (xG) Shot Map")
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 6))

    scatter = ax.scatter(
        df["x"] * 120, df["y"] * 80,  # Scale to pitch size
        c=df["xG"], cmap="Reds", edgecolors="black", s=df["xG"] * 200, alpha=0.8
    )
    plt.colorbar(scatter, ax=ax, label="Expected Goals (xG)")
    plt.title("Shot Map with Expected Goals")
    st.pyplot(fig)

# Function: xG Distribution
def plot_xG_distribution(df):
    st.subheader("xG Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["xG"], bins=20, kde=True, color="red", alpha=0.7)
    plt.xlabel("Expected Goals (xG)")
    plt.ylabel("Number of Shots")
    plt.title("xG Distribution")
    st.pyplot(fig)

# Function: xG Over Time
def plot_xG_timeline(df):
    st.subheader("xG Progression Over Time")
    df["minute_group"] = df["minute"] // 10 * 10  # Group by 10-min intervals
    xg_timeline = df.groupby("minute_group")["xG"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(xg_timeline.index, xg_timeline.values, marker="o", linestyle="-", color="red")
    plt.xlabel("Minute")
    plt.ylabel("Total xG")
    plt.title("xG Progression Over Time")
    st.pyplot(fig)

# Function: Passing Network
def plot_passing_network(df):
    st.subheader("Passing Network")

    if df is None or df.empty:
        st.warning("No passing data found. Please check your dataset.")
        return

    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 6))

    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(row["passer"], row["receiver"], weight=row.get("pass_count", 1))

    pos = {player: (row["x"] * 120, row["y"] * 80) for _, row in df.iterrows()}

    node_sizes = [G.degree(player) * 100 for player in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="blue", alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="black", alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)

    plt.title("Passing Network")
    st.pyplot(fig)

# Display Selected Page
if page == "xG Shot Map" and df_xg is not None:
    plot_xG_shotmap(df_xg)
elif page == "xG Distribution" and df_xg is not None:
    plot_xG_distribution(df_xg)
elif page == "xG Over Time" and df_xg is not None:
    plot_xG_timeline(df_xg)
elif page == "Passing Network" and df_passing is not None:
    plot_passing_network(df_passing)
else:
    st.warning("Please upload valid processed data to continue.")

st.success("Dashboard Loaded Successfully!")
