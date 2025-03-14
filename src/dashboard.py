import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import networkx as nx

# Define Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

df = load_data()

# Streamlit UI
st.title("Football Analytics Dashboard")
if df is None:
    st.warning("Please upload valid processed data to continue.")
    st.stop()

# Shot Data Visualization
st.subheader("Shot Data")
fig, ax = plt.subplots()
pitch = Pitch(line_color='black')
pitch.draw(ax=ax)
ax.scatter(df["x"] * 120, df["y"] * 80, c=df["xG"], cmap='Reds', edgecolors='black', s=80)
st.pyplot(fig)

# Passing Network Visualization
st.subheader("Passing Network")
fig, ax = plt.subplots()
pitch.draw(ax=ax)
G = nx.DiGraph()
for _, row in df.iterrows():
    if "passer" in df.columns and "receiver" in df.columns:
        G.add_edge(row["passer"], row["receiver"], weight=row.get("pass_count", 1))
if G.number_of_nodes() == 0:
    st.warning("Passing network not available.")
st.pyplot(fig)

st.success("Dashboard Loaded Successfully!")
