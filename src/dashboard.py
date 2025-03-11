import os
import networkx as nx
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from passing_network import create_passing_network, plot_passing_network

# Load Data
DATA_PATH = "data/processed_shots.csv"

@st.cache_data
def load_data():
    """Loads processed shot data."""
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

# Streamlit UI
st.title("Football Analytics Dashboard")

df = load_data()

if df is None:
    st.error("Please upload valid processed data to continue.")
    st.stop()

# Shot Visualization
st.subheader("Shot Data")
fig, ax = plt.subplots(figsize=(8, 5))
pitch = Pitch(pitch_type="custom", pitch_length=120, pitch_width=80, line_color="black")
pitch.draw(ax=ax)

ax.scatter(df["x"] * 120, df["y"] * 80, c=df["xG"], cmap="Reds", edgecolors="black", s=100)
st.pyplot(fig)

# Passing Network
st.subheader("Passing Network")
G = create_passing_network(df)
fig, ax = plt.subplots(figsize=(10, 7))
plot_passing_network(G, df)
st.pyplot(fig)
