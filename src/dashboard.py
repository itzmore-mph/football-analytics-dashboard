import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Define the correct project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load Data (Using Absolute Paths)
DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "../models/xgboost_xg_model.pkl")

# Ensure file exists
if not os.path.exists(DATA_PATH):
    st.error(f"Missing data file: {DATA_PATH}")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()

# Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

st.title("Football Analytics Dashboard")
view = st.sidebar.radio("Choose Analysis", ["Expected Goals (xG)", "Passing Network"])

if view == "Expected Goals (xG)":
    st.subheader("Expected Goals (xG) Shot Map")
    player = st.selectbox("Choose a Player:", df["player"].unique())
    player_data = df[df["player"] == player]

    fig, ax = plt.subplots(figsize=(8, 5))
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    pitch.draw(ax=ax)

    ax.scatter(player_data["shot_distance"] * 120, player_data["shot_angle"] * 80,
               c=player_data["predicted_xG"], cmap="Reds", edgecolors="black", s=100)
    fig.colorbar(ax.collections[0], label="Expected Goals (xG)")

    st.pyplot(fig)
    st.dataframe(player_data[["minute", "shot_distance", "shot_angle", "predicted_xG"]])
