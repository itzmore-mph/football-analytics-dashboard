import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch

# Define data paths
DATA_PATH = "data/processed_shots.csv"

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        if "x" not in df.columns or "y" not in df.columns or "xG" not in df.columns:
            st.error("Data is missing required columns: 'x', 'y', 'xG'")
            return None
        return df
    except FileNotFoundError:
        st.error(f"File not found: {DATA_PATH}")
        return None

df = load_data()

# Streamlit Layout
st.title("Football Analytics Dashboard")
st.sidebar.header("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["xG Shot Map", "xG Distribution", "xG Over Time"])

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
    st.subheader("⏳ xG Progression Over Time")
    df["minute_group"] = df["minute"] // 10 * 10  # Group by 10-min intervals
    xg_timeline = df.groupby("minute_group")["xG"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(xg_timeline.index, xg_timeline.values, marker="o", linestyle="-", color="red")
    plt.xlabel("Minute")
    plt.ylabel("Total xG")
    plt.title("xG Progression Over Time")
    st.pyplot(fig)

# Display Selected Page
if df is not None:
    if page == "xG Shot Map":
        plot_xG_shotmap(df)
    elif page == "xG Distribution":
        plot_xG_distribution(df)
    elif page == "xG Over Time":
        plot_xG_timeline(df)
else:
    st.warning("Please upload valid processed shot data to continue.")

st.success("Dashboard Loaded Successfully!")
