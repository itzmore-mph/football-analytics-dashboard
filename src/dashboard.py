import streamlit as st
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

# Load data
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"Data file not found: {DATA_PATH}")
        return None

df = load_data()

# Function to create a passing network
def create_passing_network(df):
    """Generates a passing network graph."""
    
    if df is None:
        return None
    
    required_cols = {'player', 'team'}
    if not required_cols.issubset(df.columns):
        st.error("Missing required columns in data!")
        return None

    G = nx.DiGraph()

    for i in range(1, len(df)):
        passer = df.loc[i - 1, 'player']
        receiver = df.loc[i, 'player']
        team = df.loc[i, 'team']

        if passer != receiver:
            if G.has_edge(passer, receiver):
                G[passer][receiver]['weight'] += 1
            else:
                G.add_edge(passer, receiver, weight=1, team=team)

    return G

# Streamlit UI
st.title("Football Analytics Dashboard")
st.sidebar.header("Navigation")
selection = st.sidebar.radio("Select Visualization", ["Expected Goals (xG)", "Passing Network"])

if selection == "Passing Network":
    st.subheader("Passing Network Visualization")

    if df is not None:
        G, player_positions = create_passing_network(df)

        if G is not None:
            pitch = Pitch(pitch_type='statsbomb', line_color='black')
            fig, ax = pitch.draw(figsize=(10, 7))

            nx.draw_networkx_nodes(G, player_positions, node_size=500, ax=ax, node_color="red")
            nx.draw_networkx_edges(G, player_positions, ax=ax, width=[d['weight'] for (_, _, d) in G.edges(data=True)], alpha=0.7, edge_color="blue")
            nx.draw_networkx_labels(G, player_positions, font_size=10, ax=ax)

            st.pyplot(fig)
        else:
            st.warning("No valid passing network data to display.")
