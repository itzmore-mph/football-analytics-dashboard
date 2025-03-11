import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Define file paths dynamically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

def create_passing_network(df):
    """Generates a passing network graph from the dataset."""
    
    # Ensure required columns exist
    required_cols = {'player', 'team'}
    if not required_cols.issubset(df.columns):
        print(f"Error: Missing required columns in {DATA_PATH}")
        return None, None

    # Create a NetworkX directed graph
    G = nx.DiGraph()
    player_positions = {}  # Dictionary to store player positions

    # Assign random positions for now
    unique_players = df["player"].unique()
    for i, player in enumerate(unique_players):
        player_positions[player] = (i % 5, i // 5)  # Adjust layout positioning

    # Add edges based on passes
    for i in range(1, len(df)):
        passer = df.loc[i - 1, 'player']
        receiver = df.loc[i, 'player']
        team = df.loc[i, 'team']

        if passer != receiver:  # Avoid self-passes
            if G.has_edge(passer, receiver):
                G[passer][receiver]['weight'] += 1  # Increase pass count
            else:
                G.add_edge(passer, receiver, weight=1, team=team)

    return G, player_positions

def plot_passing_network(G, player_positions):
    """Plots the passing network on a football pitch."""
    
    if G is None:
        print("⚠️ No valid passing network data to plot.")
        return
    
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 7))

    # Draw network nodes & edges
    nx.draw_networkx_nodes(G, player_positions, node_size=500, ax=ax, node_color="red")
    nx.draw_networkx_edges(G, player_positions, ax=ax, width=[d['weight'] for (_, _, d) in G.edges(data=True)], alpha=0.7, edge_color="blue")
    nx.draw_networkx_labels(G, player
