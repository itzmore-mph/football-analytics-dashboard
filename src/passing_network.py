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
        return None

    # Initialize NetworkX graph
    G = nx.DiGraph()

    # Add edges based on passes (assuming previous player made the pass)
    for i in range(1, len(df)):
        passer = df.loc[i - 1, 'player']
        receiver = df.loc[i, 'player']
        team = df.loc[i, 'team']

        if passer != receiver:  # Avoid self-passes
            if G.has_edge(passer, receiver):
                G[passer][receiver]['weight'] += 1  # Increase pass count
            else:
                G.add_edge(passer, receiver, weight=1, team=team)

    return G

def plot_passing_network(G):
    """Plots the passing network on a football pitch."""
    
    if G is None:
        print("⚠️ No valid passing network data to plot.")
        return
    
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 7))

    # Extract positions (dummy positions)
    pos = nx.spring_layout(G, seed=42)

    # Draw network nodes & edges
    nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[d['weight'] for (_, _, d) in G.edges(data=True)], ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    plt.title("Passing Network")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
    else:
        df = pd.read_csv(DATA_PATH)
        G = create_passing_network(df)
        plot_passing_network(G)
