import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch

def generate_passing_network():
    # Define absolute path
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(PROJECT_ROOT, "../data/passing_data.csv")

    # Ensure data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Missing passing data file: {DATA_PATH}")
        return

    # Load data
    df = pd.read_csv(DATA_PATH)
    team = "Team X"  # Example team
    df_team = df[df["team"] == team]

    # Create a passing network
    G = nx.DiGraph()
    for _, row in df_team.iterrows():
        G.add_edge(row["passer"], row["receiver"], weight=row["pass_count"])

    # Plot passing network
    fig, ax = plt.subplots(figsize=(8, 5))
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    pitch.draw(ax=ax)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=5000, edge_color="black", width=2)
    plt.title(f"Passing Network: {team}")
    plt.show()

if __name__ == "__main__":
    generate_passing_network()
