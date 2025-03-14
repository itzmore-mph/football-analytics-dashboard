import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Define Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "../data/processed_shots.csv")

# Ensure Data Exists
if not os.path.exists(DATA_PATH):
    print(f"Error: Data file {DATA_PATH} not found.")
    exit()

# Load Data
df = pd.read_csv(DATA_PATH)

# Validate Required Columns
required_columns = ["passer", "receiver", "x", "y"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"Error: Passing data is missing required columns: {missing_cols}")
    exit()

# Create Passing Network
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["passer"], row["receiver"], weight=row.get("pass_count", 1))

# Plot Passing Network
fig, ax = plt.subplots(figsize=(10, 6))
pitch = Pitch(line_color='black')
pitch.draw(ax=ax)

# Position Nodes on the Pitch
pos = {player: (df[df['player'] == player]['x'].mean(), df[df['player'] == player]['y'].mean()) for player in df['player'].unique()}

# Draw Network
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="blue", alpha=0.7, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5, edge_color='black', ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', ax=ax)

plt.title("Passing Network")
plt.show()
