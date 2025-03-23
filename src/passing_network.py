import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch

data_path = "data/passing_data.csv"

def load_data():
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return None
    
    df = pd.read_csv(data_path)
    required_columns = {'passer', 'receiver', 'x', 'y'}
    
    if not required_columns.issubset(df.columns):
        print(f"Error: Passing data is missing required columns: {required_columns - set(df.columns)}")
        return None
    
    return df

def create_passing_network(df):
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        passer, receiver = row['passer'], row['receiver']
        x, y = row['x'], row['y']
        
        if not G.has_node(passer):
            G.add_node(passer, pos=(x, y))
        
        if not G.has_node(receiver):
            G.add_node(receiver, pos=(x + 5, y))  # Slight offset for better visibility
        
        if G.has_edge(passer, receiver):
            G[passer][receiver]['weight'] += 1
        else:
            G.add_edge(passer, receiver, weight=1)
    
    return G

def plot_passing_network(G):
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 6))
    
    pos = nx.get_node_attributes(G, 'pos')
    node_sizes = [G.degree(n) * 150 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='blue', alpha=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='white', font_weight='bold')
    
    edges = G.edges(data=True)
    edge_widths = [0.5 + (d['weight'] * 0.3) for (_, _, d) in edges]
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=edge_widths, alpha=0.6)
    
    plt.title("Passing Network", fontsize=14, fontweight='bold')
    plt.show()

def main():
    df = load_data()
    if df is None:
        print("Passing network not available.")
        return
    
    G = create_passing_network(df)
    plot_passing_network(G)

if __name__ == "__main__":
    main()

centrality = nx.betweenness_centrality(G, weight="weight")
# Use centrality to scale node sizes
node_sizes = [centrality.get(n, 0.01) * 3000 for n in G.nodes()]
