import os
import glob
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community

import matplotlib.pyplot as plt

from hyphi.communities_centrality.adjacency_from_pickle import load_pickle_adjacency

#%% Function

def find_centrality(file_path):
    '''
    Calculates matrix, graph, communities, and centrality
    '''
    adj_matrix = load_pickle_adjacency(file_path)
    graph = nx.from_numpy_array(adj_matrix)
    
    # Calculate communities
    found_communities = community.louvain_communities(graph, weight='weight', seed=42)
    node_to_community = {node: i for i, comm in enumerate(found_communities) for node in comm}
            
    # Calculate centrality
    deg_centrality = nx.degree_centrality(graph)
    bet_centrality = nx.betweenness_centrality(graph, weight='weight')
    
    # Build results table
    results = pd.DataFrame({
        'Community': pd.Series(node_to_community),
        'Degree': pd.Series(deg_centrality),
        'Betweenness': pd.Series(bet_centrality)
    })
    
    return adj_matrix, graph, results

#%% Process & save

input_folder = "code/hyphi/communities_centrality/data"
output_folder = "code/hyphi/communities_centrality/results"
os.makedirs(output_folder, exist_ok=True)

input_files = sorted(glob.glob(os.path.join(input_folder, "*.pkl")))

for file_path in input_files:
    adj_matrix, graph, results = find_centrality(file_path)
    
    prefix = os.path.basename(file_path).split('_')[0]
    output_base = f"{prefix}_processed"
    
    with open(os.path.join(output_folder, f"{output_base}_graph.pkl"), "wb") as f:
        pickle.dump(graph, f)

    np.save(os.path.join(output_folder, f"{output_base}_matrix.npy"), adj_matrix)
    results.to_csv(os.path.join(output_folder, f"{output_base}_stats.csv"))
    
    
#%% Graph visualisation

with open("code/hyphi/communities_centrality/results/01_processed_graph.pkl", "rb") as f:
    G = pickle.load(f)
df = pd.read_csv("code/hyphi/communities_centrality/results/01_processed_stats.csv", index_col=0)

#visualising style
node_colors = [df.loc[node, 'Community'] for node in G.nodes()]
node_sizes = [df.loc[node, 'Betweenness'] * 5000 + 50 for node in G.nodes()]
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, 
        node_color=node_colors, 
        node_size=node_sizes, 
        cmap='tab20', 
        with_labels=False,
        edge_color='gray',
        alpha=0.8,
        width=0.5)

plt.axis('off')
plt.show()