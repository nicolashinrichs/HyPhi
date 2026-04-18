# -*- coding: utf-8 -*-
import os
import pickle

import pandas as pd
import networkx as nx
from networkx.algorithms import community

import matplotlib.pyplot as plt

from hyphi.io import load_connectivity_data
from hyphi.communities_centrality.adjacency_from_pickle import load_pickle_adjacency

PATH_TO_FILE = "code/hyphi/communities_centrality/time_frame_data/29_prebase_graph.pkl"


#%% #load file into adjacency matrix (NumPy array)

data = load_connectivity_data(PATH_TO_FILE)
adj_matrix = load_pickle_adjacency(PATH_TO_FILE)

#%% #transform matrix to graph & calculate communities

graph = nx.from_numpy_array(adj_matrix)
found_communities = community.louvain_communities(graph, weight='weight', seed=42)

node_to_community = {}
for i, comm in enumerate(found_communities):
    for node in comm:
        node_to_community[node] = i
        
communities = {}
for node, comm_id in node_to_community.items():
    if comm_id not in communities:
        communities[comm_id] = []
    communities[comm_id].append(node)


#%% #calculate centrality values

deg_centrality = nx.degree_centrality(graph)
bet_centrality = nx.betweenness_centrality(graph, weight='weight')

results = pd.DataFrame({
    'Community': pd.Series(node_to_community),
    'Degree': pd.Series(deg_centrality),
    'Betweenness': pd.Series(bet_centrality)
})

results = results.sort_values(by='Betweenness', ascending=False)

#%% #create graph
cmap = plt.get_cmap('tab20')
node_colors = [node_to_community[node] for node in graph.nodes()]

bet_cent = nx.betweenness_centrality(graph, weight='weight')
node_sizes = [v * 5000 + 50 for v in bet_cent.values()]

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(graph, seed=42)

nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=0.8)
nx.draw_networkx_edges(graph, pos, alpha=0.1, edge_color='black')

labels = {node: node for node in graph.nodes() if bet_cent[node] > 0.02}
nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight='bold')

plt.show()


#%% #save processed file
output_folder = "code/hyphi/communities_centrality/results"

save_package = {
    'graph': graph,          # The original NetworkX graph object
    'results_table': results, # The DataFrame with Community, Centrality, etc.
}

full_filename = os.path.basename(PATH_TO_FILE)
prefix = full_filename.split('_')[0]
output_name = f"{prefix}_processed"

with open(os.path.join(output_folder, f"{output_name}.pkl"), "wb") as f:
        pickle.dump(save_package, f)

results.to_csv(os.path.join(output_folder, f"{output_name}.csv"))
