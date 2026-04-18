# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
from networkx.algorithms import community

import matplotlib.pyplot as plt

from hyphi.io import load_connectivity_data
from hyphi.communities_centrality.adjacency_from_pickle import load_pickle_adjacency


#%% #inspect start data

#import pickle

#with open("code/hyphi/communities_centrality/time_frame_data/01_prebase_graph.pkl", "rb") as f:
#    data = pickle.load(f)

#nx.draw(data, with_labels=True)
#plt.show()

#%% #load file into adjacency matrix (NumPy array)

data = load_connectivity_data("code/hyphi/communities_centrality/time_frame_data/01_prebase_graph.pkl")
adj_matrix = load_pickle_adjacency("code/hyphi/communities_centrality/time_frame_data/01_prebase_graph.pkl")

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

#%%
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

#%%

for node_name in list(data.nodes)[:5]:
    print(f"Node: {node_name}")
    print(f"Data: {data.nodes[node_name]}\n")
    
#test  
names = list(data.nodes)
print(f"Index 17 is: {names[17]}")
print(f"Index 19 is: {names[19]}")
print(f"Index 54 is: {names[54]}")

