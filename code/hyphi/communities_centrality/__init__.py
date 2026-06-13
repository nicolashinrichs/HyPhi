"""
Community detection and centrality subpackage.

Wraps Louvain community detection (NetworkX) and degree/betweenness centrality
into a per-graph result table, with helpers for stacking per-window adjacency
matrices into a temporal tensor for hyper-brain state analyses.
"""

# %% Imports
from .adjacency_from_pickle import load_pickle_adjacency
from .processing import find_centrality, plot_processed_graph, process_folder
from .temporal_tensor import create_temporal_tensor, plot_state_stability_heatmap

__all__ = [
    "create_temporal_tensor",
    "find_centrality",
    "load_pickle_adjacency",
    "plot_processed_graph",
    "plot_state_stability_heatmap",
    "process_folder",
]
