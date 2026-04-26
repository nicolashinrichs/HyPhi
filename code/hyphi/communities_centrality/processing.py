"""
Per-graph community detection and centrality, with batch processing helpers.

The reusable bit is :func:`find_centrality`: pass a pickle path, get back the
adjacency matrix, the NetworkX graph, and a per-node DataFrame with Louvain
community assignment plus degree and betweenness centrality.

The :func:`process_folder` helper applies that to every ``*.pkl`` in a folder
and writes ``<prefix>_processed_{graph.pkl, matrix.npy, stats.csv}`` triples
to an output folder.

The :func:`plot_processed_graph` helper renders a node-community / centrality-
sized plot for a single processed result.

Importing this module has no side effects: previous versions ran the batch
loop and rendered a plot at import time, which broke ``import hyphi``.  Use
``python -m hyphi.communities_centrality.processing`` from the repo root or
call the helpers directly from your own script / notebook.
"""

import glob
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community

from hyphi.communities_centrality.adjacency_from_pickle import load_pickle_adjacency

# %% Function


def find_centrality(file_path):
    """
    Calculates matrix, graph, communities, and centrality.
    """
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
        'Betweenness': pd.Series(bet_centrality),
    })

    return adj_matrix, graph, results


# %% Process & save


def process_folder(
    input_folder="code/hyphi/communities_centrality/data",
    output_folder="code/hyphi/communities_centrality/results",
    pattern="*.pkl",
):
    """
    Run :func:`find_centrality` on every ``pattern`` file in ``input_folder``
    and write the resulting graph / adjacency / stats CSV triples to
    ``output_folder``.  Returns the list of processed prefixes.
    """
    os.makedirs(output_folder, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(input_folder, pattern)))

    processed = []
    for file_path in input_files:
        adj_matrix, graph, results = find_centrality(file_path)

        prefix = os.path.basename(file_path).split('_')[0]
        output_base = f"{prefix}_processed"

        with open(os.path.join(output_folder, f"{output_base}_graph.pkl"), "wb") as f:
            pickle.dump(graph, f)

        np.save(os.path.join(output_folder, f"{output_base}_matrix.npy"), adj_matrix)
        results.to_csv(os.path.join(output_folder, f"{output_base}_stats.csv"))

        processed.append(prefix)

    return processed


# %% Graph visualisation


def plot_processed_graph(
    graph_pkl_path,
    stats_csv_path,
    figsize=(10, 8),
    cmap='tab20',
    seed=42,
    show=False,
):
    """
    Render a community / centrality plot for a single ``find_centrality`` result.

    Parameters
    ----------
    graph_pkl_path
        Path to the ``..._processed_graph.pkl`` produced by :func:`process_folder`.
    stats_csv_path
        Path to the matching ``..._processed_stats.csv``.
    figsize, cmap, seed
        Forwarded to matplotlib / networkx.
    show
        If True, also call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with open(graph_pkl_path, "rb") as f:
        G = pickle.load(f)
    df = pd.read_csv(stats_csv_path, index_col=0)

    node_colors = [df.loc[node, 'Community'] for node in G.nodes()]
    node_sizes = [df.loc[node, 'Betweenness'] * 5000 + 50 for node in G.nodes()]

    fig = plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=cmap,
        with_labels=False,
        edge_color='gray',
        alpha=0.8,
        width=0.5,
    )
    plt.axis('off')
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    prefixes = process_folder()
    print(f"Processed {len(prefixes)} pickle(s).")
    if prefixes:
        first = prefixes[0]
        plot_processed_graph(
            f"code/hyphi/communities_centrality/results/{first}_processed_graph.pkl",
            f"code/hyphi/communities_centrality/results/{first}_processed_stats.csv",
            show=True,
        )
