"""Tests for the `hyphi.analyses` module."""

# %% Import
import networkx as nx
import pytest
from hyphi.analyses import (
    build_sliding_window_graphs,
    compute_entropy_kde_plugin,
)
from hyphi.modeling.graph_curvatures import compute_frc

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Test Functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_entropy_zero_variance(complete_graph):
    # For K_n, all curvatures map to same value. Entropy should return 0.0
    G_curv = compute_frc(complete_graph, method="1d")
    entropy = compute_entropy_kde_plugin(G_curv)
    assert entropy == 0.0


def test_sliding_window_graphs(conn_matrix):
    graphs = build_sliding_window_graphs(conn_matrix)
    assert len(graphs) == 3
    for g in graphs:
        assert isinstance(g, nx.Graph)
        assert len(g.nodes) == 5


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
