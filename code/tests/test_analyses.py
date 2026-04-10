"""Tests for the `hyphi.analyses` module."""

# %% Import
import networkx as nx
import numpy as np
import pytest
from hyphi.analyses import (
    build_sliding_window_graphs,
    compute_entropy_kde_plugin,
    compute_frc,
    compute_orc,
    extract_curvatures,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Test Functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_frc_complete_graph(complete_graph):
    # FRC on unweighted complete graph $K_n$ is $(4 - n)$
    # For K_5, 4 - 5 = -1
    G_curv = compute_frc(complete_graph, method="1d")
    curvatures = extract_curvatures(G_curv, "formanCurvature")
    # All edges should have curvature -1
    np.testing.assert_allclose(curvatures, -1.0)


def test_frc_ring_lattice(ring_lattice):
    # FRC on unweighted cycle graph is 0 for all edges
    G_curv = compute_frc(ring_lattice, method="1d")
    curvatures = extract_curvatures(G_curv, "formanCurvature")
    np.testing.assert_allclose(curvatures, 0.0)


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
