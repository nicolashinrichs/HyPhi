"""
Shared pytest fixtures for HyPhi test suite.

Provides lightweight toy graphs with mathematically known properties.
"""

# %% Import
import networkx as nx
import numpy as np
import pytest

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@pytest.fixture
def complete_graph_k5():
    """
    Complete graph K_5 (5 nodes, each connected to all others).

    Properties:
      - degree of every node = 4
      - Unweighted FRC per edge = 4 - d(u) - d(v) = 4 - 4 - 4 = -4.
    """
    return nx.complete_graph(5)


@pytest.fixture
def ring_lattice_c10():
    """
    Ring lattice C_10 (cycle graph with 10 nodes).

    Properties:
      - degree of every node = 2
      - Unweighted FRC per edge = 4 - 2 - 2 = 0

    """
    return nx.cycle_graph(10)


@pytest.fixture
def star_graph_s6():
    """
    Star graph S_6 (1 centre + 5 leaves, 6 nodes total).

    Properties:
      - Centre node degree = 5
      - Leaf node degree = 1
      - FRC for centre-leaf edge = 4 - 5 - 1 = -2
    """
    return nx.star_graph(5)  # creates star with 6 nodes (0=centre, 1-5=leaves)


@pytest.fixture
def complete_graph():
    """
    Complete graph K_5 used for the zero-variance entropy check.

    Every edge of a complete graph has the same unweighted Forman curvature, so the
    curvature distribution is degenerate (a single value) and any entropy estimator
    of it should collapse to the documented sentinel of 0.0.
    """
    return nx.complete_graph(5)


@pytest.fixture
def conn_matrix():
    """
    Build a small windowed connectivity tensor of shape (3, 5, 5).

    Three windows of a symmetric, zero-diagonal 5-node connectivity matrix, the shape
    `build_sliding_window_graphs` consumes. Deterministic (fixed seed) so the derived
    graphs are reproducible across runs.
    """
    rng = np.random.default_rng(0)
    windows = []
    for _ in range(3):
        a = rng.uniform(0.0, 1.0, size=(5, 5))
        a = (a + a.T) / 2.0  # symmetric (undirected connectivity)
        np.fill_diagonal(a, 0.0)  # no self-connectivity
        windows.append(a)
    return np.stack(windows, axis=0)


# %% Test functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

pass


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
