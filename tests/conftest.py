import pytest
import networkx as nx
import numpy as np

@pytest.fixture
def complete_graph():
    """Generates K_5 complete graph."""
    # Add dummy weights because curvature needs them
    G = nx.complete_graph(5)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G

@pytest.fixture
def ring_lattice():
    """Generates 1D ring lattice (C_10)."""
    G = nx.cycle_graph(10)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G

@pytest.fixture
def star_graph():
    """Generates S_5 star graph."""
    G = nx.star_graph(4) # 1 center, 4 points
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G

@pytest.fixture
def conn_matrix():
    """Generates a random 3D connectivity matrix."""
    return np.random.rand(3, 5, 5) # 3 windows, 5 nodes
