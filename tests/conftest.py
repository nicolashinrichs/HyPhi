"""
Shared pytest fixtures for HyPhi test suite.
Provides lightweight toy graphs with mathematically known properties.
"""

import sys
import os
import pytest
import networkx as nx

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def complete_graph_k5():
    """Complete graph K_5 (5 nodes, each connected to all others).
    Properties:
      - degree of every node = 4
      - Unweighted FRC per edge = 4 - d(u) - d(v) = 4 - 4 - 4 = -4
    """
    return nx.complete_graph(5)


@pytest.fixture
def ring_lattice_c10():
    """Ring lattice C_10 (cycle graph with 10 nodes).
    Properties:
      - degree of every node = 2
      - Unweighted FRC per edge = 4 - 2 - 2 = 0
    """
    return nx.cycle_graph(10)


@pytest.fixture
def star_graph_s6():
    """Star graph S_6 (1 centre + 5 leaves, 6 nodes total).
    Properties:
      - Centre node degree = 5
      - Leaf node degree = 1
      - FRC for centre-leaf edge = 4 - 5 - 1 = -2
    """
    return nx.star_graph(5)  # creates star with 6 nodes (0=centre, 1-5=leaves)
