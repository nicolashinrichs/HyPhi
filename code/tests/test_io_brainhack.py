"""
Tests for the canonical adjacency-pickle loader (``hyphi.io_brainhack``).

``load_pickle_adjacency`` is the single implementation that the spectral and
communities_centrality ``adjacency_from_pickle`` shims both re-export, so these
round-trip tests are the only coverage that dedup relies on: pickle a graph,
load it back, and assert the matrix matches NetworkX's own conversion.
"""

# %% Import
import pickle

import networkx as nx
import numpy as np
import pytest
from hyphi.io_brainhack import load_pickle_adjacency


def _pickle_graph(graph, tmp_path):
    """Write ``graph`` to a pickle under ``tmp_path`` and return the path."""
    path = tmp_path / "graph.pickle"
    with path.open("wb") as f:
        pickle.dump(graph, f)
    return path


# %% Round-trip >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_round_trip_matches_networkx(tmp_path):
    """A pickled weighted graph loads to exactly NetworkX's own adjacency matrix."""
    g = nx.Graph()
    g.add_edge("a", "b", weight=0.7)
    g.add_edge("b", "c", weight=0.2)
    g.add_edge("a", "c", weight=0.5)
    path = _pickle_graph(g, tmp_path)

    loaded = load_pickle_adjacency(path)
    expected = nx.to_numpy_array(g, nodelist=list(g.nodes()), weight="weight")
    np.testing.assert_allclose(loaded, expected)


def test_custom_weight_key_and_missing_default(tmp_path):
    """A non-default weight_key is honored; edges missing that key default to 1.0."""
    g = nx.Graph()
    g.add_edge("a", "b", plv=0.4)
    g.add_edge("b", "c")  # no 'plv' attribute -> should load as 1.0
    path = _pickle_graph(g, tmp_path)

    loaded = load_pickle_adjacency(path, weight_key="plv")
    nodes = list(g.nodes())
    i_a, i_b, i_c = nodes.index("a"), nodes.index("b"), nodes.index("c")
    assert loaded[i_a, i_b] == pytest.approx(0.4)
    assert loaded[i_b, i_c] == pytest.approx(1.0)


def test_return_nodes_preserves_order(tmp_path):
    """return_nodes=True yields the node order used to index the matrix."""
    g = nx.Graph()
    g.add_edge("x", "y", weight=1.0)
    g.add_edge("y", "z", weight=2.0)
    path = _pickle_graph(g, tmp_path)

    matrix, nodes = load_pickle_adjacency(path, return_nodes=True)
    assert nodes == list(g.nodes())
    assert matrix.shape == (len(nodes), len(nodes))


def test_symmetrize_takes_elementwise_max(tmp_path):
    """symmetrize=True returns max(A, A.T) (exercised via a directed graph's asymmetry)."""
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=0.9)  # one direction only
    path = _pickle_graph(g, tmp_path)

    raw = load_pickle_adjacency(path)
    sym = load_pickle_adjacency(path, symmetrize=True)
    np.testing.assert_allclose(sym, np.maximum(raw, raw.T))
    assert not np.allclose(raw, raw.T)  # the directed input really was asymmetric


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
