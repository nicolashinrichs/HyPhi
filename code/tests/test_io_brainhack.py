"""
Tests for the canonical adjacency-pickle loader (``hyphi.io_brainhack``).

``load_pickle_adjacency`` is the single implementation that the spectral and
communities_centrality ``adjacency_from_pickle`` shims both re-export. These tests
cover the contract the dedup relies on: round-trip loading matches NetworkX's own
conversion, the shims are identity re-exports (not stale copies), and the
cross-NetworkX-version compatibility fallback still recovers a graph.
"""

# %% Import
import pickle

import hyphi.communities_centrality.adjacency_from_pickle as comm_mod
import hyphi.io_brainhack as io_mod
import hyphi.spectral.adjacency_from_pickle as spec_mod
import networkx as nx
import numpy as np
import pytest
from hyphi import communities_centrality as comm_pkg
from hyphi import spectral as spec_pkg
from hyphi.io_brainhack import load_pickle_adjacency


def _pickle_graph(graph, tmp_path):
    """Write ``graph`` to a pickle under ``tmp_path`` and return the path."""
    path = tmp_path / "graph.pickle"
    with path.open("wb") as f:
        pickle.dump(graph, f)
    return path


class _ScalarAdjGraph:
    """Graph-like object whose ``_adj`` maps neighbors to bare scalar weights, not attr dicts."""

    # Exercises the loader's non-dict-attr branch. Defined at module level so pickle resolves it.
    def __init__(self, adj):
        self._adj = adj


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
    """symmetrize=True returns max(A, A.T), with data chosen so it differs from a sum bug."""
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=0.9)
    g.add_edge("b", "a", weight=0.3)  # both directions, different weights
    path = _pickle_graph(g, tmp_path)

    raw = load_pickle_adjacency(path)
    sym = load_pickle_adjacency(path, symmetrize=True)
    np.testing.assert_allclose(sym, np.maximum(raw, raw.T))
    # The bidirectional-asymmetric data makes max-symmetrize (0.9) differ from a
    # sum-symmetrize bug (1.2), so the assertion above cannot pass vacuously.
    assert not np.allclose(np.maximum(raw, raw.T), raw + raw.T)
    assert not np.allclose(raw, raw.T)  # the directed input really was asymmetric


# %% Dedup contract and compatibility >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_shims_are_identity_reexports():
    """The shims are identity re-exports of the canonical loader (module and package paths)."""
    # Both the module path (used by processing.py) and the package path (re-exported in each
    # __init__) must resolve to the one canonical object, so a stale duplicate cannot ship green.
    assert spec_mod.load_pickle_adjacency is load_pickle_adjacency
    assert comm_mod.load_pickle_adjacency is load_pickle_adjacency
    assert spec_pkg.load_pickle_adjacency is load_pickle_adjacency
    assert comm_pkg.load_pickle_adjacency is load_pickle_adjacency


def test_list_wrapped_pickle_uses_first_item(tmp_path):
    """A pickle holding a [graph] list loads its first item (the list/tuple-unwrap branch)."""
    g = nx.Graph()
    g.add_edge("a", "b", weight=0.5)
    path = tmp_path / "wrapped.pickle"
    with path.open("wb") as f:
        pickle.dump([g], f)

    loaded = load_pickle_adjacency(path)
    np.testing.assert_allclose(loaded, nx.to_numpy_array(g, weight="weight"))


def test_scalar_edge_attribute_branch(tmp_path):
    """An ``_adj`` whose edge values are bare scalars loads via the non-dict-attr branch."""
    obj = _ScalarAdjGraph({"a": {"b": 0.5}, "b": {"a": 0.5}})
    path = tmp_path / "scalar.pickle"
    with path.open("wb") as f:
        pickle.dump(obj, f)

    loaded = load_pickle_adjacency(path)
    np.testing.assert_allclose(loaded, np.array([[0.0, 0.5], [0.5, 0.0]]))


def test_compat_fallback_recovers_when_fast_unpickle_fails(tmp_path, monkeypatch):
    """The _CompatUnpickler fallback recovers the adjacency when the fast pickle.load path fails."""
    # A raised pickle.load simulates a cross-NetworkX-version class mismatch; the loader must fall
    # through to _compat_load and still recover the matrix (the back-compat path #79 centralizes).
    g = nx.Graph()
    g.add_edge("a", "b", weight=0.7)
    path = _pickle_graph(g, tmp_path)

    calls = {"n": 0}

    def _raising_load(_f):
        calls["n"] += 1
        raise ModuleNotFoundError("simulated cross-networkx-version class path")

    monkeypatch.setattr(io_mod.pickle, "load", _raising_load)
    loaded = load_pickle_adjacency(path)  # must fall through to _compat_load
    np.testing.assert_allclose(loaded, nx.to_numpy_array(g, weight="weight"))
    assert calls["n"] == 1  # the fast path was attempted and rejected


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
