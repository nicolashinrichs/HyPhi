"""Tests for hyphi.io: config loading and the consolidated adjacency-pickle loader."""

# %% Import
import io as _io
import pickle

import networkx as nx
import numpy as np
import pytest
from hyphi.communities_centrality.adjacency_from_pickle import load_pickle_adjacency as _comm_loader
from hyphi.io import CompatUnpickler, load_config, load_network_pkl, load_pickle_adjacency
from hyphi.io_brainhack import load_pickle_adjacency as _brain_loader
from hyphi.spectral.adjacency_from_pickle import load_pickle_adjacency as _spec_loader

_N_NODES = 5  # nodes in the toy cycle graph used across these tests

# %% Tests for load_config >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_compat_unpickler_loads_numpy():
    """CompatUnpickler loads a standard numpy pickle (the numpy._core->numpy.core remap is a no-op here)."""
    arr = np.arange(6, dtype=float).reshape(2, 3)
    raw = pickle.dumps(arr)
    out = CompatUnpickler(_io.BytesIO(raw)).load()
    np.testing.assert_array_equal(out, arr)


def test_load_config_reads_toml(tmp_path):
    """load_config returns the parsed TOML as a dict."""
    cfg = tmp_path / "c.toml"
    cfg.write_text('name = "x"\n[section]\nvalue = 3\n')
    out = load_config(cfg)
    assert out == {"name": "x", "section": {"value": 3}}


# %% Tests for the network/adjacency loaders >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@pytest.fixture
def weighted_cycle_pickle(tmp_path):
    """Pickle a 5-node weighted cycle graph and return its path."""
    graph = nx.cycle_graph(_N_NODES)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 0.5
    path = tmp_path / "graph.pkl"
    with path.open("wb") as fp:
        pickle.dump(graph, fp)
    return path


def test_load_network_pkl_roundtrip(weighted_cycle_pickle):
    """load_network_pkl returns the unpickled graph unchanged."""
    graph = load_network_pkl(weighted_cycle_pickle)
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == _N_NODES


def test_load_pickle_adjacency_recovers_weights(weighted_cycle_pickle):
    """The adjacency matrix carries the edge weights and the right shape."""
    adjacency = load_pickle_adjacency(weighted_cycle_pickle)
    assert adjacency.shape == (_N_NODES, _N_NODES)
    assert np.isclose(adjacency[0, 1], 0.5)
    assert np.isclose(adjacency[1, 0], 0.5)  # undirected: both directions present


def test_load_pickle_adjacency_return_nodes(weighted_cycle_pickle):
    """return_nodes yields the node order alongside the matrix."""
    adjacency, nodes = load_pickle_adjacency(weighted_cycle_pickle, return_nodes=True)
    assert len(nodes) == _N_NODES
    assert adjacency.shape == (_N_NODES, _N_NODES)


def test_load_pickle_adjacency_symmetrize(weighted_cycle_pickle):
    """symmetrize=True returns max(A, A.T)."""
    adjacency = load_pickle_adjacency(weighted_cycle_pickle, symmetrize=True)
    np.testing.assert_allclose(adjacency, adjacency.T)


def test_adjacency_shims_reexport_canonical():
    """The spectral, communities, and brainhack shims re-export the canonical loader object."""
    assert _spec_loader is load_pickle_adjacency
    assert _comm_loader is load_pickle_adjacency
    assert _brain_loader is load_pickle_adjacency


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
