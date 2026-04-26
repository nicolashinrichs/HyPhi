"""Tests for the `hyphi.analyses` module."""

# %% Import
import networkx as nx
import numpy as np
import pytest
from hyphi.analyses import (
    build_sliding_window_graphs,
    compute_entropy_kde_plugin,
    prune_graph_by_weight,
    remove_self_loops_copy,
    summarize_network,
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


class TestRemoveSelfLoopsCopy:
    """Self-loop removal returns a new graph without mutating the input."""

    def test_self_loops_dropped(self):
        G = nx.Graph()
        G.add_edges_from([(0, 0), (0, 1), (1, 2), (2, 2)])
        H = remove_self_loops_copy(G)
        assert nx.number_of_selfloops(H) == 0
        # Input untouched
        assert nx.number_of_selfloops(G) == 2

    def test_preserves_regular_edges(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1, {"weight": 0.5}), (0, 0, {"weight": 1.0})])
        H = remove_self_loops_copy(G)
        assert H.number_of_edges() == 1
        assert H[0][1]["weight"] == 0.5


class TestPruneGraphByWeight:
    """Threshold pruning keeps `>= threshold`, drops below."""

    def test_edges_below_threshold_dropped(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.1)
        G.add_edge(1, 2, weight=0.6)
        G.add_edge(2, 3, weight=0.5)
        H = prune_graph_by_weight(G, threshold=0.5, keep_all_nodes=False)
        assert set(H.edges()) == {(1, 2), (2, 3)}

    def test_keep_all_nodes_flag(self):
        G = nx.Graph()
        G.add_nodes_from(range(4))
        G.add_edge(0, 1, weight=0.9)  # keeps nodes 0, 1; leaves 2, 3 isolated
        H_drop = prune_graph_by_weight(G, threshold=0.5, keep_all_nodes=False)
        H_keep = prune_graph_by_weight(G, threshold=0.5, keep_all_nodes=True)
        assert set(H_drop.nodes()) == {0, 1}
        assert set(H_keep.nodes()) == {0, 1, 2, 3}

    def test_preserves_edge_attributes(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.9, custom="keep-me")
        H = prune_graph_by_weight(G, threshold=0.5)
        assert H[0][1]["custom"] == "keep-me"

    def test_returns_same_graph_class(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1.0)
        H = prune_graph_by_weight(G, threshold=0.5)
        assert isinstance(H, nx.DiGraph)


class TestSummarizeNetwork:
    """Descriptive network summary dict."""

    def test_dict_keys_present(self):
        G = nx.erdos_renyi_graph(10, 0.3, seed=0)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
        s = summarize_network(G, show_n=3, verbose=False)
        expected = {
            "directed",
            "n_nodes",
            "n_edges",
            "density",
            "weight_stats",
            "is_complete",
            "top_degree",
            "bottom_degree",
            "top_weighted_degree",
            "n_self_loops",
            "n_isolates",
            "connected",
            "n_components",
            "largest_component_size",
            "component_sizes",
        }
        assert expected.issubset(s.keys())

    def test_complete_graph_flag(self):
        G = nx.complete_graph(6)
        s = summarize_network(G, verbose=False)
        assert s["is_complete"] is True

    def test_isolates_counted(self):
        G = nx.Graph()
        G.add_nodes_from(range(5))
        G.add_edge(0, 1)
        s = summarize_network(G, verbose=False)
        assert s["n_isolates"] == 3

    def test_self_loops_counted(self):
        G = nx.Graph()
        G.add_edge(0, 0)
        G.add_edge(1, 2)
        s = summarize_network(G, verbose=False)
        assert s["n_self_loops"] == 1

    def test_directed_graph_uses_weak_connectivity(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2)])
        s = summarize_network(G, verbose=False)
        assert s["directed"] is True
        assert s["connected"] is True

    def test_verbose_prints_without_error(self, capsys):
        G = nx.path_graph(4)
        summarize_network(G, verbose=True, title="Unit test")
        captured = capsys.readouterr()
        assert "Unit test" in captured.out
        assert "nodes: 4" in captured.out


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
