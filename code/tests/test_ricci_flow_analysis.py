"""
Tests for the Forman-Ricci flow.

forman_ricci_flow evolves edge weights proportionally to Forman curvature, so a
curvature-annotated graph's geometry is smoothed over iterations. These tests check that it
runs, updates weights, is deterministic, and does not crash on degenerate graphs.
"""

# %% Import
import networkx as nx
import numpy as np
import pytest
from hyphi.modeling import forman_ricci_flow

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _small_world():
    return nx.watts_strogatz_graph(20, 4, 0.3, seed=0)


class TestFormanRicciFlow:
    """The Forman-Ricci flow updates weights, converges, and is deterministic."""

    def test_returns_graph_with_curvature_and_updated_weights(self):
        """The flow returns a graph carrying Forman curvature and modified edge weights."""
        graph = _small_world()
        out = forman_ricci_flow(graph, iterations=5, step=0.1, delta=1e-3, method="1d")
        assert isinstance(out, nx.Graph)
        assert out.number_of_edges() == graph.number_of_edges()
        assert nx.get_edge_attributes(out, "formanCurvature")
        # The default initial weight is 1.0; the flow should move at least one weight off 1.0.
        weights = list(nx.get_edge_attributes(out, "weight").values())
        assert any(w != pytest.approx(1.0) for w in weights)

    def test_records_original_frc(self):
        """The flow snapshots the pre-flow curvature under original_FRC."""
        out = forman_ricci_flow(_small_world(), iterations=3, step=0.1, delta=1e-3, method="1d")
        assert nx.get_edge_attributes(out, "original_FRC")

    def test_is_deterministic(self):
        """The flow has no randomness, so the same input gives identical weights."""
        a = forman_ricci_flow(_small_world(), iterations=4, step=0.1, delta=1e-3, method="1d")
        b = forman_ricci_flow(_small_world(), iterations=4, step=0.1, delta=1e-3, method="1d")
        wa = nx.get_edge_attributes(a, "weight")
        wb = nx.get_edge_attributes(b, "weight")
        assert wa.keys() == wb.keys()
        np.testing.assert_allclose(list(wa.values()), [wb[k] for k in wa], rtol=0, atol=0)

    def test_does_not_mutate_input(self):
        """The flow operates on a copy and leaves the input graph untouched."""
        graph = _small_world()
        before = dict(nx.get_edge_attributes(graph, "weight"))
        forman_ricci_flow(graph, iterations=3, step=0.1, delta=1e-3, method="1d")
        assert dict(nx.get_edge_attributes(graph, "weight")) == before

    def test_single_edge_graph(self):
        """A single-edge graph runs without error."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        out = forman_ricci_flow(graph, iterations=3, step=0.1, delta=1e-3, method="1d")
        assert out.number_of_edges() == 1

    def test_edgeless_graph(self):
        """An edgeless graph does not crash and returns no edges."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        out = forman_ricci_flow(graph, iterations=3, step=0.1, delta=1e-3, method="1d")
        assert out.number_of_edges() == 0


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
