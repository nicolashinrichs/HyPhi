"""
Known-Answer Tests for FRC / AFRC Curvature Computation.

Uses lightweight toy graphs where curvature values are
mathematically derivable.

Forman-Ricci (unweighted, 1d method):
  FRC(e) = 4 - deg(u) - deg(v)   for edge e = (u, v)
"""

# %% Import
import matplotlib
import networkx as nx
import numpy as np
import pytest
from hyphi.modeling.curvatures import sim_graph
from hyphi.modeling.graph_curvatures import (
    compute_afrc,
    compute_frc,
    compute_frc_vec,
    extract_curvature_matrices,
    extract_curvatures,
    extract_curvatures_vec,
)

matplotlib.use("Agg")  # headless: no display needed for the viz smoke test
from hyphi.visualization.curvature_visualization import visualize_graph_partitions_markers

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestFormanRicci:
    """Known-answer tests for Forman-Ricci curvature."""

    def test_frc_complete_graph(self, complete_graph_k5):
        """On K_5, every edge should have FRC = 4 - 4 - 4 = -4."""
        G = compute_frc(complete_graph_k5)
        curvatures = extract_curvatures(G, curvature="formanCurvature")
        assert len(curvatures) == 10  # C(5,2) = 10 edges
        np.testing.assert_array_almost_equal(curvatures, -4.0)

    def test_frc_ring_lattice(self, ring_lattice_c10):
        """On C_10, every edge should have FRC = 4 - 2 - 2 = 0."""
        G = compute_frc(ring_lattice_c10)
        curvatures = extract_curvatures(G, curvature="formanCurvature")
        assert len(curvatures) == 10  # 10 edges in a 10-cycle
        np.testing.assert_array_almost_equal(curvatures, 0.0)


class TestSimGraph:
    """The knn graph builder must not emit self-loops."""

    def test_sim_graph_has_no_self_loops(self):
        """A KNN similarity graph must contain zero self-loops."""
        # Regression: an unguarded branch re-added each edge unconditionally,
        # so self-loops survived despite the i != neighbor check.
        n_nodes = 12
        rng = np.random.default_rng(0)
        d = rng.uniform(0.1, 1.0, size=(n_nodes, n_nodes))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0.0)
        graph, _ = sim_graph(d, k=4)
        assert nx.number_of_selfloops(graph) == 0
        assert graph.number_of_nodes() == n_nodes


class TestPartitionVisualization:
    """Smoke-cover the partition plot wired into the ricci-flow CLI."""

    def test_forman_graph_renders_via_curvature_branch(self, tmp_path, complete_graph_k5):
        """A Forman graph (formanCurvature, no ricciCurvature) must still color edges by curvature."""
        # Regression: the plot read only ricciCurvature, so a Forman flow graph
        # fell back to plain gray edges (a curvature-blind "curvature" plot).
        graph = compute_frc(complete_graph_k5)
        assert not nx.get_edge_attributes(graph, "ricciCurvature")
        assert nx.get_edge_attributes(graph, "formanCurvature")
        partitions = [sorted(c) for c in nx.connected_components(graph)]
        visualize_graph_partitions_markers(
            graph=graph, partitions=partitions, name="frc_smoke", save=True, save_path=tmp_path, show=False
        )
        out = tmp_path / "frc_smoke.png"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_frc_star_graph(self, star_graph_s6):
        """On S_6 (centre deg=5, leaf deg=1), each edge FRC = 4 - 5 - 1 = -2."""
        G = compute_frc(star_graph_s6)
        curvatures = extract_curvatures(G, curvature="formanCurvature")
        assert len(curvatures) == 5  # 5 edges
        np.testing.assert_array_almost_equal(curvatures, -2.0)

    def test_frc_vec(self, complete_graph_k5, ring_lattice_c10):
        """Vectorised FRC should return correct results for multiple graphs."""
        graphs = [complete_graph_k5, ring_lattice_c10]
        result = compute_frc_vec(graphs)
        assert len(result) == 2

        curvs_k5 = extract_curvatures(result[0])
        np.testing.assert_array_almost_equal(curvs_k5, -4.0)

        curvs_c10 = extract_curvatures(result[1])
        np.testing.assert_array_almost_equal(curvs_c10, 0.0)


class TestAugmentedFormanRicci:
    """Known-answer tests for Augmented Forman-Ricci curvature."""

    def test_afrc_complete_graph(self, complete_graph_k5):
        """AFRC on K_5 — verify it runs and produces edge attributes."""
        G = compute_afrc(complete_graph_k5)
        curvatures = extract_curvatures(G, curvature="formanCurvature")
        assert len(curvatures) == 10
        # AFRC on K_5: all edges have identical curvature by symmetry
        assert np.std(curvatures) < 1e-10

    def test_afrc_ring_lattice(self, ring_lattice_c10):
        """AFRC on C_10 — all edges should yield identical curvature."""
        G = compute_afrc(ring_lattice_c10)
        curvatures = extract_curvatures(G, curvature="formanCurvature")
        assert len(curvatures) == 10
        assert np.std(curvatures) < 1e-10


class TestCurvatureExtraction:
    """Tests for curvature extraction helpers."""

    def test_extract_curvatures_vec(self, complete_graph_k5, star_graph_s6):
        """extractCurvaturesVec returns list of arrays."""
        graphs = compute_frc_vec([complete_graph_k5, star_graph_s6])
        curvs = extract_curvatures_vec(graphs)
        assert len(curvs) == 2
        assert len(curvs[0]) == 10  # K5
        assert len(curvs[1]) == 5  # S6

    def test_extract_curvature_matrices(self, ring_lattice_c10):
        """extract_curvature_matrices should return (n_graphs, n_nodes, n_nodes)."""
        graphs = compute_frc_vec([ring_lattice_c10])
        mats = extract_curvature_matrices(graphs)
        assert mats.shape == (1, 10, 10)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
