"""
Known-Answer Tests for FRC / AFRC Curvature Computation.

Uses lightweight toy graphs where curvature values are
mathematically derivable.

Forman-Ricci (unweighted, 1d method):
  FRC(e) = 4 - deg(u) - deg(v)   for edge e = (u, v)
"""

# %% Import
import warnings

import networkx as nx
import numpy as np
import pytest
from hyphi.modeling.curvatures import compute_laplacian_matrix, sim_graph
from hyphi.modeling.graph_curvatures import (
    compute_afrc,
    compute_frc,
    compute_frc_vec,
    extract_curvature_matrices,
    extract_curvatures,
    extract_curvatures_vec,
)

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


class TestSimGraph:
    """sim_graph builds a kNN graph with no self-loops and a working weight toggle."""

    def _distance_matrix(self):
        # Symmetric distances with a zero diagonal, so each node's nearest neighbour is itself.
        return np.array(
            [
                [0.0, 0.7, 2.0, 3.0],
                [0.7, 0.0, 1.5, 2.5],
                [2.0, 1.5, 0.0, 0.9],
                [3.0, 2.5, 0.9, 0.0],
            ]
        )

    def test_no_self_loops(self):
        """A node listing itself as its nearest neighbour must not create a self-loop."""
        d = self._distance_matrix()
        g_weighted, _ = sim_graph(d, k=2, weight=True)
        g_unweighted, _ = sim_graph(d, k=2, weight=False)
        assert nx.number_of_selfloops(g_weighted) == 0
        assert nx.number_of_selfloops(g_unweighted) == 0

    def test_weight_toggle(self):
        """weight=True keeps the distances; weight=False sets every edge weight to 1."""
        d = self._distance_matrix()
        g_weighted, _ = sim_graph(d, k=2, weight=True)
        g_unweighted, _ = sim_graph(d, k=2, weight=False)
        # Same edge set either way.
        assert set(g_weighted.edges()) == set(g_unweighted.edges())
        # Weighted graph carries real distances (not all 1); unweighted is all 1.
        w_weighted = [data["weight"] for _, _, data in g_weighted.edges(data=True)]
        w_unweighted = [data["weight"] for _, _, data in g_unweighted.edges(data=True)]
        assert any(w != 1 for w in w_weighted)
        assert all(w == 1 for w in w_unweighted)

    def test_no_self_loops_on_random_matrix(self):
        """Regression on a larger random matrix: the self-loop guard holds for both weight toggles."""
        n_nodes = 12
        rng = np.random.default_rng(0)
        d = rng.uniform(0.1, 1.0, size=(n_nodes, n_nodes))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0.0)
        graph, _ = sim_graph(d, k=4, weight=True)
        graph_unweighted, _ = sim_graph(d, k=4, weight=False)
        assert nx.number_of_selfloops(graph) == 0
        assert nx.number_of_selfloops(graph_unweighted) == 0
        assert graph.number_of_nodes() == n_nodes

    def test_distance_matrix_records_kept_edge_distances(self):
        """The returned distance_matrix records the real distance for each kept edge (not all zeros)."""
        d = self._distance_matrix()
        graph, dm = sim_graph(d, k=2, weight=True)
        assert np.any(dm > 0)  # a mutant that never populates distance_matrix would fail here
        for i, j in graph.edges():
            recorded = dm[i, j] if dm[i, j] > 0 else dm[j, i]
            assert recorded == pytest.approx(round(float(d[i, j]), 3))


class TestLaplacianWeightAttribute:
    """compute_laplacian_matrix warns when the requested weight attribute is absent."""

    def test_warns_on_missing_attribute(self):
        """A plain graph has no ricciCurvature, so weighting by it warns about the fallback."""
        graph = nx.path_graph(4)
        with pytest.warns(UserWarning, match="no edge carries the attribute"):
            laplacian = compute_laplacian_matrix(graph, g_weight="ricciCurvature")
        assert laplacian.shape == (4, 4)

    def test_no_warning_when_attribute_present(self):
        """When every edge carries the attribute, there is no fallback and no warning."""
        graph = nx.path_graph(4)
        for u, v in graph.edges():
            graph[u][v]["ricciCurvature"] = 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            laplacian = compute_laplacian_matrix(graph, g_weight="ricciCurvature")
        assert laplacian.shape == (4, 4)

    def test_warns_on_partial_attribute(self):
        """Partial coverage (some edges lack the attribute) still warns: those edges silently fall to 1."""
        graph = nx.path_graph(4)  # edges (0,1), (1,2), (2,3)
        graph[0][1]["ricciCurvature"] = 0.5
        graph[1][2]["ricciCurvature"] = 0.5  # edge (2,3) is left unannotated
        with pytest.warns(UserWarning, match="edges lack the attribute"):
            laplacian = compute_laplacian_matrix(graph, g_weight="ricciCurvature")
        assert laplacian.shape == (4, 4)

    def test_weighted_laplacian_honors_attribute(self):
        """The Laplacian uses the named attribute as the edge weight, not a unit weight."""
        graph = nx.path_graph(3)  # edges (0,1), (1,2)
        for u, v in graph.edges():
            graph[u][v]["w"] = 0.5
        weighted = compute_laplacian_matrix(graph, g_weight="w")
        unweighted = compute_laplacian_matrix(graph, g_weight="non")
        # A mutant that ignores g_weight (always unweighted) would make these equal.
        assert not np.allclose(weighted, unweighted)
        expected = np.array([[0.5, -0.5, 0.0], [-0.5, 1.0, -0.5], [0.0, -0.5, 0.5]])
        np.testing.assert_allclose(weighted, expected)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
