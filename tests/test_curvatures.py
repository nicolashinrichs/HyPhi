"""
Known-Answer Tests for FRC / AFRC Curvature Computation
========================================================

Uses lightweight toy graphs where curvature values are
mathematically derivable.

Forman-Ricci (unweighted, 1d method):
  FRC(e) = 4 - deg(u) - deg(v)   for edge e = (u, v)
"""

import numpy as np
import pytest
from hyphi.curvatures import (
    compute_frc,
    compute_frc_vec,
    compute_afrc,
    extract_curvatures,
    extract_curvatures_vec,
    extract_curvature_matrices,
)


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
        assert len(curvs[1]) == 5   # S6

    def test_extract_curvature_matrices(self, ring_lattice_c10):
        """extract_curvature_matrices should return (n_graphs, n_nodes, n_nodes)."""
        graphs = compute_frc_vec([ring_lattice_c10])
        mats = extract_curvature_matrices(graphs)
        assert mats.shape == (1, 10, 10)
