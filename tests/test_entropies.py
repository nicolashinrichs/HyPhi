"""
Known-Answer Tests for Entropy Estimation
==========================================

Tests entropy estimators against analytically known values
using synthetic distributions.
"""

import numpy as np
import pytest
from hyphi.curvatures import compute_frc
from hyphi.entropies import (
    entropy_vasicek,
    entropy_kde_plugin,
    vec_entropy,
    get_quantiles,
    vec_quantiles,
)


class TestEntropyEstimation:
    """Tests for differential entropy estimators."""

    def _make_graph_with_known_curvatures(self, curvatures):
        """Helper: build a graph and manually set curvature values.

        Creates a path graph and overrides edge curvatures with provided values.
        """
        import networkx as nx
        n_edges = len(curvatures)
        G = nx.path_graph(n_edges + 1)

        for idx, (u, v) in enumerate(G.edges()):
            G[u][v]["formanCurvature"] = curvatures[idx]

        return G

    def test_vasicek_uniform_distribution(self):
        """Uniform[0,1] → theoretical entropy = 0.
        Vasicek estimate should be close to 0.
        """
        np.random.seed(42)
        curvatures = np.random.uniform(0, 1, 500)
        G = self._make_graph_with_known_curvatures(curvatures)
        H = entropy_vasicek(G)
        # Uniform[0,1] has differential entropy = ln(1) = 0
        assert abs(H - 0.0) < 0.3  # generous tolerance for finite sample

    def test_vasicek_gaussian_distribution(self):
        """N(0,1) → theoretical entropy ≈ 1.4189 (½ ln(2πe)).
        Check Vasicek estimate is in the right ballpark.
        """
        np.random.seed(42)
        curvatures = np.random.randn(1000)
        G = self._make_graph_with_known_curvatures(curvatures)
        H = entropy_vasicek(G)
        expected = 0.5 * np.log(2 * np.pi * np.e)  # ≈ 1.4189
        assert abs(H - expected) < 0.3

    def test_kde_plugin_gaussian(self):
        """KDE plugin on N(0,1) should approximate ½ ln(2πe)."""
        np.random.seed(42)
        curvatures = np.random.randn(1000)
        G = self._make_graph_with_known_curvatures(curvatures)
        H = entropy_kde_plugin(G)
        expected = 0.5 * np.log(2 * np.pi * np.e)
        assert abs(H - expected) < 0.3


class TestQuantiles:
    """Tests for quantile extraction."""

    def _make_graph_with_known_curvatures(self, curvatures):
        import networkx as nx
        n_edges = len(curvatures)
        G = nx.path_graph(n_edges + 1)
        for idx, (u, v) in enumerate(G.edges()):
            G[u][v]["formanCurvature"] = curvatures[idx]
        return G

    def test_quantiles_symmetric_distribution(self):
        """For a symmetric distribution centred at 0, median should be ≈ 0."""
        np.random.seed(42)
        curvatures = np.random.randn(1000)
        G = self._make_graph_with_known_curvatures(curvatures)
        qs = get_quantiles(G, qs=[0.25, 0.5, 0.75])
        assert abs(qs[1]) < 0.15  # median near 0

    def test_vec_quantiles(self):
        """vec_quantiles should work on multiple graphs."""
        np.random.seed(42)
        G1 = self._make_graph_with_known_curvatures(np.random.randn(200))
        G2 = self._make_graph_with_known_curvatures(np.random.randn(200))
        result = vec_quantiles([G1, G2], qs=[0.5])
        assert result.shape == (2, 1)


class TestVecEntropy:
    """Tests for vectorised entropy."""

    def _make_graph_with_known_curvatures(self, curvatures):
        import networkx as nx
        n_edges = len(curvatures)
        G = nx.path_graph(n_edges + 1)
        for idx, (u, v) in enumerate(G.edges()):
            G[u][v]["formanCurvature"] = curvatures[idx]
        return G

    def test_vec_entropy_returns_array(self):
        """vec_entropy should return an ndarray of length n_graphs."""
        np.random.seed(42)
        graphs = [self._make_graph_with_known_curvatures(np.random.randn(100))
                   for _ in range(3)]
        result = vec_entropy(graphs, estimator=entropy_vasicek)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
