"""
Tests for Sliding-Window PLV Graph Construction.

Verifies windowing mechanics and PLV properties using
synthetic phase signals.
"""

# %% Import
import numpy as np
import pytest
from hyphi.modeling.windowing import sliding_window_plv, compute_plv_matrix, build_graphs_from_matrices

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestPLVMatrix:
    """Tests for the PLV matrix computation."""

    def test_perfectly_locked_phases(self):
        """For identical phase signals, PLV should be 1.0 everywhere."""
        N, T = 5, 200
        # All oscillators have the same phase trajectory
        phases = np.tile(np.linspace(0, 4 * np.pi, T), (N, 1))
        C = compute_plv_matrix(phases)
        assert C.shape == (N, N)
        np.testing.assert_array_almost_equal(C, 1.0, decimal=5)

    def test_plv_symmetry(self):
        """PLV matrix should be symmetric."""
        np.random.seed(42)
        N, T = 6, 300
        phases = np.random.uniform(0, 2 * np.pi, (N, T))
        C = compute_plv_matrix(phases)
        np.testing.assert_array_almost_equal(C, C.T, decimal=10)

    def test_plv_diagonal_is_one(self):
        """Diagonal of PLV matrix should be 1 (self-synchrony)."""
        np.random.seed(42)
        N, T = 4, 100
        phases = np.random.uniform(0, 2 * np.pi, (N, T))
        C = compute_plv_matrix(phases)
        np.testing.assert_array_almost_equal(np.diag(C), 1.0, decimal=10)

    def test_uncorrelated_low_plv(self):
        """For random independent phases, PLV should be low (near 0)."""
        np.random.seed(42)
        N, T = 10, 10000
        phases = np.random.uniform(0, 2 * np.pi, (N, T))
        C = compute_plv_matrix(phases)
        # Off-diagonal elements should be small
        off_diag = C[np.triu_indices(N, k=1)]
        assert np.mean(off_diag) < 0.1


class TestSlidingWindowPLV:
    """Tests for sliding window graph generation."""

    def test_correct_number_of_windows(self):
        """Number of output graphs should match expected window count."""
        N, T = 5, 1000
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (N, T))
        win_size = 200
        win_stride = 100
        graphs = sliding_window_plv(phases, win_size, win_stride)
        expected = (T - win_size) // win_stride + 1
        assert len(graphs) == expected

    def test_graphs_are_networkx(self):
        """All outputs should be NetworkX graphs."""
        import networkx as nx

        N, T = 3, 500
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (N, T))
        graphs = sliding_window_plv(phases, 100, 50)
        for G in graphs:
            assert isinstance(G, nx.Graph)

    def test_graph_node_count(self):
        """Each graph should have N nodes."""
        N, T = 7, 400
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (N, T))
        graphs = sliding_window_plv(phases, 100, 50)
        for G in graphs:
            assert G.number_of_nodes() == N

    def test_transposed_input(self):
        """Should handle (T, N) input by auto-transposing."""
        N, T = 4, 600
        np.random.seed(42)
        phases = np.random.uniform(0, 2 * np.pi, (T, N))  # (T, N) format
        graphs = sliding_window_plv(phases, 100, 50)
        assert len(graphs) > 0
        assert graphs[0].number_of_nodes() == N


class TestBuildGraphsFromMatrices:
    """Tests for adjacency matrix → graph conversion."""

    def test_identity_matrix(self):
        """Identity matrix → graph with self-loops of weight 1."""
        import networkx as nx

        mat = np.eye(3)
        graphs = build_graphs_from_matrices([mat])
        assert len(graphs) == 1
        assert isinstance(graphs[0], nx.Graph)

    def test_symmetric_matrix(self):
        """Symmetric matrix should produce an undirected graph."""
        mat = np.array([[0, 0.5, 0.3], [0.5, 0, 0.8], [0.3, 0.8, 0]])
        graphs = build_graphs_from_matrices([mat])
        G = graphs[0]
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
