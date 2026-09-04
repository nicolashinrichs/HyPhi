"""
Known-Answer Tests for Graph Laplacian and Eigenvalue Helpers.

Covers :func:`hyphi.spectral.laplace.laplace` and
:func:`hyphi.spectral.laplace.eigen_in_time`.

Mathematical reference used throughout
---------------------------------------
Complete graph K_n (n nodes, every node connected to every other):
  - Each node has degree n-1.
  - Laplacian L = (n-1)*I - A.
  - Eigenvalues: 0 (multiplicity 1) and n (multiplicity n-1).
  - Algebraic connectivity (Fiedler value, lambda_2) = n.

Disconnected graph with k components:
  - Exactly k zero eigenvalues.

Connected graph:
  - Smallest eigenvalue is exactly 0.
"""

# %% Import
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless: must be set before any pyplot import

import networkx as nx
import numpy as np
from hyphi.spectral.laplace import eigen_in_time, laplace

# %% Constants >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

_K5_N = 5
_K5_FIEDLER = 5.0  # second-smallest eigenvalue of K_5 Laplacian = n = 5
_K5_LARGE_EIG = 5.0  # all non-zero eigenvalues of K_5 Laplacian equal n = 5
_N_EIG_K5 = 5  # total eigenvalues = number of nodes

_C4_N = 4  # 4-node graph used in isolated tests
_N_COMPONENTS_TWO = 2  # two connected components
_N_COMPONENTS_FOUR = 4  # four isolated nodes (= _C4_N)
_N_RETURN_ARRAYS = 2  # eigen_in_time returns (lambdas, gaps)
_ATOL = 1e-10  # absolute tolerance for eigenvalue comparisons

_TIME_STEPS = 4  # number of matrices in the eigen_in_time smoke sequence
_FS_DEFAULT = 1  # default sampling frequency

# %% Test helpers >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _adj(G: nx.Graph) -> np.ndarray:
    """Return the unweighted adjacency matrix of *G* as a float64 ndarray."""
    return nx.to_numpy_array(G, dtype=float)


# %% Test classes >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestLaplaceReturnShape:
    """laplace() returns three objects with consistent shapes."""

    def test_returns_three_objects(self, complete_graph_k5):
        """laplace() must return a 3-tuple (eigenvalues, eigenvectors, L)."""
        result = laplace(_adj(complete_graph_k5))
        assert len(result) == _K5_N - _K5_N + 3  # explicit 3

    def test_eigenvalues_length_equals_n_nodes(self, complete_graph_k5):
        """Number of eigenvalues must equal the number of nodes."""
        eigenvalues, _, _ = laplace(_adj(complete_graph_k5))
        assert len(eigenvalues) == _K5_N

    def test_eigenvectors_shape(self, complete_graph_k5):
        """Eigenvector matrix must be (n, n)."""
        _, eigenvectors, _ = laplace(_adj(complete_graph_k5))
        assert eigenvectors.shape == (_K5_N, _K5_N)

    def test_laplacian_shape(self, complete_graph_k5):
        """Returned Laplacian must be square (n x n)."""
        _, _, L = laplace(_adj(complete_graph_k5))
        assert L.shape == (_K5_N, _K5_N)


class TestLaplaceKnownAnswerK5:
    """Known-answer tests for the complete graph K_5.

    L(K_n) has eigenvalues: 0 (x1) and n (x(n-1)).
    For K_5: one zero and four fives.
    """

    def test_smallest_eigenvalue_is_zero(self, complete_graph_k5):
        """Smallest eigenvalue of K_5 Laplacian must be 0 (connected graph)."""
        eigenvalues, _, _ = laplace(_adj(complete_graph_k5))
        assert abs(eigenvalues[0]) < _ATOL

    def test_fiedler_value_equals_n(self, complete_graph_k5):
        """Second-smallest eigenvalue of K_5 Laplacian must equal n = 5."""
        eigenvalues, _, _ = laplace(_adj(complete_graph_k5))
        assert abs(eigenvalues[1] - _K5_FIEDLER) < _ATOL

    def test_all_nonzero_eigenvalues_equal_n(self, complete_graph_k5):
        """All non-zero eigenvalues of K_5 Laplacian must equal n = 5."""
        eigenvalues, _, _ = laplace(_adj(complete_graph_k5))
        np.testing.assert_allclose(eigenvalues[1:], _K5_LARGE_EIG, atol=_ATOL)

    def test_eigenvalues_sorted_ascending(self, complete_graph_k5):
        """laplace() guarantees eigenvalues are sorted in ascending order."""
        eigenvalues, _, _ = laplace(_adj(complete_graph_k5))
        assert np.all(np.diff(eigenvalues) >= -_ATOL)

    def test_exactly_one_zero_eigenvalue(self, complete_graph_k5):
        """K_5 is connected: exactly one zero eigenvalue."""
        eigenvalues, _, _ = laplace(_adj(complete_graph_k5))
        n_zeros = int(np.sum(np.abs(eigenvalues) < _ATOL))
        assert n_zeros == 1


class TestLaplaceConnectedGraphProperties:
    """Generic properties that hold for any connected graph."""

    def test_ring_lattice_smallest_eigenvalue_is_zero(self, ring_lattice_c10):
        """Smallest eigenvalue of a connected ring graph must be 0."""
        eigenvalues, _, _ = laplace(_adj(ring_lattice_c10))
        assert abs(eigenvalues[0]) < _ATOL

    def test_star_graph_smallest_eigenvalue_is_zero(self, star_graph_s6):
        """Smallest eigenvalue of a connected star graph must be 0."""
        eigenvalues, _, _ = laplace(_adj(star_graph_s6))
        assert abs(eigenvalues[0]) < _ATOL

    def test_laplacian_row_sums_are_zero(self, complete_graph_k5):
        """Row sums of any graph Laplacian must all be 0 (L * 1 = 0)."""
        _, _, L = laplace(_adj(complete_graph_k5))
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=_ATOL)

    def test_laplacian_is_symmetric(self, ring_lattice_c10):
        """The Laplacian of an undirected graph must be symmetric."""
        _, _, L = laplace(_adj(ring_lattice_c10))
        np.testing.assert_allclose(L, L.T, atol=_ATOL)


class TestLaplaceDisconnectedGraph:
    """Disconnected graphs: number of zero eigenvalues equals number of components."""

    def test_two_components_yield_two_zero_eigenvalues(self):
        """A graph with 2 connected components has exactly 2 zero eigenvalues."""
        # Two disjoint K_2 edges: 4 nodes, 2 components
        G = nx.Graph()
        G.add_nodes_from(range(_C4_N))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        eigenvalues, _, _ = laplace(_adj(G))
        n_zeros = int(np.sum(np.abs(eigenvalues) < _ATOL))
        assert n_zeros == _N_COMPONENTS_TWO

    def test_isolated_nodes_yield_n_zero_eigenvalues(self):
        """A fully disconnected graph (no edges) has n zero eigenvalues."""
        G = nx.Graph()
        G.add_nodes_from(range(_C4_N))
        eigenvalues, _, _ = laplace(_adj(G))
        n_zeros = int(np.sum(np.abs(eigenvalues) < _ATOL))
        assert n_zeros == _N_COMPONENTS_FOUR


class TestEigenInTime:
    """Tests for eigen_in_time over a temporal sequence of adjacency matrices."""

    def _make_sequence(self, n_steps: int) -> np.ndarray:
        """Build a length-n_steps sequence of K_5 adjacency matrices."""
        A = _adj(nx.complete_graph(_K5_N))
        return np.stack([A] * n_steps)

    def test_returns_two_arrays(self):
        """eigen_in_time must return a 2-tuple (lambdas, gaps)."""
        seq = self._make_sequence(_TIME_STEPS)
        result = eigen_in_time(seq, plot=False, Fs=_FS_DEFAULT)
        assert len(result) == _N_RETURN_ARRAYS

    def test_lambdas_shape(self):
        """Lambdas array length must equal the number of time steps."""
        seq = self._make_sequence(_TIME_STEPS)
        lambdas, _ = eigen_in_time(seq, plot=False, Fs=_FS_DEFAULT)
        assert lambdas.shape == (_TIME_STEPS,)

    def test_gaps_shape(self):
        """Gaps array length must equal the number of time steps."""
        seq = self._make_sequence(_TIME_STEPS)
        _, gaps = eigen_in_time(seq, plot=False, Fs=_FS_DEFAULT)
        assert gaps.shape == (_TIME_STEPS,)

    def test_fiedler_value_matches_k5(self):
        """Fiedler value (lambdas) for a K_5 sequence must equal n = 5 at every step."""
        seq = self._make_sequence(_TIME_STEPS)
        lambdas, _ = eigen_in_time(seq, plot=False, Fs=_FS_DEFAULT)
        np.testing.assert_allclose(lambdas, _K5_FIEDLER, atol=_ATOL)

    def test_gaps_equal_negative_fiedler(self):
        """Gap is eigenvalues[0] - eigenvalues[1].

        For K_5: eigenvalues[0] = 0, eigenvalues[1] = 5, so gap = -5 everywhere.
        """
        _expected_gap = -_K5_FIEDLER
        seq = self._make_sequence(_TIME_STEPS)
        _, gaps = eigen_in_time(seq, plot=False, Fs=_FS_DEFAULT)
        np.testing.assert_allclose(gaps, _expected_gap, atol=_ATOL)

    def test_plot_smoke(self):
        """eigen_in_time with plot=True must run without error (headless Agg backend)."""
        seq = self._make_sequence(_TIME_STEPS)
        eigen_in_time(seq, plot=True, Fs=_FS_DEFAULT)
        plt.close("all")  # clean up figure state

    def test_single_timestep(self):
        """eigen_in_time works for a sequence of length 1."""
        A = _adj(nx.complete_graph(_K5_N))
        seq = np.stack([A])
        lambdas, gaps = eigen_in_time(seq, plot=False)
        assert lambdas.shape == (1,)
        assert gaps.shape == (1,)
        assert abs(lambdas[0] - _K5_FIEDLER) < _ATOL


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
