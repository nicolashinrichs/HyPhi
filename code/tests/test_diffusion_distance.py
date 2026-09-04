"""
Tests for hyphi.spectral.diffusion_distance.

Coverage:
- main()           : returns 0 (self-test sentinel)
- _frobenius()     : Frobenius norm of a matrix
- _heat_exp()      : heat operator U exp(t Lambda) U^-1 (tested with 1-D eigvals)
- edge_deletion()  : removes an edge symmetrically; does not mutate original
- diffusion_distance(): a graph metric -> d(a,a)==0, symmetric, non-negative
- EDP()            : absent-edge branch (chi=0, copy of A returned);
                     present-edge branch returns a finite non-negative chi

Note: all adjacency matrices are built directly as numpy arrays so that the
tests remain independent of networkx and run at toy scale.
"""

import hyphi.spectral.diffusion_distance as dd
import hyphi.spectral.laplace as lap
import numpy as np

# ---------------------------------------------------------------------------
# Named constants (PLR2004 compliance: no bare numeric magic values in asserts)
# ---------------------------------------------------------------------------
_N3 = 3  # size of 3-node graphs used in most tests
_N4 = 4  # size of 4-node graphs used in EDP tests
_T_ZERO = 0.0  # time = 0 -> heat_exp should be the identity
_FROB_345 = 5.0  # Frobenius norm of [[3, 4], [0, 0]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path3() -> np.ndarray:
    """Return the adjacency matrix of a 3-node path graph (P_3)."""
    return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)


def _complete3() -> np.ndarray:
    """Return the adjacency matrix of the complete graph K_3."""
    return np.ones((3, 3), dtype=float) - np.eye(3)


def _laplace_eig(A: np.ndarray) -> tuple:
    """Return ``(eigvals_1d, eigvecs)`` from the graph Laplacian of ``A``."""
    eigvals_1d, eigvecs, _ = lap.laplace(A)
    return eigvals_1d, eigvecs


# ===========================================================================
# TestMain
# ===========================================================================


class TestMain:
    """Tests for the module self-test entry point."""

    def test_main_returns_zero(self):
        """main() is a self-test sentinel; it must return 0 on success."""
        assert dd.main() == 0


# ===========================================================================
# TestFrobenius
# ===========================================================================


class TestFrobenius:
    """Tests for the internal _frobenius helper."""

    def test_known_value(self):
        """Frobenius norm of [[3, 4], [0, 0]] is sqrt(9+16) = 5."""
        A = np.array([[3.0, 4.0], [0.0, 0.0]])
        assert np.isclose(dd._frobenius(A), _FROB_345)

    def test_zero_matrix(self):
        """Frobenius norm of the zero matrix is 0."""
        zeros = np.zeros((_N3, _N3))
        assert dd._frobenius(zeros) == 0.0

    def test_identity(self):
        """Frobenius norm of I_n is sqrt(n)."""
        n = _N3
        assert np.isclose(dd._frobenius(np.eye(n)), np.sqrt(n))

    def test_non_negative(self):
        """Frobenius norm is always non-negative."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((_N3, _N3))
        assert dd._frobenius(A) >= 0.0


# ===========================================================================
# TestHeatExp
# ===========================================================================


class TestHeatExp:
    """Tests for the internal _heat_exp helper.

    _heat_exp takes the 1-D eigenvalue vector from laplace() and builds the
    diagonal heat operator internally.
    """

    def test_at_t0_is_identity(self):
        """At t=0 the heat operator exp(0 * L) should equal the identity."""
        A = _path3()
        eigvals, eigvecs = _laplace_eig(A)
        result = dd._heat_exp(_T_ZERO, eigvals, eigvecs)
        np.testing.assert_allclose(result, np.eye(_N3), atol=1e-10)

    def test_returns_square_matrix(self):
        """_heat_exp returns a square matrix with the same size as the input."""
        A = _path3()
        eigvals, eigvecs = _laplace_eig(A)
        result = dd._heat_exp(0.5, eigvals, eigvecs)
        assert result.shape == (_N3, _N3)

    def test_different_times_differ(self):
        """The heat operator at t=0 and t=1 should not be equal."""
        adj = _path3()
        eigvals, eigvecs = _laplace_eig(adj)
        heat0 = dd._heat_exp(_T_ZERO, eigvals, eigvecs)
        heat1 = dd._heat_exp(1.0, eigvals, eigvecs)
        assert not np.allclose(heat0, heat1)


# ===========================================================================
# TestEdgeDeletion
# ===========================================================================


class TestEdgeDeletion:
    """Tests for edge_deletion().

    edge_deletion(A, i, j) must:
      - zero out A[i, j] and A[j, i] in the returned copy
      - leave all other entries unchanged
      - not mutate the original array
    """

    def test_deleted_entries_are_zero(self):
        """After deletion adj_prime[i,j] and adj_prime[j,i] must be 0."""
        adj = _path3()
        adj_prime = dd.edge_deletion(adj, 0, 1)
        assert adj_prime[0, 1] == 0.0
        assert adj_prime[1, 0] == 0.0

    def test_other_entries_unchanged(self):
        """Entries not involving the deleted edge must remain unchanged."""
        adj = _complete3()
        adj_prime = dd.edge_deletion(adj, 0, 1)
        # Edge (0,2) and (1,2) should still be 1
        assert adj_prime[0, 2] == 1.0
        assert adj_prime[2, 0] == 1.0
        assert adj_prime[1, 2] == 1.0
        assert adj_prime[2, 1] == 1.0

    def test_original_not_mutated(self):
        """edge_deletion must return a copy and not modify the input array."""
        adj = _path3()
        adj_copy = adj.copy()
        _ = dd.edge_deletion(adj, 0, 1)
        np.testing.assert_array_equal(adj, adj_copy)

    def test_delete_absent_edge_is_idempotent(self):
        """Deleting an already-absent edge leaves the matrix unchanged."""
        adj = _path3()  # edge (0,2) does not exist
        adj_prime = dd.edge_deletion(adj, 0, 2)
        np.testing.assert_array_equal(adj_prime, adj)

    def test_returns_ndarray(self):
        """edge_deletion returns a numpy ndarray."""
        adj = _path3()
        adj_prime = dd.edge_deletion(adj, 0, 1)
        assert isinstance(adj_prime, np.ndarray)

    def test_shape_preserved(self):
        """The returned matrix has the same shape as the input."""
        adj = np.eye(_N4, dtype=float)
        adj_prime = dd.edge_deletion(adj, 0, 1)
        assert adj_prime.shape == (_N4, _N4)


# ===========================================================================
# TestDiffusionDistance
# ===========================================================================


class TestDiffusionDistance:
    """Tests for diffusion_distance(): it behaves as a non-negative graph metric."""

    def test_distance_to_self_is_zero(self):
        """The distance between a graph and itself is zero."""
        A = _path3()
        assert dd.diffusion_distance(A, A, 1.0, 10.0) == 0.0

    def test_symmetric(self):
        """d(a, b) == d(b, a)."""
        adj1 = _path3()
        adj2 = _complete3()
        d_ab = dd.diffusion_distance(adj1, adj2, 1.0, 10.0)
        d_ba = dd.diffusion_distance(adj2, adj1, 1.0, 10.0)
        assert np.isclose(d_ab, d_ba)

    def test_distinct_graphs_positive(self):
        """Distinct graphs have a strictly positive, finite distance."""
        adj1 = _path3()
        adj2 = _complete3()
        d = dd.diffusion_distance(adj1, adj2, 1.0, 10.0)
        assert np.isfinite(d)
        assert d > 0.0


# ===========================================================================
# TestEDP
# ===========================================================================


class TestEDP:
    """Tests for EDP() (Edge-Deletion Perturbation).

    Two branches:
    - Absent edge  : chi = 0, A_prime is an unmodified copy of A
    - Present edge : delegates to diffusion_distance(); returns a finite chi >= 0
    """

    def test_absent_edge_chi_is_zero(self):
        """When A[m, n] == 0, EDP returns chi = 0."""
        adj = _path3()  # edge (0, 2) is absent
        chi, _ = dd.EDP(adj, 0, 2, 10.0)
        assert chi == 0

    def test_absent_edge_returns_copy_of_a(self):
        """When A[m, n] == 0, EDP returns an unmodified copy of A."""
        adj = _path3()
        _, adj_prime = dd.EDP(adj, 0, 2, 10.0)
        np.testing.assert_array_equal(adj_prime, adj)

    def test_absent_edge_does_not_mutate_original(self):
        """EDP must not mutate the input matrix even in the absent-edge branch."""
        adj = _path3()
        adj_copy = adj.copy()
        dd.EDP(adj, 0, 2, 10.0)
        np.testing.assert_array_equal(adj, adj_copy)

    def test_present_edge_returns_finite_chi(self):
        """EDP on a present edge returns a finite, non-negative perturbation chi."""
        adj = _path3()  # edge (0, 1) is present
        chi, _ = dd.EDP(adj, 0, 1, 10.0)
        assert np.isfinite(chi)
        assert chi >= 0.0
