"""
Tests for the spectral diffusion / Laplacian helpers.

Covers the heat-diffusion distance and edge-deletion perturbation in
:mod:`hyphi.spectral.diffusion_distance` (the regression for issue #106, where
``diffusion_distance`` crashed by calling ``expm`` on a 1-D eigenvalue array) and
the Laplacian eigendecomposition helpers in :mod:`hyphi.spectral.laplace`, both of
which were previously at low/zero coverage.
"""

import numpy as np
import pytest
from hyphi.spectral.diffusion_distance import EDP, diffusion_distance, edge_deletion
from hyphi.spectral.laplace import eigen_in_time, laplace

# Numerical tolerance for eigenvalue ordering / sign comparisons.
_TOL = 1e-9

# Small deterministic adjacency matrices.
_A_PATH = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)  # path 0-1-2
_A_TRIANGLE = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)  # K3
_A_WEIGHTED = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)  # weighted path


def test_diffusion_distance_runs_and_is_zero_for_identical_graphs():
    """Regression for #106: diffusion_distance no longer crashes, and d(G, G) == 0.

    The bug was ``scipy.linalg.expm`` applied to a 1-D eigenvalue array. The fix builds the
    heat operator from the eigenvalues, so the call must succeed and a graph's distance from
    itself is exactly zero (identical heat operators at every sampled time).
    """
    distance = diffusion_distance(_A_TRIANGLE, _A_TRIANGLE, time_limit=10.0, fs=2.0)
    assert distance == pytest.approx(0.0, abs=1e-9)


def test_diffusion_distance_distinct_graphs_is_positive_and_finite():
    """Distinct graphs have a positive, finite diffusion distance (a pinned known value)."""
    distance = diffusion_distance(_A_PATH, _A_TRIANGLE, time_limit=10.0, fs=2.0)
    assert np.isfinite(distance)
    assert distance > 0.0
    # Known value from the fixed heat-kernel sweep over t in [0, 10) at fs=2 (stable across backends).
    assert distance == pytest.approx(0.3834, abs=2e-3)


def test_diffusion_distance_is_symmetric():
    """The diffusion distance is symmetric in its two graph arguments."""
    forward = diffusion_distance(_A_PATH, _A_TRIANGLE, time_limit=10.0, fs=2.0)
    backward = diffusion_distance(_A_TRIANGLE, _A_PATH, time_limit=10.0, fs=2.0)
    assert forward == pytest.approx(backward, abs=1e-9)


def test_edge_deletion_removes_symmetric_edge_and_returns_a_copy():
    """edge_deletion zeroes both ``(i, j)`` and ``(j, i)`` and does not mutate the input."""
    pruned = edge_deletion(_A_TRIANGLE, 0, 1)

    assert pruned[0, 1] == 0.0
    assert pruned[1, 0] == 0.0
    # Other edges are untouched, and the original matrix is unchanged.
    assert pruned[0, 2] == 1.0
    assert _A_TRIANGLE[0, 1] == 1.0
    assert pruned is not _A_TRIANGLE


def test_edp_present_edge_normalizes_by_edge_weight():
    """For a present edge, EDP returns chi = diffusion_distance / weight and the pruned matrix."""
    chi, pruned = EDP(_A_WEIGHTED, 1, 2, fs=2.0)

    assert pruned[1, 2] == 0.0
    assert pruned[2, 1] == 0.0
    assert np.isfinite(chi)
    assert chi > 0.0
    # chi is the diffusion distance normalized by the edge weight A[1, 2] == 3.
    expected = diffusion_distance(_A_WEIGHTED, pruned, 100, 2.0) / _A_WEIGHTED[1, 2]
    assert chi == pytest.approx(expected, rel=1e-9)


def test_edp_absent_edge_returns_zero_and_unmodified_copy():
    """For an absent edge, EDP returns chi == 0 and an unmodified copy (no UnboundLocalError)."""
    chi, pruned = EDP(_A_PATH, 0, 2, fs=2.0)  # no 0-2 edge in the path

    assert chi == 0
    assert pruned[0, 2] == 0.0
    assert np.array_equal(pruned, _A_PATH)
    assert pruned is not _A_PATH


def test_laplace_returns_psd_laplacian_with_zero_smallest_eigenvalue():
    """Laplace returns L = D - A (symmetric), ascending eigenvalues with a zero mode, orthonormal vectors."""
    eigenvalues, eigenvectors, laplacian = laplace(_A_TRIANGLE)

    degree = np.diag(_A_TRIANGLE.sum(axis=1))
    assert np.allclose(laplacian, degree - _A_TRIANGLE)
    assert np.allclose(laplacian, laplacian.T)
    # Connected graph: smallest Laplacian eigenvalue is 0; eigenvalues are ascending and non-negative.
    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(eigenvalues) >= -_TOL)
    assert np.all(eigenvalues >= -_TOL)
    # eigh yields orthonormal eigenvectors.
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(3))


def test_eigen_in_time_tracks_fiedler_value_and_nonpositive_gap():
    """eigen_in_time returns the per-matrix Fiedler value and the (<= 0) two-smallest gap."""
    lambdas, gaps = eigen_in_time(np.array([_A_PATH, _A_TRIANGLE]))

    assert lambdas.shape == (2,)
    assert gaps.shape == (2,)
    for i, matrix in enumerate((_A_PATH, _A_TRIANGLE)):
        eigenvalues, _, _ = laplace(matrix)
        assert lambdas[i] == pytest.approx(eigenvalues[1])
        assert gaps[i] == pytest.approx(eigenvalues[0] - eigenvalues[1])
    # The gap is the ascending (smallest - second-smallest), so it is non-positive.
    assert np.all(gaps <= _TOL)
