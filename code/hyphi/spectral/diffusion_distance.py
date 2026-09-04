"""
Heat-diffusion distance between two graphs and edge-deletion perturbation (EDP).

This is the spectral counterpart to the FRC-weighted GDD pipeline in
:mod:`hyphi.modeling.gdd_frc_helpers`: instead of computing a single closed-form
heat-kernel distance, this module sweeps a time axis ``[0, time_limit]`` at
sampling rate ``fs`` and returns the *maximum* Frobenius difference between the
two heat operators.  Useful when you want to detect the timescale at which two
networks are most distinguishable, not just an aggregate scalar.
"""

import numpy as np
import scipy.linalg as la

import hyphi.spectral.laplace as lap


def main():
    """Run module self-test; returns 0 on success."""
    return 0


def _frobenius(A: np.ndarray) -> np.floating:
    """Compute the Frobenius norm of matrix A."""
    return la.norm(A, ord="fro")


def _heat_exp(t: float, eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Compute the heat operator ``U exp(-t Lambda) U^-1`` at time ``t``.

    The diffusion (heat) kernel decays as ``exp(-t * Lambda)`` for the
    positive-semidefinite graph Laplacian (consistent with the negative sign
    used in :mod:`hyphi.modeling.gdd_frc_helpers`); the positive sign would grow
    without bound and overflow at realistic timescales. ``laplace`` returns the
    eigenvalues as a 1-D array, so the diagonal matrix is built explicitly
    (``expm`` of a diagonal equals the diagonal of the element-wise ``exp``);
    passing the 1-D array straight to ``scipy.linalg.expm`` would raise because
    it requires a 2-D matrix.
    """
    return eigvecs @ np.diag(np.exp(-t * eigvals)) @ la.inv(eigvecs)


def diffusion_distance(adj1: np.ndarray, adj2: np.ndarray, time_limit: float, fs: float) -> float:
    """Compute heat diffusion distance between two square adjacency matrices.

    Sweeps the time axis ``[0, time_limit)`` at rate ``fs`` and returns the
    maximum Frobenius difference between the two heat operators across that
    window.

    Parameters
    ----------
    adj1 : np.ndarray
        First square adjacency matrix.
    adj2 : np.ndarray
        Second square adjacency matrix, must match ``adj1`` in shape.
    time_limit : float
        Length of the time window (in seconds) over which to search for the
        maximum heat-kernel distance.
    fs : float
        Sampling rate (timepoints per second) controlling grid resolution.

    Returns
    -------
    float
        Maximum diffusion distance ``d_gdd`` over the window ``[0, time_limit)``.
    """
    time = np.arange(0, time_limit, 1 / fs)

    eigvals1, eigvecs1, _ = lap.laplace(adj1)
    eigvals2, eigvecs2, _ = lap.laplace(adj2)

    distances = np.array(
        [_frobenius(_heat_exp(t, eigvals1, eigvecs1) - _heat_exp(t, eigvals2, eigvecs2)) for t in time]
    )

    return np.max(distances)


def edge_deletion(A: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return a copy of adjacency matrix A with the edge between nodes i and j removed.

    Parameters
    ----------
    A : np.ndarray
        Square adjacency matrix.
    i : int
        Index of the first node.
    j : int
        Index of the second node.

    Returns
    -------
    np.ndarray
        Copy of ``A`` with ``A[i, j]`` and ``A[j, i]`` set to zero.
    """
    A_prime = np.copy(A)
    A_prime[j, i] = 0
    A_prime[i, j] = 0
    return A_prime


def EDP(A: np.ndarray, m: int, n: int, Fs: float) -> tuple:  # noqa: N802 (EDP is the established domain acronym for Edge-Deletion Perturbation)
    """Compute the normalized edge-deletion perturbation for edge (m, n).

    If edge ``(m, n)`` is absent (``A[m, n] == 0``), ``chi`` is 0 and
    ``A_prime`` is returned as an unmodified copy of ``A`` so callers can
    rely on a valid matrix in either branch.

    Parameters
    ----------
    A : np.ndarray
        Original square adjacency matrix.
    m : int
        Row index of the edge to evaluate.
    n : int
        Column index of the edge to evaluate.
    Fs : float
        Sampling rate (timepoints per second) passed to :func:`diffusion_distance`.

    Returns
    -------
    chi : float
        Normalized diffusion distance: ``dgdd / A[m, n]`` when the edge
        exists, otherwise 0.
    A_prime : np.ndarray
        Adjacency matrix with edge ``(m, n)`` deleted, or an unmodified copy
        of ``A`` if the edge was absent.
    """
    if A[m, n]:
        A_prime = edge_deletion(A, m, n)
        dgdd = diffusion_distance(A, A_prime, 100, Fs)
        chi = dgdd / A[m, n]
    else:
        # No edge to delete — return an untouched copy so callers always
        # receive a valid (n, n) matrix instead of an UnboundLocalError.
        A_prime = np.copy(A)
        chi = 0
    return chi, A_prime


if __name__ == "__main__":
    main()
