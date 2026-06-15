"""
Heat-diffusion distance between two graphs and edge-deletion perturbation (EDP).

This is the spectral counterpart to the FRC-weighted GDD pipeline in
:mod:`hyphi.modeling.GDD_FRc_helpers`: instead of computing a single closed-form
heat-kernel distance, this module sweeps a time axis ``[0, time_limit]`` at
sampling rate ``fs`` and returns the *maximum* Frobenius difference between the
two heat operators.  Useful when you want to detect the timescale at which two
networks are most distinguishable, not just an aggregate scalar.
"""

import numpy as np
import scipy.linalg as la

import hyphi.spectral.laplace as lap


def main():
    """Return 0 as the module entry-point placeholder."""
    return 0


def diffusion_distance(adj1: np.ndarray, adj2: np.ndarray, time_limit: float, fs: float) -> float:
    """
    Compute the heat-diffusion distance (d_gdd) between two equally shaped square matrices.

    Sweeps the time axis ``[0, time_limit)`` at sampling rate ``fs`` and returns the
    maximum Frobenius difference between the two graphs' heat operators over that window.

    Parameters
    ----------
    adj1 : numpy.ndarray
        First adjacency matrix.
    adj2 : numpy.ndarray
        Second adjacency matrix, matching ``adj1`` in shape.
    time_limit : float
        Upper bound (in seconds) of the time window analyzed to find the maximum
        distance between the two heat distributions.
    fs : float
        Sampling rate: how many timepoints per second are analyzed.

    Returns
    -------
    float
        The maximum diffusion distance (d_gdd) over the time window ``[0, time_limit)``.

    """

    def frobenius(A):
        return la.norm(A, ord="fro")

    def heat_operator(t, eigvals, eigvecs):
        # lap.laplace uses eigh on the symmetric Laplacian, so its eigenvectors are orthonormal
        # and the heat kernel is exp(-t * L) = V @ diag(exp(-t * lambda)) @ V.T. Build the matrix
        # exponential from the (1-D) eigenvalues rather than calling expm on the eigenvalue vector.
        # The decaying sign matches hyphi.modeling.curvatures.heat_kernel_distance.
        return eigvecs @ np.diag(np.exp(-t * eigvals)) @ eigvecs.T

    time = np.arange(0, time_limit, 1 / fs)

    eigvals1, eigvecs1, _ = lap.laplace(adj1)
    eigvals2, eigvecs2, _ = lap.laplace(adj2)

    distances = np.array(
        [frobenius(heat_operator(t, eigvals1, eigvecs1) - heat_operator(t, eigvals2, eigvecs2)) for t in time]
    )

    return np.max(distances)


def edge_deletion(A, i, j):
    """
    Return a copy of adjacency matrix ``A`` with the edge between nodes ``i`` and ``j`` removed.

    Parameters
    ----------
    A : numpy.ndarray
        The original adjacency matrix.
    i : int
        First endpoint of the edge to remove.
    j : int
        Second endpoint of the edge to remove.

    Returns
    -------
    numpy.ndarray
        A copy of ``A`` with entries ``[i, j]`` and ``[j, i]`` set to 0.

    """
    A_prime = np.copy(A)

    A_prime[j, i] = 0
    A_prime[i, j] = 0

    return A_prime


def EDP(A, m, n, fs):
    """
    Compute the normalized edge-deletion perturbation (EDP) for edge ``(m, n)``.

    Deletes the edge ``(m, n)`` from ``A`` and returns the diffusion distance between the
    original and perturbed graphs, normalized by the edge weight ``A[m, n]``. If edge
    ``(m, n)`` is absent (``A[m, n] == 0``), ``chi`` is 0 and ``A_prime`` is an unmodified
    copy of ``A`` (so callers can rely on a valid matrix in either branch).

    Parameters
    ----------
    A : numpy.ndarray
        The original adjacency matrix.
    m : int
        First endpoint of the edge to perturb.
    n : int
        Second endpoint of the edge to perturb.
    fs : float
        Sampling rate passed to :func:`diffusion_distance` (timepoints per second).

    Returns
    -------
    tuple[float, numpy.ndarray]
        The normalized perturbation ``chi`` and the perturbed adjacency matrix
        ``A_prime``.

    """
    if A[m, n]:
        A_prime = edge_deletion(A, m, n)
        dgdd = diffusion_distance(A, A_prime, 100, fs)
        chi = dgdd / A[m, n]
    else:
        # No edge to delete — return an untouched copy so callers always
        # receive a valid (n, n) matrix instead of an UnboundLocalError.
        A_prime = np.copy(A)
        chi = 0
    return chi, A_prime


if __name__ == "__main__":
    main()
