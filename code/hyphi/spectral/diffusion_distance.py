"""
Heat-diffusion distance between two graphs and edge-deletion perturbation (EDP).

This is the spectral counterpart to the FRC-weighted GDD pipeline in
:mod:`hyphi.modeling.GDD_FRc_helpers`: instead of computing a single closed-form
heat-kernel distance, this module sweeps a time axis ``[0, time_limit]`` at
sampling rate ``fs`` and returns the *maximum* Frobenius difference between the
two heat operators.  Useful when you want to detect the time scale at which two
networks are most distinguishable, not just an aggregate scalar.
"""

import numpy as np
import scipy.linalg as la

import hyphi.spectral.laplace as lap


def main():
    return 0


def diffusion_distance(adj1: np.ndarray, adj2: np.ndarray, time_limit: float, fs: float) -> float:
    """
    Function calculating heat diffusion distance (d_gdd) of two square matrices matching in shape.

    Input
    Adjacency matrix 1, adjacency matrix 2, time_limit, fs
    time limit - how many timepoints (in seconds) should be analysed to find the maximum distance of two heat distributions
    fs - how many timepoints in one second should be analysed

    Output
    maximum diffusion distance (d_gdd) in the given time window (0, time_limit)

    """

    frobenius = lambda A: la.norm(A, ord='fro')
    exp = lambda t, eigvals, eigvecs: eigvecs @ la.expm(t * eigvals) @ la.inv(eigvecs)

    time = np.arange(0, time_limit, 1 / fs)

    eigvals1, eigvecs1, _ = lap.laplace(adj1)
    eigvals2, eigvecs2, _ = lap.laplace(adj2)

    diffusion_distance = np.array(
        [frobenius(exp(t, eigvals1, eigvecs1) - exp(t, eigvals2, eigvecs2)) for t in time]
    )  # should be a 1D matrix

    return np.max(diffusion_distance)


def edge_deletion(A, i, j):
    """
    A is an adjucency matrix, i and j are nodes between which we want to remove the edge
    """
    A_prime = np.copy(A)

    A_prime[j, i] = 0
    A_prime[i, j] = 0

    return A_prime


def EDP(A, m, n, Fs):
    """
    Function which returns the normalized edge detetion perturbation.
    A - adjucency matrix (original)
    A_prime - is a  A matrix with a delted edge.

    If edge (m, n) is absent (``A[m, n] == 0``), ``chi`` is 0 and ``A_prime``
    is returned as an unmodified copy of ``A`` (so callers can rely on a valid
    matrix in either branch).
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


if __name__ == '__main__':
    main()
