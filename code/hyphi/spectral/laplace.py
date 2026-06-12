"""
Graph Laplacian and eigenvalue helpers used by the spectral diffusion / GDD
analyses.  Pure NumPy; only depends on ``matplotlib`` for the optional plot
inside :func:`eigen_in_time`.
"""

import matplotlib.pyplot as plt
import numpy as np


def laplace(matrix: np.ndarray):
    """
    Compute the eigendecomposition of the graph Laplacian of an adjacency matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square adjacency matrix of the graph.

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues of the Laplacian ``L = D - A``, in ascending order.
    eigenvectors : np.ndarray
        The corresponding eigenvectors as columns.
    L : np.ndarray
        The Laplacian matrix itself.

    """
    # Calculating degrees of nodes
    degrees = np.sum(matrix, axis=1)

    # degree matrix is diagonal
    D = np.diag(degrees)

    # Laplacian matrix
    L = D - matrix

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    return eigenvalues, eigenvectors, L


def eigen_in_time(matrices: np.ndarray, plot=False, Fs=1):
    """
    Track the algebraic connectivity of a sequence of matrices over time.

    Parameters
    ----------
    matrices : np.ndarray
        A sequence of square adjacency matrices, one per time point.
    plot : bool, default=False
        If True, plot the tracked values against a time axis built from ``Fs``.
    Fs : float, default=1
        Sampling frequency, used only to build the time axis when plotting.

    Returns
    -------
    lambdas : np.ndarray
        The second-smallest Laplacian eigenvalue (the algebraic connectivity, or
        Fiedler value) of each matrix.
    gaps : np.ndarray
        The difference ``eigenvalues[0] - eigenvalues[1]`` for each matrix, that is
        the smallest (near-zero null) eigenvalue minus the algebraic connectivity.

    """
    lambdas = np.zeros(len(matrices))
    gaps = np.zeros(len(matrices))  # gap between smallest and largest eigenvalues

    for i, matrix_item in enumerate(matrices):
        eigenvalues, eigenvectors, _ = laplace(matrix_item)
        lambdas[i] = eigenvalues[1]
        gaps[i] = eigenvalues[0] - eigenvalues[1]

    if plot:
        T = len(matrices) / Fs
        t = np.arange(0, T, 1 / Fs)
        plt.plot(t, lambdas)
        plt.plot(t, gaps)

    return lambdas, gaps
