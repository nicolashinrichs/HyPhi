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
        Sequence of square adjacency matrices, one per time step.
    plot : bool, optional
        If True, plot ``lambdas`` and ``gaps`` against a time axis built from ``Fs``.
        Default is False.
    Fs : int or float, optional
        Sampling frequency used to construct the time axis for plotting. Default is 1.

    Returns
    -------
    lambdas : np.ndarray
        Second-smallest Laplacian eigenvalue of each matrix (the algebraic
        connectivity, or Fiedler value).
    gaps : np.ndarray
        Signed gap ``eigenvalues[0] - eigenvalues[1]`` for each matrix. Because the
        eigenvalues are sorted ascending this is non-positive (``<= 0``); its sign
        convention is under review (see issue #70).

    """
    lambdas = np.zeros(len(matrices))
    gaps = np.zeros(len(matrices))  # signed gap between the two smallest eigenvalues (<= 0); see issue #70

    for i, matrix_item in enumerate(matrices):
        eigenvalues, _, _ = laplace(matrix_item)
        lambdas[i] = eigenvalues[1]
        gaps[i] = eigenvalues[0] - eigenvalues[1]

    if plot:
        T = len(matrices) / Fs
        t = np.arange(0, T, 1 / Fs)
        plt.plot(t, lambdas)
        plt.plot(t, gaps)

    return lambdas, gaps
