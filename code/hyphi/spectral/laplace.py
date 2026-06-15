"""
Graph Laplacian and eigenvalue helpers for the spectral diffusion / GDD analyses.

Pure NumPy; only depends on ``matplotlib`` for the optional plot inside
:func:`eigen_in_time`.
"""

import matplotlib.pyplot as plt
import numpy as np

# The Fiedler value lambda_2 needs at least two eigenvalues; the eigengap lambda_3 - lambda_2 needs three.
_MIN_NODES_FOR_FIEDLER = 2
_MIN_NODES_FOR_EIGENGAP = 3


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


def eigen_in_time(matrices: np.ndarray, plot=False, fs=1):
    """
    Track the algebraic connectivity of a sequence of matrices over time.

    Parameters
    ----------
    matrices : np.ndarray
        Sequence of square adjacency matrices, one per time step.
    plot : bool, optional
        If True, plot ``lambdas`` and ``gaps`` against a time axis built from ``fs``.
        Default is False.
    fs : int or float, optional
        Sampling frequency used to construct the time axis for plotting. Default is 1.

    Returns
    -------
    lambdas : np.ndarray
        Second-smallest Laplacian eigenvalue of each matrix (the algebraic
        connectivity, or Fiedler value lambda_2).
    gaps : np.ndarray
        The eigengap ``lambda_3 - lambda_2 = eigenvalues[2] - eigenvalues[1]`` for each matrix
        (non-negative). This is the gap *above* the Fiedler value, an indicator of how cleanly the
        network separates into communities over time; it is informative and non-redundant, unlike the
        former ``eigenvalues[0] - eigenvalues[1]`` which (since the smallest Laplacian eigenvalue is
        always 0) was just the negated Fiedler value (resolves issue #70). NaN where a matrix has
        fewer than three nodes (no lambda_3).

    """
    lambdas = np.zeros(len(matrices))
    gaps = np.zeros(len(matrices))  # eigengap lambda_3 - lambda_2 (>= 0); see issue #70

    for i, matrix_item in enumerate(matrices):
        eigenvalues, _, _ = laplace(matrix_item)
        lambdas[i] = eigenvalues[1] if eigenvalues.size >= _MIN_NODES_FOR_FIEDLER else np.nan
        gaps[i] = eigenvalues[2] - eigenvalues[1] if eigenvalues.size >= _MIN_NODES_FOR_EIGENGAP else np.nan

    if plot:
        T = len(matrices) / fs
        t = np.arange(0, T, 1 / fs)
        plt.plot(t, lambdas)
        plt.plot(t, gaps)

    return lambdas, gaps
