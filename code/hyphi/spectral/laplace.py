"""
Graph Laplacian and eigenvalue helpers used by the spectral diffusion / GDD
analyses.  Pure NumPy; only depends on ``matplotlib`` for the optional plot
inside :func:`eigen_in_time`.
"""

import numpy as np
import matplotlib.pyplot as plt


def laplace(matrix: np.ndarray):
    """
    Function calculating eigenvalues of a Laplacian of a given matrix

    input
    Adjacency matrix

    returns
    eigenvalues, eigenvectors and Laplacian matrix

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
    This function takes multiple matrices as input and finds second to smallest eigenvalue of each.
    Fs - sampling frequency

    returns
    lambdas - second to smallest eigenvalue of each matrix
    gaps - moduli of two largest eigenvalues

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
