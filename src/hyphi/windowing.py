# =================================
# Sliding Window PLV Graph Builder
# =================================
"""
Centralised sliding-window Phase Locking Value (PLV) graph construction.
Extracted from KuramotoSimulations.py and connectome_kuramoto.ipynb.
"""

import numpy as np
import networkx as nx


def compute_plv_matrix(phase_window):
    """Compute the PLV connectivity matrix for a single window.

    Parameters
    ----------
    phase_window : np.ndarray
        Phase time series of shape (N_oscillators, T_window).

    Returns
    -------
    np.ndarray
        Symmetric PLV matrix of shape (N, N) with values in [0, 1].
    """
    N = phase_window.shape[0]
    # Vectorised PLV: PLV_ij = |mean(exp(j * (phi_i - phi_j)))|
    # phase_window shape: (N, T)
    # Compute pairwise phase differences
    # exp_phases shape: (N, T)
    exp_phases = np.exp(1j * phase_window)

    # PLV matrix: (N, N) = |<exp(j*phi_i) * conj(exp(j*phi_j))>_t|
    #                     = |mean over t of exp(j*(phi_i - phi_j))|
    T = phase_window.shape[1]
    C = np.abs(exp_phases @ exp_phases.conj().T) / T

    # Ensure exact symmetry
    C = (C + C.T) / 2.0
    return C


def sliding_window_plv(phases, win_size, win_stride):
    """Build a time series of PLV graphs using a sliding window.

    Parameters
    ----------
    phases : np.ndarray
        Phase trajectories of shape (n_steps, N_oscillators) or (N_oscillators, n_steps).
        If shape[0] < shape[1], assumed to be (N, T) already; otherwise transposed.
    win_size : int
        Number of time-steps per window.
    win_stride : int
        Stride between consecutive windows.

    Returns
    -------
    list[nx.Graph]
        List of weighted NetworkX graphs, one per window.
    """
    # Normalise to (N, T)
    if phases.ndim != 2:
        raise ValueError(f"phases must be 2-D, got shape {phases.shape}")

    # Heuristic: if first dim > second dim, it's (T, N) — transpose
    if phases.shape[0] > phases.shape[1]:
        phases = phases.T

    N, T = phases.shape
    graphs = []

    for start in range(0, T - win_size + 1, win_stride):
        window_phases = phases[:, start:start + win_size]
        C = compute_plv_matrix(window_phases)
        G = nx.from_numpy_array(C, create_using=nx.Graph)
        graphs.append(G)

    return graphs


def build_graphs_from_matrices(matrices):
    """Convert a sequence of adjacency matrices to NetworkX graphs.

    Parameters
    ----------
    matrices : np.ndarray or list of np.ndarray
        Each element is a 2-D adjacency matrix.

    Returns
    -------
    list[nx.Graph]
        List of weighted NetworkX graphs.
    """
    graphs = []
    for mat in matrices:
        G = nx.from_numpy_array(np.asarray(mat), create_using=nx.Graph)
        graphs.append(G)
    return graphs
