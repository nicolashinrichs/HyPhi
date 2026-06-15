"""
Centralized sliding-window Phase Locking Value (PLV) graph construction.

This is the canonical PLV implementation for HyPhi: ``compute_plv_pair`` for a single
channel pair and ``compute_plv_matrix`` for the full connectivity matrix. The JAX variant
``get_plv_matrix`` in :mod:`hyphi.simulation.kuramoto_simulations` is numerically equivalent
but exists only for Kuramoto-simulated phases (the simulation package and the
simulate-missing-phases step of ``modeling.ricci_flow_analysis``); everything else uses this
module. The two agree to float32 tolerance (pinned by ``test_jax_plv_matches_numpy``).

Extracted from the original KuramotoSimulations.py and connectome exploration code.
"""

# %% Import
import networkx as nx
import numpy as np

__all__ = [
    "build_graphs_from_matrices",
    "compute_plv_matrix",
    "compute_plv_pair",
    "sliding_window_plv",
]

# Number of dimensions expected for a 2-D phase array.
_NDIM_2D = 2

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_plv_pair(phases_i: np.ndarray, phases_j: np.ndarray) -> float:
    """
    Phase-locking value (PLV) between two phase time series.

    ``PLV = |mean_t exp(i * (phi_i - phi_j))|``. This is the single-pair form of
    :func:`compute_plv_matrix`; the two agree numerically.

    Parameters
    ----------
    phases_i, phases_j : np.ndarray
        Phase time series of shape ``(T,)`` in radians.

    Returns
    -------
    float
        PLV in [0, 1]; 1.0 means perfect phase locking.

    """
    phi_i = np.asarray(phases_i, dtype=float)
    phi_j = np.asarray(phases_j, dtype=float)
    return float(np.abs(np.mean(np.exp(1j * (phi_i - phi_j)))))


def compute_plv_matrix(phase_window):
    """
    Compute the PLV connectivity matrix for a single window.

    Parameters
    ----------
    phase_window : np.ndarray
        Phase time series of shape (N_oscillators, T_window).

    Returns
    -------
    np.ndarray
        Symmetric PLV matrix of shape (N, N) with values in [0, 1].

    """
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
    return (C + C.T) / 2.0


def sliding_window_plv(phases, win_size, win_stride, samples_axis=-1):
    """
    Build a time series of PLV graphs using a sliding window.

    Parameters
    ----------
    phases : np.ndarray
        2-D phase trajectories. By default the channels/oscillators are the first axis and
        time is the last axis, i.e. shape ``(N, T)``.
    win_size : int
        Number of time-steps per window.
    win_stride : int
        Stride between consecutive windows.
    samples_axis : int, default=-1
        Which axis is the time/samples axis. The default (``-1``) treats ``phases`` as
        ``(N, T)``; pass ``samples_axis=0`` for ``(T, N)`` input. The orientation is taken from
        this argument rather than guessed from the shape, so windows with more channels than time
        samples are no longer silently transposed.

    Returns
    -------
    list[nx.Graph]
        List of weighted NetworkX graphs, one per window.

    """
    if phases.ndim != _NDIM_2D:
        raise ValueError(f"phases must be 2-D, got shape {phases.shape}")

    # Normalise to (N, T) using the declared samples axis (no shape-based guessing).
    phases = np.moveaxis(phases, samples_axis, -1)

    _N, T = phases.shape
    graphs = []

    for start in range(0, T - win_size + 1, win_stride):
        window_phases = phases[:, start : start + win_size]
        C = compute_plv_matrix(window_phases)
        G = nx.from_numpy_array(C, create_using=nx.Graph)
        graphs.append(G)

    return graphs


def build_graphs_from_matrices(matrices):
    """
    Convert a sequence of adjacency matrices to NetworkX graphs.

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


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
