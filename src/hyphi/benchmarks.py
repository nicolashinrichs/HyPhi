# =====================================
# Benchmark Hyperscanning Metrics
# =====================================
"""
Standard hyperscanning connectivity metrics for comparison
against curvature-based measures.
"""

import numpy as np
import networkx as nx
from typing import Optional


def compute_plv(phases_i, phases_j):
    """Compute Phase Locking Value between two phase time series.

    Parameters
    ----------
    phases_i, phases_j : np.ndarray
        Phase time series of shape (T,).

    Returns
    -------
    float
        PLV value in [0, 1].
    """
    return float(np.abs(np.mean(np.exp(1j * (phases_i - phases_j)))))


def compute_wpli(phases_i, phases_j):
    """Compute weighted Phase Lag Index (wPLI).

    The wPLI is defined as:
        wPLI = |E[|Im(S)| * sign(Im(S))]| / E[|Im(S)|]
    where S = exp(j*(phi_i - phi_j)).

    Parameters
    ----------
    phases_i, phases_j : np.ndarray
        Phase time series of shape (T,).

    Returns
    -------
    float
        wPLI value in [0, 1].
    """
    cross_spectrum = np.exp(1j * (phases_i - phases_j))
    imag_part = np.imag(cross_spectrum)
    numerator = np.abs(np.mean(np.abs(imag_part) * np.sign(imag_part)))
    denominator = np.mean(np.abs(imag_part))
    if denominator < 1e-15:
        return 0.0
    return float(numerator / denominator)


def compute_imaginary_coherence(signal_i, signal_j, fs, nperseg=None):
    """Compute imaginary part of coherence between two signals.

    Parameters
    ----------
    signal_i, signal_j : np.ndarray
        Real-valued time series of shape (T,).
    fs : float
        Sampling frequency.
    nperseg : int, optional
        Segment length for Welch method.

    Returns
    -------
    tuple
        (freqs, imag_coh) — frequency array and imaginary coherence values.
    """
    from scipy.signal import csd, welch

    if nperseg is None:
        nperseg = min(256, len(signal_i))

    freqs, Pxy = csd(signal_i, signal_j, fs=fs, nperseg=nperseg)
    _, Pxx = welch(signal_i, fs=fs, nperseg=nperseg)
    _, Pyy = welch(signal_j, fs=fs, nperseg=nperseg)

    denom = np.sqrt(Pxx * Pyy)
    denom[denom < 1e-15] = 1e-15

    coherence = Pxy / denom
    imag_coh = np.abs(np.imag(coherence))

    return freqs, imag_coh


def compute_modularity(G, weight="weight"):
    """Compute modularity of a graph using Louvain community detection.

    Parameters
    ----------
    G : nx.Graph
        Weighted or unweighted graph.
    weight : str
        Edge attribute for weights.

    Returns
    -------
    float
        Modularity score.
    """
    from networkx.algorithms.community import greedy_modularity_communities
    communities = greedy_modularity_communities(G, weight=weight)
    return nx.algorithms.community.quality.modularity(G, communities, weight=weight)


def compute_global_efficiency(G, weight=None):
    """Compute global efficiency of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    weight : str, optional
        Edge attribute for distance (if None, uses hop count).

    Returns
    -------
    float
        Global efficiency.
    """
    return nx.global_efficiency(G)


def classify_curvature_vs_benchmarks(X_curvature, X_benchmarks, y, cv=5):
    """Skeleton for a cross-validated classifier comparing curvature entropy
    features against standard hyperscanning metrics.

    Parameters
    ----------
    X_curvature : np.ndarray
        Feature matrix from curvature entropy, shape (n_samples, n_curvature_features).
    X_benchmarks : np.ndarray
        Feature matrix from standard metrics, shape (n_samples, n_benchmark_features).
    y : np.ndarray
        Labels, shape (n_samples,).
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Dictionary with keys 'curvature_scores' and 'benchmark_scores',
        each containing cross-validation accuracy scores.
    """
    # TODO: Execute classification. Uncomment and run when data is available.
    #
    # from sklearn.model_selection import cross_val_score
    # from sklearn.svm import SVC
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.pipeline import make_pipeline
    #
    # clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    #
    # curvature_scores = cross_val_score(clf, X_curvature, y, cv=cv, scoring='accuracy')
    # benchmark_scores = cross_val_score(clf, X_benchmarks, y, cv=cv, scoring='accuracy')
    #
    # return {
    #     'curvature_scores': curvature_scores,
    #     'benchmark_scores': benchmark_scores,
    # }

    raise NotImplementedError(
        "TODO: Execute classification when empirical feature matrices are available."
    )
