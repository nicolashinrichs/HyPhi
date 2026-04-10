"""
Benchmarks module for HyPhi: Standard connectivity metrics and classifier evaluation.

Years: 2026
"""

# %% Import
import networkx as nx
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_plv(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Compute Phase Locking Value (PLV) between two signals."""
    if len(signal1) != len(signal2):
        raise ValueError("Signals must be of same length.")

    # Calculate phase difference
    phase_diff = np.angle(signal1) - np.angle(signal2)

    # Compute PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return float(plv)


def compute_wpli(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Compute weighted Phase Lag Index (wPLI) between two signals."""
    # Cross-spectrum
    cross_spec = signal1 * np.conj(signal2)
    imag_cs = np.imag(cross_spec)

    numerator = np.abs(np.mean(imag_cs))
    denominator = np.mean(np.abs(imag_cs))

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_imaginary_coherence(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Compute Imaginary Coherence between two signals."""
    cross_spec = np.mean(signal1 * np.conj(signal2))
    psd1 = np.mean(np.abs(signal1) ** 2)
    psd2 = np.mean(np.abs(signal2) ** 2)

    if psd1 == 0 or psd2 == 0:
        return 0.0

    coh = np.imag(cross_spec) / np.sqrt(psd1 * psd2)
    return float(np.abs(coh))


def compute_global_efficiency(G: nx.Graph) -> float:
    """Compute Global Efficiency of a network."""
    try:
        return nx.global_efficiency(G)
    except nx.NetworkXError:
        return 0.0


def compute_modularity(G: nx.Graph) -> float:
    """Compute Modularity using Clauset-Newman-Moore greedy modularity maximization."""
    try:
        communities = nx.community.greedy_modularity_communities(G)
        return nx.community.modularity(G, communities)
    except (ZeroDivisionError, nx.NetworkXError):
        return 0.0


def evaluate_classifier_skeleton(X_curvature: np.ndarray, X_benchmark: np.ndarray, y: np.ndarray):
    """Skeleton for cross-validated classifier comparing geometric metrics to benchmarks."""
    # TODO: Implement complete classifier analysis
    clf1 = SVC(kernel="linear")
    clf2 = SVC(kernel="linear")

    # Example placeholder return
    scores_curv = cross_val_score(clf1, X_curvature, y, cv=5)
    scores_bench = cross_val_score(clf2, X_benchmark, y, cv=5)

    return scores_curv.mean(), scores_bench.mean()


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
