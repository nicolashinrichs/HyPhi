"""
Benchmarks for HyPhi: standard hyperscanning metrics and classifier evaluation.

Three families of benchmark features for head-to-head comparison against
curvature-entropy features (reviewer-driven addition, 2026-04):

1. **Phase-based metrics** (PLV, wPLI, imaginary coherence) — for pipelines
   where raw phase time series are available (Kuramoto simulation path).
2. **Graph-theoretic metrics** (global efficiency, modularity, assortativity,
   mean clustering) — computed on the same windowed graphs HyPhi uses for
   curvature, addressing the reviewer's *"computed on the same windowed
   graphs"* point directly.
3. **Connectivity-matrix features** — cross- and within-brain strengths and
   graph summaries derived from a pre-computed inter-brain circular-correlation
   (CCORR) matrix.  This is the empirical path in the reference dataset, where
   raw phase is not persisted alongside the connectivity outputs.

A cross-validated classifier (:func:`classify_curvature_vs_benchmarks`) compares
feature sets using :class:`sklearn.model_selection.StratifiedGroupKFold` on
the dyad grouping, so train/test splits never leak across dyads.  Results
should be read as a **proof-of-concept**; with N = 2 empirical dyads, even
group-aware CV yields a best-of-two split whose variance dominates any
between-feature difference.

Years: 2026
"""

from __future__ import annotations

# %% Import
import logging
from typing import Any

import networkx as nx
import numpy as np

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
logger = logging.getLogger(__name__)

__all__ = [
    "classify_curvature_vs_benchmarks",
    "compute_assortativity",
    "compute_global_efficiency",
    "compute_imaginary_coherence",
    "compute_mean_clustering",
    "compute_modularity",
    "compute_plv",
    "compute_wpli",
    "connectivity_matrix_features",
    "extract_window_features",
]

_FEATURE_NAMES: tuple[str, ...] = (
    "mean_cross",
    "std_cross",
    "mean_intra_A",
    "mean_intra_B",
    "mean_abs_cross",
    "global_efficiency",
    "modularity",
    "assortativity",
    "mean_clustering",
    "total_strength",
)


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ---------------------
# Phase-based metrics
# ---------------------


def compute_plv(phases_i: np.ndarray, phases_j: np.ndarray) -> float:
    """
    Phase-locking value (PLV) between two phase time series.

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


def compute_wpli(phases_i: np.ndarray, phases_j: np.ndarray) -> float:
    """
    Weighted phase-lag index (wPLI) from two phase time series.

    Uses the magnitude-weighted sign of the imaginary part of the cross
    spectrum and is therefore less biased by zero-lag volume-conduction
    leakage than PLV.

    Parameters
    ----------
    phases_i, phases_j : np.ndarray
        Phase time series of shape ``(T,)``.

    Returns
    -------
    float
        wPLI in [0, 1].

    """
    phi_i = np.asarray(phases_i, dtype=float)
    phi_j = np.asarray(phases_j, dtype=float)
    cs = np.exp(1j * (phi_i - phi_j))
    im = np.imag(cs)
    denom = float(np.mean(np.abs(im)))
    if denom < 1e-15:
        return 0.0
    return float(np.abs(np.mean(np.abs(im) * np.sign(im))) / denom)


def compute_imaginary_coherence(
    signal_i: np.ndarray,
    signal_j: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Imaginary part of the (magnitude-squared) coherence between two real signals.

    Parameters
    ----------
    signal_i, signal_j : np.ndarray
        Real-valued time series of shape ``(T,)``.
    fs : float
        Sampling frequency (Hz).
    nperseg : int, optional
        Welch segment length; defaults to ``min(256, len(signal_i))``.

    Returns
    -------
    tuple
        ``(freqs, imag_coh)`` — frequency array and ``|Im(coherence)|``.

    """
    from scipy.signal import csd, welch

    si = np.asarray(signal_i, dtype=float)
    sj = np.asarray(signal_j, dtype=float)
    if nperseg is None:
        nperseg = min(256, len(si))

    freqs, pxy = csd(si, sj, fs=fs, nperseg=nperseg)
    _, pxx = welch(si, fs=fs, nperseg=nperseg)
    _, pyy = welch(sj, fs=fs, nperseg=nperseg)

    denom = np.sqrt(pxx * pyy)
    denom = np.where(denom < 1e-15, 1e-15, denom)
    return freqs, np.abs(np.imag(pxy / denom))


# ---------------------
# Graph-theoretic metrics
# ---------------------


def compute_global_efficiency(G: nx.Graph, weight: str | None = None) -> float:
    """Global efficiency of a graph; returns 0.0 on degenerate input."""
    if G.number_of_nodes() < 2:
        return 0.0
    return float(nx.global_efficiency(G))


def compute_modularity(G: nx.Graph, weight: str = "weight") -> float:
    """Modularity via greedy community detection; 0.0 if no edges."""
    if G.number_of_edges() == 0:
        return 0.0
    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(G, weight=weight)
    return float(nx.algorithms.community.quality.modularity(G, communities, weight=weight))


def compute_assortativity(G: nx.Graph, weight: str | None = "weight") -> float:
    """Degree assortativity coefficient; 0.0 when undefined on the graph."""
    if G.number_of_edges() == 0:
        return 0.0
    try:
        r = nx.degree_assortativity_coefficient(G, weight=weight)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(0.0 if np.isnan(r) else r)


def compute_mean_clustering(G: nx.Graph, weight: str | None = "weight") -> float:
    """Mean (weighted) clustering coefficient; 0.0 on empty graphs."""
    if G.number_of_nodes() == 0:
        return 0.0
    return float(nx.average_clustering(G, weight=weight))


# ---------------------
# Connectivity-matrix feature extraction
# ---------------------


def connectivity_matrix_features(
    corr_matrix: np.ndarray,
    n_ch_per_subject: int = 64,
    weight_threshold: float = 0.0,
) -> dict[str, float]:
    """
    Extract a fixed-length vector of standard connectivity summaries from a
    single windowed inter-brain correlation matrix.

    Splits a ``(2n × 2n)`` matrix into three blocks — within-A (top-left),
    within-B (bottom-right), and cross-brain A-B (top-right) — and computes
    block-wise means/std plus whole-graph summaries.  All outputs are real
    scalars suitable as classifier features.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Shape ``(2n, 2n)``.  Expected to be a symmetric, zero-diagonal
        connectivity matrix (CCORR, PLV, coherence, etc.).
    n_ch_per_subject : int
        Number of channels per subject (default 64 → total 128).
    weight_threshold : float
        Absolute edge-weight floor for graph construction; edges with
        ``|w| <= weight_threshold`` are dropped.

    Returns
    -------
    dict
        Feature name → value.  Canonical order is ``_FEATURE_NAMES``.

    """
    M = np.asarray(corr_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {M.shape}")
    total = M.shape[0]
    n = n_ch_per_subject
    if total != 2 * n:
        raise ValueError(f"Matrix is {total}x{total} but n_ch_per_subject={n} implies {2 * n}.")

    intra_A = M[:n, :n]
    intra_B = M[n:, n:]
    cross = M[:n, n:]

    iu = np.triu_indices(n, k=1)
    mean_intra_A = float(np.mean(np.abs(intra_A[iu])))
    mean_intra_B = float(np.mean(np.abs(intra_B[iu])))
    mean_cross = float(np.mean(cross))
    std_cross = float(np.std(cross))
    mean_abs_cross = float(np.mean(np.abs(cross)))
    total_strength = float(np.sum(np.abs(np.triu(M, k=1))))

    W = np.abs(M).copy()
    np.fill_diagonal(W, 0.0)
    if weight_threshold > 0:
        W[W <= weight_threshold] = 0.0
    G = nx.from_numpy_array(W)

    return {
        "mean_cross": mean_cross,
        "std_cross": std_cross,
        "mean_intra_A": mean_intra_A,
        "mean_intra_B": mean_intra_B,
        "mean_abs_cross": mean_abs_cross,
        "global_efficiency": compute_global_efficiency(G),
        "modularity": compute_modularity(G, weight="weight"),
        "assortativity": compute_assortativity(G, weight="weight"),
        "mean_clustering": compute_mean_clustering(G, weight="weight"),
        "total_strength": total_strength,
    }


def extract_window_features(
    ccorr_tensor: np.ndarray,
    n_ch_per_subject: int = 64,
    weight_threshold: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Broadcast :func:`connectivity_matrix_features` over a 5-D CCORR tensor.

    Parameters
    ----------
    ccorr_tensor : np.ndarray
        Shape ``(n_freq, n_trials, n_windows, 2n, 2n)`` — the on-disk layout
        produced by ``hyper_ccorr_{frc,aug_frc}.py``.
    n_ch_per_subject : int
        Channels per subject.
    weight_threshold : float
        Forwarded to :func:`connectivity_matrix_features`.

    Returns
    -------
    tuple
        ``(features, feature_names)`` — ``features`` has shape
        ``(n_freq, n_trials, n_windows, n_features)``.

    """
    if ccorr_tensor.ndim != 5:
        raise ValueError(f"Expected 5-D tensor (freq, trial, window, 2n, 2n), got {ccorr_tensor.shape}")
    n_freq, n_trials, n_windows = ccorr_tensor.shape[:3]
    n_feats = len(_FEATURE_NAMES)
    out = np.empty((n_freq, n_trials, n_windows, n_feats), dtype=float)
    for f in range(n_freq):
        for t in range(n_trials):
            for w in range(n_windows):
                feats = connectivity_matrix_features(
                    ccorr_tensor[f, t, w],
                    n_ch_per_subject=n_ch_per_subject,
                    weight_threshold=weight_threshold,
                )
                for k, name in enumerate(_FEATURE_NAMES):
                    out[f, t, w, k] = feats[name]
    return out, list(_FEATURE_NAMES)


# ---------------------
# Cross-validated classifier comparison
# ---------------------


def classify_curvature_vs_benchmarks(
    X_curvature: np.ndarray,
    X_benchmarks: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    cv: int = 5,
    scoring: str = "accuracy",
    classifier: str = "svm",
    random_state: int = 42,
    X_combined: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Cross-validated head-to-head classification of curvature vs. benchmark features.

    When ``groups`` is provided (typically dyad id),
    :class:`sklearn.model_selection.StratifiedGroupKFold` keeps every dyad's
    rows on one side of each split — this prevents leakage and mirrors the
    reviewer's hierarchical concern.  Otherwise the splitter falls back to
    :class:`StratifiedKFold`.

    Parameters
    ----------
    X_curvature : np.ndarray
        Curvature-based feature matrix ``(n_samples, n_curv_features)``.
    X_benchmarks : np.ndarray
        Benchmark feature matrix ``(n_samples, n_bench_features)``.
    y : np.ndarray
        Sample labels of length ``n_samples``.
    groups : np.ndarray, optional
        Group labels (e.g. dyad id) for group-aware splitting.
    cv : int
        Number of CV folds; automatically reduced if fewer groups exist.
    scoring : str
        Anything accepted by :func:`sklearn.model_selection.cross_val_score`.
    classifier : {"svm", "logreg", "rf"}
        Classifier family.
    random_state : int
        RNG seed.
    X_combined : np.ndarray, optional
        Optional concatenated feature matrix to evaluate as well (e.g. the
        stack of curvature + benchmark features).

    Returns
    -------
    dict
        Keys include ``curvature_scores``, ``benchmark_scores``,
        ``curvature_mean/std``, ``benchmark_mean/std``, ``cv_splitter``,
        ``scoring``, ``classifier``, ``n_samples``, ``n_classes``,
        ``n_groups`` — plus ``combined_*`` if ``X_combined`` was supplied.

    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X_curvature = np.asarray(X_curvature, dtype=float)
    X_benchmarks = np.asarray(X_benchmarks, dtype=float)
    y = np.asarray(y)

    if X_curvature.shape[0] != y.shape[0] or X_benchmarks.shape[0] != y.shape[0]:
        raise ValueError("Feature matrices and labels must share the first axis.")

    if classifier == "svm":
        base = SVC(kernel="rbf", random_state=random_state)
    elif classifier == "logreg":
        base = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier == "rf":
        base = RandomForestClassifier(n_estimators=200, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier!r}")

    pipe = make_pipeline(StandardScaler(), base)

    if groups is not None:
        groups = np.asarray(groups)
        n_uniq = int(len(np.unique(groups)))
        if n_uniq < cv:
            logger.warning(
                "Only %d unique groups for cv=%d; reducing n_splits to %d.",
                n_uniq,
                cv,
                n_uniq,
            )
            cv = max(2, n_uniq)
        splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X_curvature, y, groups=groups))
        splitter_name = "StratifiedGroupKFold"
    else:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X_curvature, y))
        splitter_name = "StratifiedKFold"

    def _score(X: np.ndarray) -> np.ndarray:
        return np.array(
            [cross_val_score(pipe, X, y, scoring=scoring, cv=[(tr, te)])[0] for tr, te in splits]
        )

    curv_scores = _score(X_curvature)
    bench_scores = _score(X_benchmarks)

    out: dict[str, Any] = {
        "curvature_scores": curv_scores,
        "benchmark_scores": bench_scores,
        "curvature_mean": float(curv_scores.mean()),
        "benchmark_mean": float(bench_scores.mean()),
        "curvature_std": float(curv_scores.std()),
        "benchmark_std": float(bench_scores.std()),
        "cv_splitter": splitter_name,
        "n_splits": int(len(splits)),
        "scoring": scoring,
        "classifier": classifier,
        "n_samples": int(X_curvature.shape[0]),
        "n_classes": int(len(np.unique(y))),
        "groups_used": groups is not None,
        "n_groups": int(len(np.unique(groups))) if groups is not None else None,
    }
    if X_combined is not None:
        X_combined = np.asarray(X_combined, dtype=float)
        combo_scores = _score(X_combined)
        out.update(
            {
                "combined_scores": combo_scores,
                "combined_mean": float(combo_scores.mean()),
                "combined_std": float(combo_scores.std()),
            }
        )
    return out


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
