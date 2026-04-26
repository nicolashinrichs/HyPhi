"""
Tests for hyphi.benchmarks: phase, graph-theoretic, and connectivity-matrix features.

Covers:
  * PLV/wPLI/imaginary-coherence known-answer cases (identical, anti-phase, random)
  * Graph-metric degeneracy handling (empty graph, single node)
  * connectivity_matrix_features returns canonical feature order
  * extract_window_features broadcasts a 5-D CCORR tensor correctly
  * classify_curvature_vs_benchmarks picks group-aware CV when groups are given
    and falls back to StratifiedKFold otherwise

Years: 2026
"""

# %% Import
import networkx as nx
import numpy as np
import pytest

from hyphi.benchmarks import (
    classify_curvature_vs_benchmarks,
    compute_assortativity,
    compute_global_efficiency,
    compute_imaginary_coherence,
    compute_mean_clustering,
    compute_modularity,
    compute_plv,
    compute_wpli,
    connectivity_matrix_features,
    extract_window_features,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

_CANONICAL_FEATURE_NAMES = (
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


# %% Helpers >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _symmetric_corr_matrix(size: int, seed: int) -> np.ndarray:
    """Random symmetric, zero-diagonal matrix with entries in [-1, 1]."""
    rng = np.random.default_rng(seed)
    M = rng.uniform(-1.0, 1.0, size=(size, size))
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return M


# %% Tests for phase-based metrics >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestPLV:
    """Phase-locking value."""

    def test_identical_phases_equal_one(self):
        rng = np.random.default_rng(0)
        phi = rng.uniform(-np.pi, np.pi, 1024)
        assert compute_plv(phi, phi) == pytest.approx(1.0, abs=1e-12)

    def test_constant_lag_equal_one(self):
        rng = np.random.default_rng(0)
        phi = rng.uniform(-np.pi, np.pi, 1024)
        assert compute_plv(phi, phi + 0.7) == pytest.approx(1.0, abs=1e-12)

    def test_independent_phases_near_zero(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(-np.pi, np.pi, 4096)
        b = rng.uniform(-np.pi, np.pi, 4096)
        assert compute_plv(a, b) < 0.1

    def test_bounded_unit_interval(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(-np.pi, np.pi, 256)
        b = rng.uniform(-np.pi, np.pi, 256)
        val = compute_plv(a, b)
        assert 0.0 <= val <= 1.0


class TestWPLI:
    """Weighted phase-lag index."""

    def test_zero_lag_returns_zero(self):
        rng = np.random.default_rng(0)
        phi = rng.uniform(-np.pi, np.pi, 1024)
        # Zero lag → imag part is ~0, so wPLI should be ~0 (numerator/denominator tiny).
        assert compute_wpli(phi, phi) < 0.2

    def test_consistent_lag_gives_high_value(self):
        rng = np.random.default_rng(0)
        phi = rng.uniform(-np.pi, np.pi, 2048)
        # 90° lag keeps imag sign constant → wPLI close to 1.
        assert compute_wpli(phi, phi + np.pi / 2) > 0.8

    def test_bounded_unit_interval(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(-np.pi, np.pi, 256)
        b = rng.uniform(-np.pi, np.pi, 256)
        val = compute_wpli(a, b)
        assert 0.0 <= val <= 1.0


class TestImaginaryCoherence:
    """Imaginary part of coherence via Welch/CSD."""

    def test_returns_expected_shape(self):
        rng = np.random.default_rng(0)
        fs = 200.0
        sig_a = rng.normal(size=1024)
        sig_b = rng.normal(size=1024)
        freqs, icoh = compute_imaginary_coherence(sig_a, sig_b, fs=fs)
        assert freqs.shape == icoh.shape
        assert freqs[0] == 0.0
        assert np.all(icoh >= 0.0)


# %% Tests for graph metrics >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestGraphMetrics:
    """Degenerate inputs and simple known graphs."""

    def test_global_efficiency_empty_graph(self):
        assert compute_global_efficiency(nx.Graph()) == 0.0

    def test_global_efficiency_complete_graph_is_one(self):
        G = nx.complete_graph(5)
        assert compute_global_efficiency(G) == pytest.approx(1.0)

    def test_modularity_nonnegative(self):
        G = nx.karate_club_graph()
        q = compute_modularity(G, weight=None)
        assert q >= 0.0

    def test_modularity_empty_graph_zero(self):
        assert compute_modularity(nx.empty_graph(5)) == 0.0

    def test_assortativity_finite(self):
        G = nx.erdos_renyi_graph(20, 0.3, seed=0)
        val = compute_assortativity(G, weight=None)
        assert np.isfinite(val)

    def test_assortativity_empty_graph_zero(self):
        assert compute_assortativity(nx.empty_graph(5)) == 0.0

    def test_mean_clustering_bounds(self):
        G = nx.complete_graph(6)
        assert compute_mean_clustering(G, weight=None) == pytest.approx(1.0)


# %% Tests for connectivity-matrix features >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestConnectivityMatrixFeatures:
    """Block decomposition and canonical feature ordering."""

    def test_returns_all_canonical_names(self):
        M = _symmetric_corr_matrix(size=8, seed=0)
        feats = connectivity_matrix_features(M, n_ch_per_subject=4)
        assert set(feats.keys()) == set(_CANONICAL_FEATURE_NAMES)
        for v in feats.values():
            assert np.isfinite(v)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            connectivity_matrix_features(np.zeros((4, 5)), n_ch_per_subject=2)

    def test_mismatched_subject_count_raises(self):
        with pytest.raises(ValueError, match="implies"):
            connectivity_matrix_features(np.zeros((6, 6)), n_ch_per_subject=4)

    def test_block_decomposition_reflects_structure(self):
        n = 4
        M = np.zeros((2 * n, 2 * n))
        # Strong within-A, weak cross — expect high mean_intra_A, low mean_abs_cross.
        M[:n, :n] = 0.9
        np.fill_diagonal(M, 0.0)
        M = (M + M.T) / 2.0
        feats = connectivity_matrix_features(M, n_ch_per_subject=n)
        assert feats["mean_intra_A"] > feats["mean_abs_cross"]


class TestExtractWindowFeatures:
    """Broadcast over 5-D CCORR tensor."""

    def test_shape_matches_expected_layout(self):
        rng = np.random.default_rng(0)
        n_freq, n_trials, n_windows, n_ch = 2, 3, 2, 4
        total = 2 * n_ch
        tensor = rng.uniform(-1, 1, size=(n_freq, n_trials, n_windows, total, total))
        # Symmetrise and zero-diagonal each window
        for f in range(n_freq):
            for t in range(n_trials):
                for w in range(n_windows):
                    M = tensor[f, t, w]
                    M[:] = (M + M.T) / 2.0
                    np.fill_diagonal(M, 0.0)
        feats, names = extract_window_features(tensor, n_ch_per_subject=n_ch)
        assert feats.shape == (n_freq, n_trials, n_windows, len(_CANONICAL_FEATURE_NAMES))
        assert tuple(names) == _CANONICAL_FEATURE_NAMES

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="5-D"):
            extract_window_features(np.zeros((3, 4, 4)))


# %% Tests for classifier comparison >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestClassifyCurvatureVsBenchmarks:
    """Group-aware vs. plain CV dispatch and report shape."""

    def _make_data(self, n_samples: int = 40, n_curv: int = 4, n_bench: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        X_c = rng.normal(size=(n_samples, n_curv))
        X_b = rng.normal(size=(n_samples, n_bench))
        y = np.tile([0, 1], n_samples // 2)
        # give class 1 a shift in both feature sets so CV can actually classify
        X_c[y == 1] += 1.0
        X_b[y == 1] += 1.0
        return X_c, X_b, y

    def test_group_aware_cv_used_when_groups_passed(self):
        X_c, X_b, y = self._make_data(n_samples=40)
        groups = np.repeat(np.arange(4), 10)
        out = classify_curvature_vs_benchmarks(
            X_curvature=X_c, X_benchmarks=X_b, y=y, groups=groups, cv=4
        )
        assert out["cv_splitter"] == "StratifiedGroupKFold"
        assert out["groups_used"] is True
        assert out["n_groups"] == 4
        assert out["curvature_scores"].shape[0] == out["n_splits"]

    def test_stratified_kfold_when_no_groups(self):
        X_c, X_b, y = self._make_data(n_samples=40)
        out = classify_curvature_vs_benchmarks(
            X_curvature=X_c, X_benchmarks=X_b, y=y, groups=None, cv=5
        )
        assert out["cv_splitter"] == "StratifiedKFold"
        assert out["groups_used"] is False
        assert out["n_groups"] is None

    def test_combined_feature_evaluation(self):
        X_c, X_b, y = self._make_data(n_samples=40)
        X_combo = np.concatenate([X_c, X_b], axis=1)
        out = classify_curvature_vs_benchmarks(
            X_curvature=X_c,
            X_benchmarks=X_b,
            y=y,
            groups=None,
            cv=5,
            X_combined=X_combo,
        )
        assert "combined_mean" in out
        assert "combined_std" in out
        assert out["combined_scores"].shape[0] == out["n_splits"]

    def test_row_length_mismatch_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="share the first axis"):
            classify_curvature_vs_benchmarks(
                X_curvature=rng.normal(size=(10, 3)),
                X_benchmarks=rng.normal(size=(9, 3)),
                y=np.tile([0, 1], 5),
                groups=None,
                cv=2,
            )

    def test_unknown_classifier_raises(self):
        X_c, X_b, y = self._make_data(n_samples=20)
        with pytest.raises(ValueError, match="Unknown classifier"):
            classify_curvature_vs_benchmarks(
                X_curvature=X_c,
                X_benchmarks=X_b,
                y=y,
                groups=None,
                cv=2,
                classifier="xgboost",
            )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
