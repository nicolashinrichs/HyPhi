"""Tests for the fMRI module.

Checks the ROI correlation and partial-correlation connectivity, the end-to-end path from ROI
time series to a curvature-entropy series, and that parcellation raises a clear error when
nilearn is absent.
"""

# %% Import
import importlib.util

import numpy as np
import pytest
from hyphi.analyses import build_sliding_window_graphs, compute_entropy, compute_windowed_curvatures
from hyphi.fmri import parcellate, roi_correlation, roi_partial_correlation

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _roi_timeseries(n_rois=8, n_times=600, seed=0):
    """A synthetic ROI time series with some shared structure between ROIs."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(n_times)
    return np.stack([0.5 * shared + rng.standard_normal(n_times) for _ in range(n_rois)])


class TestROIConnectivity:
    """Both connectivity functions produce a (windows, ROIs, ROIs) tensor."""

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_shape_and_symmetry(self, fn):
        """Each window is a symmetric matrix with unit diagonal."""
        data = _roi_timeseries(n_rois=6, n_times=400)
        tensor = fn(data, win_size=120, win_stride=60)
        n_windows = (400 - 120) // 60 + 1
        assert tensor.shape == (n_windows, 6, 6)
        for mat in tensor:
            np.testing.assert_allclose(mat, mat.T, atol=1e-8)
            np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-8)

    def test_pearson_in_range(self):
        """Pearson correlation stays within [-1, 1]."""
        tensor = roi_correlation(_roi_timeseries(), win_size=150, win_stride=75)
        assert np.all(tensor >= -1.0 - 1e-9) and np.all(tensor <= 1.0 + 1e-9)

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_rejects_non_2d(self, fn):
        """A non-2D input is rejected with a clear error."""
        with pytest.raises(ValueError, match="n_rois, n_times"):
            fn(np.zeros((2, 3, 4)), win_size=2, win_stride=1)

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_rejects_dead_roi(self, fn):
        """A constant (dead/saturated) ROI is rejected loudly, not silently NaN-collapsed."""
        data = _roi_timeseries(n_rois=6, n_times=400)
        data[2, :] = 7.0  # a dead ROI railed at a nonzero constant
        with pytest.raises(ValueError, match="zero variance"):
            fn(data, win_size=120, win_stride=60)

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    @pytest.mark.parametrize(
        ("win_size", "win_stride", "match"),
        [(1, 1, "at least 2"), (500, 50, "exceeds the series length"), (100, 0, "win_stride must be at least 1")],
    )
    def test_rejects_bad_window_params(self, fn, win_size, win_stride, match):
        """Out-of-range win_size/win_stride raise clear errors, not opaque numpy failures."""
        data = _roi_timeseries(n_rois=6, n_times=400)
        with pytest.raises(ValueError, match=match):
            fn(data, win_size=win_size, win_stride=win_stride)

    def test_partial_correlation_warns_on_short_window(self):
        """win_size <= n_rois makes the covariance rank-deficient -> warn that pcorr is unreliable."""
        data = _roi_timeseries(n_rois=10, n_times=400)
        with pytest.warns(UserWarning, match="rank-deficient"):
            roi_partial_correlation(data, win_size=8, win_stride=4)

    def test_partial_correlation_rejects_out_of_range(self, monkeypatch):
        """A degenerate window yielding |pcorr| > 1 is rejected, not fed into curvature."""
        # In the wild this comes from a near-constant ROI; the blow-up is numerically version-dependent
        # (it did not reproduce on the 3.13 BLAS), so force it deterministically: stub pinv to the
        # precision matrix P=[[1,2],[2,1]], giving pcorr_01 = -2.
        monkeypatch.setattr(np.linalg, "pinv", lambda _m: np.array([[1.0, 2.0], [2.0, 1.0]]))
        data = np.random.default_rng(0).standard_normal((2, 200))  # 2 valid (non-constant) ROIs
        with pytest.raises(ValueError, match="exceeds"):
            roi_partial_correlation(data, win_size=100, win_stride=50)

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_rejects_nonfinite_input(self, fn):
        """inf/NaN in the input is rejected with a clear ValueError (not an opaque LinAlgError)."""
        data = _roi_timeseries(n_rois=5, n_times=300)
        # Place the NaN in a stride GAP (win_size=50, win_stride=100 -> windows 0/100/200, gap 50..100),
        # so only the early _check_2d guard catches it; the per-window guard never sees this sample.
        data[0, 70] = np.nan
        with pytest.raises(ValueError, match="inf or NaN"):
            fn(data, win_size=50, win_stride=100)


class TestEndToEnd:
    """An fMRI ROI series flows to a finite curvature-entropy time series."""

    @pytest.mark.parametrize("fn", [roi_correlation, roi_partial_correlation])
    def test_roi_to_curvature_entropy(self, fn):
        """connectivity -> graphs -> Forman curvature -> entropy, all finite."""
        tensor = fn(_roi_timeseries(n_rois=10, n_times=500), win_size=100, win_stride=50)
        graphs = build_sliding_window_graphs(tensor)
        assert len(graphs) == tensor.shape[0]
        entropies = compute_entropy(compute_windowed_curvatures(graphs))
        assert len(entropies) == len(graphs)
        assert np.all(np.isfinite(entropies))


class TestSignedConnectivityIsAbsed:
    """The downstream graph builder takes |connectivity|, so the correlation sign is discarded."""

    def test_anticorrelation_equals_correlation_in_graph_weights(self):
        """+r and -r of equal magnitude give identical graph edge weights (sign discarded by abs)."""
        base = np.random.default_rng(0).standard_normal((1, 300))
        pos = np.vstack([base, base])  # corr(roi0, roi1) = +1
        neg = np.vstack([base, -base])  # corr(roi0, roi1) = -1
        g_pos = build_sliding_window_graphs(roi_correlation(pos, win_size=100, win_stride=50))
        g_neg = build_sliding_window_graphs(roi_correlation(neg, win_size=100, win_stride=50))
        wp = sorted(d["weight"] for _, _, d in g_pos[0].edges(data=True))
        wn = sorted(d["weight"] for _, _, d in g_neg[0].edges(data=True))
        np.testing.assert_allclose(wp, wn, atol=1e-9)


class TestParcellation:
    """Parcellation delegates to nilearn and errors clearly when it is absent."""

    def test_requires_nilearn_when_absent(self):
        """Without nilearn, parcellate raises an informative ImportError."""
        if importlib.util.find_spec("nilearn") is not None:
            pytest.skip("nilearn is installed; the missing-dependency path is not exercised")
        with pytest.raises(ImportError, match="requires nilearn"):
            parcellate(func_img=None, atlas_img=None)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
