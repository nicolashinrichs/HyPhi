"""Tests for the fNIRS module.

Checks haemoglobin conversion on a synthetic fNIRS recording, the windowed correlation
connectivity, and the end-to-end path from haemoglobin time series to a curvature-entropy
series through HyPhi's graph pipeline.
"""

# %% Import
import numpy as np
import pytest
from hyphi.analyses import build_sliding_window_graphs, compute_entropy, compute_windowed_curvatures
from hyphi.fnirs import haemoglobin_data, to_haemoglobin, windowed_correlation

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _synthetic_cw_amplitude_raw(n_pairs=3, n_times=600, sfreq=10.0, seed=0):
    """A tiny synthetic continuous-wave fNIRS Raw with n_pairs source-detector pairs."""
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(seed)
    ch_names = []
    for p in range(n_pairs):
        ch_names += [f"S{p}_D{p} 760", f"S{p}_D{p} 850"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="fnirs_cw_amplitude")
    for i, ch in enumerate(info["chs"]):
        ch["loc"][9] = 760.0 if "760" in ch_names[i] else 850.0
        ch["loc"][3:6] = [0.0, 0.0, 0.0]  # source position
        ch["loc"][6:9] = [0.03, 0.0, 0.0]  # detector 3 cm away
    data = np.abs(rng.standard_normal((2 * n_pairs, n_times))) + 1.0
    return mne.io.RawArray(data, info)


class TestHaemoglobinConversion:
    """to_haemoglobin runs the MNE optics pipeline and yields HbO/HbR channels."""

    def test_produces_haemoglobin_channels(self):
        """Beer-Lambert turns cw-amplitude channels into hbo/hbr channels."""
        raw = _synthetic_cw_amplitude_raw()
        hb = to_haemoglobin(raw, tddr=True)
        assert any(name.endswith("hbo") for name in hb.ch_names)
        assert any(name.endswith("hbr") for name in hb.ch_names)

    def test_haemoglobin_data_extracts_one_chromophore(self):
        """haemoglobin_data returns one chromophore as (n_channels, n_times)."""
        hb = to_haemoglobin(_synthetic_cw_amplitude_raw(n_pairs=3), tddr=False)
        hbo = haemoglobin_data(hb, chroma="hbo")
        assert hbo.ndim == 2
        assert hbo.shape[0] == 3  # one hbo channel per source-detector pair

    def test_tddr_changes_the_result(self):
        """tddr=True applies motion correction, so it is not a no-op vs tddr=False."""
        raw = _synthetic_cw_amplitude_raw(seed=2)
        hb_on = haemoglobin_data(to_haemoglobin(raw.copy(), tddr=True), chroma="hbo")
        hb_off = haemoglobin_data(to_haemoglobin(raw.copy(), tddr=False), chroma="hbo")
        assert not np.allclose(hb_on, hb_off)

    def test_ppf_is_forwarded(self):
        """The partial pathlength factor scales the Beer-Lambert output, so changing it changes hbo."""
        raw = _synthetic_cw_amplitude_raw(seed=3)
        hb6 = haemoglobin_data(to_haemoglobin(raw.copy(), ppf=6.0, tddr=False), chroma="hbo")
        hb60 = haemoglobin_data(to_haemoglobin(raw.copy(), ppf=60.0, tddr=False), chroma="hbo")
        assert not np.allclose(hb6, hb60)


class TestWindowedCorrelation:
    """windowed_correlation produces a (windows, nodes, nodes) connectivity tensor."""

    def test_shape_and_properties(self):
        """Each window is a symmetric correlation matrix with unit diagonal in [-1, 1]."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 400))
        tensor = windowed_correlation(data, win_size=100, win_stride=50)
        n_windows = (400 - 100) // 50 + 1
        assert tensor.shape == (n_windows, 5, 5)
        for mat in tensor:
            np.testing.assert_allclose(mat, mat.T, atol=1e-12)
            np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-10)
            assert np.all(mat >= -1.0 - 1e-9) and np.all(mat <= 1.0 + 1e-9)

    def test_rejects_non_2d(self):
        """A non-2D input is rejected with a clear error."""
        with pytest.raises(ValueError, match="n_channels, n_times"):
            windowed_correlation(np.zeros((2, 3, 4)), win_size=2, win_stride=1)

    def test_rejects_tiny_window(self):
        """win_size < 2 cannot define a correlation and is rejected."""
        data = np.random.default_rng(0).standard_normal((4, 50))
        with pytest.raises(ValueError, match="at least 2"):
            windowed_correlation(data, win_size=1, win_stride=1)

    def test_rejects_window_longer_than_series(self):
        """win_size > n_times fits no window and is rejected with a clear error (not an opaque stack error)."""
        data = np.random.default_rng(0).standard_normal((4, 50))
        with pytest.raises(ValueError, match="exceeds the series length"):
            windowed_correlation(data, win_size=100, win_stride=10)

    def test_rejects_dead_channel(self):
        """A zero-variance (dead/saturated) channel is rejected loudly, not silently NaN-collapsed."""
        data = np.random.default_rng(0).standard_normal((4, 200))
        data[2, :] = 0.0  # a dead optode (constant zero)
        with pytest.raises(ValueError, match="zero variance"):
            windowed_correlation(data, win_size=100, win_stride=50)

    def test_rejects_saturated_nonzero_channel(self):
        """A channel railed at a NONZERO constant (roundoff variance ~1e-31, finite corrcoef) is still
        rejected: peak-to-peak detection catches it where a NaN-only check would slip it through."""
        data = np.random.default_rng(0).standard_normal((3, 200))
        data[1, :] = 4.2  # a saturated optode at a nonzero rail
        with pytest.raises(ValueError, match="zero variance"):
            windowed_correlation(data, win_size=200, win_stride=1)

    def test_rejects_nonpositive_stride(self):
        """win_stride < 1 is rejected with a clear error, not an opaque range()/stack failure."""
        data = np.random.default_rng(0).standard_normal((4, 50))
        with pytest.raises(ValueError, match="win_stride must be at least 1"):
            windowed_correlation(data, win_size=10, win_stride=0)


class TestEndToEnd:
    """An fNIRS haemoglobin series flows to a finite curvature-entropy time series."""

    def test_haemoglobin_to_curvature_entropy(self):
        """connectivity -> graphs -> Forman curvature -> entropy, all finite."""
        rng = np.random.default_rng(1)
        haemoglobin = rng.standard_normal((8, 500))  # 8 channels, 500 time points
        tensor = windowed_correlation(haemoglobin, win_size=100, win_stride=50)
        graphs = build_sliding_window_graphs(tensor)
        assert len(graphs) == tensor.shape[0]
        curved = compute_windowed_curvatures(graphs)
        entropies = compute_entropy(curved)
        assert len(entropies) == len(graphs)
        assert np.all(np.isfinite(entropies))


class TestSignedCorrelationIsAbsed:
    """The downstream graph builder takes |connectivity|, so the correlation sign is discarded."""

    def test_anticorrelation_equals_correlation_in_graph_weights(self):
        """+r and -r of equal magnitude give identical graph edge weights (sign is discarded by abs)."""
        base = np.random.default_rng(0).standard_normal((1, 300))
        pos = np.vstack([base, base])  # corr(ch0, ch1) = +1
        neg = np.vstack([base, -base])  # corr(ch0, ch1) = -1
        g_pos = build_sliding_window_graphs(windowed_correlation(pos, win_size=100, win_stride=50))
        g_neg = build_sliding_window_graphs(windowed_correlation(neg, win_size=100, win_stride=50))
        wp = sorted(d["weight"] for _, _, d in g_pos[0].edges(data=True))
        wn = sorted(d["weight"] for _, _, d in g_neg[0].edges(data=True))
        np.testing.assert_allclose(wp, wn, atol=1e-9)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
