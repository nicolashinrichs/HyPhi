"""Tests for the HyPyP / MNE preprocessing adapter.

Checks that a preprocessed, band-limited signal (a plain array or an MNE Epochs object) flows
through epochs_to_phase into a (n_channels, n_times) phase array, and through
epochs_to_plv_graphs into a sliding-window PLV graph series.
"""

# %% Import
import logging

import networkx as nx
import numpy as np
import pytest
from hyphi.modeling.windowing import sliding_window_plv
from hyphi.preprocessing import epochs_to_phase, epochs_to_plv_graphs
from scipy.signal import hilbert

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


_PLV_TOL = 1e-9  # floating-point slack on the [0, 1] PLV range
_PLV_LOCKED_MIN = 0.9  # a phase-locked channel pair has PLV well above this


def _narrowband_signal(n_channels=4, n_times=512, freq=10.0, fs=128.0, seed=0):
    """A band-limited multichannel signal: sinusoids at one frequency with random phase offsets."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / fs
    offsets = rng.uniform(0, 2 * np.pi, size=n_channels)
    return np.stack([np.cos(2 * np.pi * freq * t + off) for off in offsets])


class TestEpochsToPhase:
    """epochs_to_phase returns a (n_channels, n_times) phase array within [-pi, pi]."""

    def test_array_1d_gets_channel_axis(self):
        """A bare 1D array is treated as a single channel: shape (1, n_times)."""
        signal = _narrowband_signal(n_channels=1)[0]  # genuine 1D (n_times,)
        phase = epochs_to_phase(signal)
        assert phase.shape == (1, signal.shape[0])

    def test_array_2d_shape_and_range(self):
        """A 2D array yields a phase array of the same shape, in radians."""
        signal = _narrowband_signal()
        phase = epochs_to_phase(signal)
        assert phase.shape == signal.shape
        assert np.all(phase >= -np.pi) and np.all(phase <= np.pi)

    def test_phase_is_real(self):
        """The phase is a real array (np.angle), not the raw complex analytic signal."""
        phase = epochs_to_phase(_narrowband_signal())
        assert not np.iscomplexobj(phase)

    def test_array_3d_epochs_concatenated(self):
        """A 3D (epochs) array concatenates per-epoch phase along time."""
        epoched = np.stack([_narrowband_signal(seed=s) for s in range(3)])  # (3, 4, 512)
        phase = epochs_to_phase(epoched)
        assert phase.shape == (4, 3 * 512)

    def test_3d_phase_is_per_epoch_not_joined(self):
        """3D phase is computed PER EPOCH (no Hilbert across joins): a join-then-Hilbert differs."""
        epoched = np.stack([_narrowband_signal(seed=s) for s in range(3)])  # (3, 4, 512)
        out = epochs_to_phase(epoched)
        per_epoch = np.concatenate([np.angle(hilbert(epoched[e], axis=-1)) for e in range(3)], axis=-1)
        np.testing.assert_allclose(out, per_epoch)
        # One Hilbert over the flattened signal would smear the epoch joins -> different result.
        joined = np.angle(hilbert(np.concatenate(list(epoched), axis=-1), axis=-1))
        assert not np.allclose(out, joined)

    def test_recovers_known_frequency(self):
        """The instantaneous phase of a 10 Hz sinusoid advances at about 2*pi*10 rad/s."""
        fs = 128.0
        signal = _narrowband_signal(n_channels=1, n_times=1024, freq=10.0, fs=fs)
        phase = epochs_to_phase(signal)[0]
        # Median instantaneous angular frequency, ignoring the +-2pi wraps.
        dphi = np.diff(np.unwrap(phase)) * fs
        assert np.median(dphi) == pytest.approx(2 * np.pi * 10.0, rel=0.05)


class TestMNEAdapter:
    """The adapter accepts an MNE Epochs object and drives the full PLV-graph path."""

    def _epochs(self):
        mne = pytest.importorskip("mne")
        mne.set_log_level("ERROR")
        data = np.stack([_narrowband_signal(n_channels=6, n_times=256, seed=s) for s in range(2)])
        info = mne.create_info(ch_names=[f"c{i}" for i in range(6)], sfreq=128.0, ch_types="eeg")
        return mne.EpochsArray(data, info)

    def test_epochs_to_phase(self):
        """An MNE EpochsArray yields a (n_channels, n_epochs*n_times) phase array."""
        phase = epochs_to_phase(self._epochs())
        assert phase.shape == (6, 2 * 256)
        assert np.all(np.isfinite(phase))

    def test_epochs_to_plv_graphs(self):
        """The plug-and-play path returns one PLV graph per window; nodes = n_channels, weights in [0, 1]."""
        graphs = epochs_to_plv_graphs(self._epochs(), win_size=128, win_stride=64)
        assert len(graphs) > 0
        for graph in graphs:
            assert isinstance(graph, nx.Graph)
            assert graph.number_of_nodes() == 6
            # PLV weights are in [0, 1]; assert it so a refactor that drops/renames the weight
            # attribute (which downstream curvature keys on) cannot pass vacuously.
            weights = [data["weight"] for _, _, data in graph.edges(data=True)]
            assert weights
            assert all(-_PLV_TOL <= w <= 1 + _PLV_TOL for w in weights)


class TestEpochWindowing:
    """epochs_to_plv_graphs windows WITHIN each epoch, so no PLV window straddles an epoch join."""

    def test_windows_do_not_straddle_epoch_joins(self):
        """Phase-locked channels with an alternating cross-epoch lag give PLV ~ 1.0 in every window."""
        # Pre-fix, windows straddling a join mixed the 0/pi lag and read a spurious ~0.0.
        fs, n_times = 128.0, 256
        t = np.arange(n_times) / fs
        epochs = []
        for k in range(5):
            lag = 0.0 if k % 2 == 0 else np.pi  # the cross-epoch relation flips at each join
            epochs.append(np.stack([np.cos(2 * np.pi * 10 * t), np.cos(2 * np.pi * 10 * t + lag)]))
        epoched = np.stack(epochs)  # (5, 2, 256)
        graphs = epochs_to_plv_graphs(epoched, win_size=128, win_stride=64)
        weights = [data["weight"] for g in graphs for _, _, data in g.edges(data=True)]
        # 5 epochs x 3 windows each (256 samples, win=128, stride=64): a skip/duplicate-epoch bug
        # would change this count.
        assert len(graphs) == 5 * 3
        # Within every epoch the two channels are perfectly phase-locked -> PLV ~ 1.0 everywhere.
        assert all(w > _PLV_LOCKED_MIN for w in weights)

    def test_matches_per_epoch_reference(self):
        """3D output equals per-epoch sliding_window_plv concatenated in order (kills skip/reorder bugs)."""
        fs, n_times, freq = 128.0, 256, 10.0
        t = np.arange(n_times) / fs
        epochs = []
        for s in range(4):
            jitter = np.random.default_rng(s)  # distinct per epoch -> distinct PLV pattern
            ch0 = np.cos(2 * np.pi * freq * t)
            ch1 = np.cos(2 * np.pi * freq * t + 0.5 * jitter.standard_normal(n_times))
            ch2 = np.cos(2 * np.pi * freq * t + 1.0 * jitter.standard_normal(n_times))
            epochs.append(np.stack([ch0, ch1, ch2]))
        epoched = np.stack(epochs)
        got = epochs_to_plv_graphs(epoched, win_size=128, win_stride=64)
        expected = []
        for e in range(epoched.shape[0]):
            expected.extend(sliding_window_plv(np.angle(hilbert(epoched[e], axis=-1)), 128, 64))
        assert len(got) == len(expected)
        for g, exp in zip(got, expected, strict=True):
            gw = sorted(d["weight"] for _, _, d in g.edges(data=True))
            ew = sorted(d["weight"] for _, _, d in exp.edges(data=True))
            np.testing.assert_allclose(gw, ew, atol=1e-12)

    def test_win_size_exceeding_epoch_warns(self, caplog):
        """If win_size exceeds the epoch length, no window fits within an epoch -> warn, not silent."""
        epoched = np.stack([_narrowband_signal(n_channels=3, n_times=64, seed=s) for s in range(2)])
        with caplog.at_level(logging.WARNING):
            graphs = epochs_to_plv_graphs(epoched, win_size=128, win_stride=64)
        assert graphs == []
        assert "exceeds the epoch length" in caplog.text

    def test_rejects_more_channels_than_times(self):
        """A window/epoch with more channels than time samples is rejected loudly (no silent transpose)."""
        epoched = np.stack([_narrowband_signal(n_channels=64, n_times=40, seed=s) for s in range(2)])  # (2,64,40)
        with pytest.raises(ValueError, match="more channels"):
            epochs_to_plv_graphs(epoched, win_size=20, win_stride=10)
        flat = _narrowband_signal(n_channels=64, n_times=40)  # 2-D, channels > times
        with pytest.raises(ValueError, match="more channels"):
            epochs_to_plv_graphs(flat, win_size=20, win_stride=10)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
