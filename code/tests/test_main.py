"""
Tests for the end-to-end pipeline entry point.

Checks the demo connectivity, the run_pipeline function, and that the command-line entry point
writes the entropy series, quantiles, and a figure.
"""

# %% Import
import json

import matplotlib

matplotlib.use("Agg")  # headless backend for the figure the CLI saves

import numpy as np
import pytest
from hyphi.main import demo_connectivity, main, run_eeg_pipeline, run_pipeline

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestRunPipeline:
    """run_pipeline takes connectivity to a curvature-entropy series."""

    def test_demo_connectivity_shape(self):
        """The demo connectivity is a (windows, nodes, nodes) tensor."""
        connectivity = demo_connectivity(n_nodes=15, n_windows=6)
        assert connectivity.shape == (6, 15, 15)

    def test_pipeline_outputs(self):
        """run_pipeline returns aligned graphs, a finite entropy trace, and quantiles."""
        n_windows, n_quantiles = 8, 5
        connectivity = demo_connectivity(n_nodes=20, n_windows=n_windows)
        results = run_pipeline(connectivity)
        assert set(results) == {"graphs", "entropy", "quantiles"}
        assert len(results["graphs"]) == n_windows
        assert results["entropy"].shape == (n_windows,)
        assert results["quantiles"].shape == (n_windows, n_quantiles)
        assert np.all(np.isfinite(results["entropy"]))

    def test_pipeline_accepts_single_window(self):
        """A single 2D connectivity matrix is treated as one window."""
        connectivity = demo_connectivity(n_nodes=12, n_windows=1)[0]  # (12, 12)
        results = run_pipeline(connectivity)
        assert len(results["graphs"]) == 1


class TestRunEEGPipeline:
    """run_eeg_pipeline takes preprocessed dual-EEG end to end to the same curvature-entropy series."""

    @staticmethod
    def _synthetic_dual_eeg(n_per_brain=8, n_times=2000, seed=0):
        """Build a synthetic (2*n_per_brain, n_times) inter-brain montage with real coupling structure.

        Brain B's first ``n_coupled`` channels share brain A's 10 Hz carrier (high inter-brain PLV); the
        rest run on independent off-band carriers (low inter-brain PLV). The coupling is genuine, not a
        constant phase offset (PLV is invariant to a constant offset), so a coupled inter-brain edge is
        distinguishable from an uncoupled one.
        """
        n_coupled = 3
        sfreq = 250.0
        rng = np.random.default_rng(seed)
        t = np.arange(n_times) / sfreq
        carrier = 2 * np.pi * 10.0 * t  # 10 Hz carrier shared by brain A and brain B's coupled channels
        brain_a = np.stack([np.sin(carrier + rng.normal(0, 0.1, n_times) + c) for c in range(n_per_brain)])
        coupled = [np.sin(carrier + rng.normal(0, 0.1, n_times) + 0.5) for _ in range(n_coupled)]
        indep_freqs = np.linspace(15.0, 22.0, n_per_brain - n_coupled)  # off-band, so PLV with A stays low
        independent = [np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in indep_freqs]
        brain_b = np.stack(coupled + independent)
        return np.vstack([brain_a, brain_b])  # (2*n_per_brain, n_times), time on the last axis

    def test_eeg_pipeline_end_to_end(self):
        """A raw (channels, time) dual-EEG array becomes a finite RNE trace + quantiles over inter-brain graphs."""
        n_quantiles = 5
        eeg = self._synthetic_dual_eeg()  # (16, 2000)
        win_size, win_stride = 200, 100
        results = run_eeg_pipeline(eeg, win_size=win_size, win_stride=win_stride)

        expected_windows = (eeg.shape[-1] - win_size) // win_stride + 1
        assert set(results) == {"graphs", "entropy", "quantiles"}
        assert len(results["graphs"]) == expected_windows
        # Nodes are the 16 inter-brain channels, not time samples (issue #92 orientation).
        assert results["graphs"][0].number_of_nodes() == eeg.shape[0]
        assert results["entropy"].shape == (expected_windows,)
        assert np.all(np.isfinite(results["entropy"]))
        assert results["quantiles"].shape == (expected_windows, n_quantiles)

    def test_eeg_pipeline_shares_validation_with_run_pipeline(self):
        """A bad estimator is rejected with the same clear error run_pipeline raises."""
        eeg = self._synthetic_dual_eeg(n_per_brain=4, n_times=800)
        with pytest.raises(ValueError, match=r"Unknown"):
            run_eeg_pipeline(eeg, win_size=200, win_stride=100, entropy_method="not_an_estimator")

    def test_eeg_pipeline_validates_before_the_plv_pass(self, monkeypatch):
        """Validation runs before the expensive PLV pass: a bad estimator never reaches epochs_to_plv_graphs."""

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("epochs_to_plv_graphs ran despite invalid validation; fail-fast ordering broke")

        monkeypatch.setattr("hyphi.main.epochs_to_plv_graphs", _fail_if_called)
        eeg = self._synthetic_dual_eeg(n_per_brain=4, n_times=800)
        with pytest.raises(ValueError, match=r"Unknown"):
            run_eeg_pipeline(eeg, win_size=200, win_stride=100, entropy_method="not_an_estimator")

    def test_eeg_pipeline_warns_when_window_exceeds_recording(self, caplog):
        """A win_size larger than the 2D recording warns and returns a shape-consistent empty result."""
        n_quantiles = 5
        eeg = self._synthetic_dual_eeg(n_per_brain=4, n_times=300)  # (8, 300)
        with caplog.at_level("WARNING"):
            results = run_eeg_pipeline(eeg, win_size=500, win_stride=100)
        assert "exceeds the recording length" in caplog.text
        assert len(results["graphs"]) == 0
        # The degenerate dict keeps the documented contract: entropy (0,), quantiles (0, 5).
        assert results["entropy"].shape == (0,)
        assert results["quantiles"].shape == (0, n_quantiles)

    def test_eeg_pipeline_recovers_inter_brain_coupling_structure(self):
        """A coupled inter-brain edge carries materially higher PLV weight than an uncoupled one.

        Guards the inter-brain orientation: a transpose or wrong slice that still produced the right node
        count and a finite entropy would scramble which edges are coupled and not survive this.
        """
        margin = 0.2
        n_per_brain = 8
        eeg = self._synthetic_dual_eeg(n_per_brain=n_per_brain, n_times=2000)
        results = run_eeg_pipeline(eeg, win_size=500, win_stride=500)
        graph = results["graphs"][0]
        a_channel = 0
        coupled_b = n_per_brain  # brain B's first channel (shares brain A's carrier)
        uncoupled_b = 2 * n_per_brain - 1  # brain B's last channel (independent off-band carrier)
        coupled_weight = graph[a_channel][coupled_b]["weight"]
        uncoupled_weight = graph[a_channel][uncoupled_b]["weight"]
        assert coupled_weight > uncoupled_weight + margin


class TestMainCLI:
    """The command-line entry point runs and writes its outputs."""

    def test_writes_outputs(self, tmp_path):
        """The CLI writes entropy.csv, quantiles.csv, entropy.png, and environment.json to the output directory."""
        out = tmp_path / "run"
        main(["--output", str(out), "--nodes", "15", "--windows", "5"])
        assert (out / "entropy.csv").exists()
        assert (out / "quantiles.csv").exists()
        assert (out / "entropy.png").exists()
        # The entropy CSV has one value per window.
        entropy = np.loadtxt(out / "entropy.csv", delimiter=",")
        assert entropy.shape == (5,)

    def test_writes_environment_record(self, tmp_path):
        """The CLI records the run environment as environment.json alongside the outputs."""
        out = tmp_path / "run"
        n_windows = 4
        main(["--output", str(out), "--nodes", "12", "--windows", str(n_windows)])
        environment = json.loads((out / "environment.json").read_text())
        assert set(environment) >= {"hyphi_version", "python_version", "platform", "args"}
        assert environment["args"]["windows"] == n_windows


class TestValidation:
    """run_pipeline and demo_connectivity reject malformed input loudly, not with a deep library traceback."""

    def test_rejects_unknown_curvature_method(self):
        """A typo'd curvature method is rejected up front, not as an opaque downstream KeyError."""
        with pytest.raises(ValueError, match="Unknown curvature method"):
            run_pipeline(demo_connectivity(n_nodes=10, n_windows=3), curvature_method="bogus")

    def test_rejects_unknown_entropy_method(self):
        """A typo'd entropy estimator fails fast (before the curvature pass) with a clear error."""
        with pytest.raises(ValueError, match="Unknown"):
            run_pipeline(demo_connectivity(n_nodes=10, n_windows=3), entropy_method="not_an_estimator")

    def test_rejects_bad_connectivity_shape(self):
        """A non-(square 2D / 3D) connectivity array is rejected, not crashed deep in the graph builder."""
        with pytest.raises(ValueError, match="windows, nodes, nodes"):
            run_pipeline(np.zeros(5))  # 1D
        with pytest.raises(ValueError, match="square"):
            run_pipeline(np.zeros((4, 6)))  # non-square matrix

    def test_demo_rejects_degenerate_params(self):
        """demo_connectivity rejects too-few nodes (Watts-Strogatz degree) and a non-positive window count."""
        with pytest.raises(ValueError, match="Watts-Strogatz degree"):
            demo_connectivity(n_nodes=3)
        with pytest.raises(ValueError, match="at least 1"):
            demo_connectivity(n_nodes=10, n_windows=0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
