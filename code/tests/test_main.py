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
from hyphi.main import demo_connectivity, main, run_pipeline

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
