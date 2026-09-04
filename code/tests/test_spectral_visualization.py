"""
Smoke-tests for the three HyPhi visualization modules.

Covers:
  - network_plots    : plot_weight_distribution, plot_network, plot_curvature_network,
                       plot_curvature_distribution
  - curvature_visualization : visualize_graph_with_curvature, curvature_distribution,
                              node_to_partition, visualize_graph_partitions_markers,
                              visualize_graph_partitions_colors
  - gdd_frc_visualization   : plot_successive_gdd, plot_gdd_heatmap

All matplotlib tests use the Agg backend (headless, no display required).
Functions that strictly require plotly are skipped via pytest.importorskip.
"""

# %% Import
# NOTE: matplotlib.use("Agg") must precede ALL pyplot / visualization imports.
import matplotlib

matplotlib.use("Agg")

# matplotlib.pyplot must be imported after matplotlib.use("Agg") — intentional ordering.
from unittest.mock import patch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from hyphi.modeling.graph_curvatures import compute_frc
from hyphi.visualization.curvature_visualization import (
    curvature_distribution,
    node_to_partition,
    visualize_graph_partitions_colors,
    visualize_graph_partitions_markers,
    visualize_graph_with_curvature,
)
from hyphi.visualization.gdd_frc_visualization import (
    plot_gdd_heatmap,
    plot_successive_gdd,
)
from hyphi.visualization.network_plots import (
    plot_curvature_distribution,
    plot_curvature_network,
    plot_network,
    plot_weight_distribution,
)

# %% Module-level constants (avoids PLR2004 magic-value lint)
_N_NODES_TINY = 5
_N_EDGES_RING = 5
_GDD_MATRIX_SIZE = 4
_N_CURVATURE_VALUES = 6
_HIST_BINS = 5
_FORMAN_CURVATURE_VALUE = 0.5
_RICCI_CURVATURE_VALUE = 0.3
_WEIGHT_VALUE = 1.0
_SEED = 42
_N_NODES_FIRST_PARTITION = 3
_N_EXPECTED_PARTITIONED = 5
_COMMUNITY_INDEX_2 = 2


# %% Helpers


def _tiny_weighted_graph() -> nx.Graph:
    """Tiny cycle-5 graph with uniform edge weights."""
    G = nx.cycle_graph(_N_NODES_TINY)
    nx.set_edge_attributes(G, _WEIGHT_VALUE, "weight")
    return G


def _tiny_forman_graph() -> nx.Graph:
    """Tiny cycle-5 graph with 'formanCurvature' on every edge."""
    G = nx.cycle_graph(_N_NODES_TINY)
    nx.set_edge_attributes(G, _FORMAN_CURVATURE_VALUE, "formanCurvature")
    return G


def _tiny_ricci_graph() -> nx.Graph:
    """Tiny cycle-5 graph with 'ricciCurvature' on every edge."""
    G = nx.cycle_graph(_N_NODES_TINY)
    nx.set_edge_attributes(G, _RICCI_CURVATURE_VALUE, "ricciCurvature")
    return G


# %% network_plots tests


class TestPlotWeightDistribution:
    """Smoke-tests for plot_weight_distribution."""

    def test_returns_figure(self):
        """Function should return a matplotlib Figure without error."""
        G = _tiny_weighted_graph()
        fig = plot_weight_distribution(G, bins=_HIST_BINS)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_raises_on_no_weights(self):
        """Should raise ValueError when graph has no weight attributes."""
        G = nx.cycle_graph(_N_NODES_TINY)  # no weights
        with pytest.raises(ValueError, match="No edge weights"):
            plot_weight_distribution(G)


class TestPlotNetwork:
    """Smoke-tests for plot_network."""

    def test_returns_figure_spring(self, complete_graph_k5):
        """Spring layout should return a matplotlib Figure."""
        fig = plot_network(complete_graph_k5, layout="spring", seed=_SEED)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_returns_figure_circular(self, ring_lattice_c10):
        """Circular layout should also succeed."""
        fig = plot_network(ring_lattice_c10, layout="circular", seed=_SEED)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_raises_on_unknown_layout(self, complete_graph_k5):
        """An unrecognised layout name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown layout"):
            plot_network(complete_graph_k5, layout="nonexistent_layout")


class TestPlotCurvatureNetwork:
    """Smoke-tests for plot_curvature_network."""

    def test_returns_figure(self):
        """Forman-curvature network plot should return a Figure."""
        G = _tiny_forman_graph()
        fig = plot_curvature_network(G, curvature_attr="formanCurvature", seed=_SEED)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_raises_on_empty_edges(self):
        """Should raise ValueError when the graph has no edges."""
        G = nx.empty_graph(_N_NODES_TINY)
        nx.set_edge_attributes(G, _FORMAN_CURVATURE_VALUE, "formanCurvature")
        with pytest.raises(ValueError, match="No edges found"):
            plot_curvature_network(G)


class TestPlotCurvatureDistribution:
    """Smoke-tests for plot_curvature_distribution (network_plots version)."""

    def test_returns_figure(self):
        """Curvature histogram should return a Figure."""
        curvatures = np.linspace(-1.0, 1.0, _N_CURVATURE_VALUES)
        fig = plot_curvature_distribution(curvatures, bins=_HIST_BINS)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# %% curvature_visualization tests


class TestVisualizeGraphWithCurvature:
    """Smoke-tests for visualize_graph_with_curvature."""

    def test_saves_png(self, tmp_path):
        """When save_dir is given, function must write a PNG to that directory."""
        G = _tiny_ricci_graph()
        visualize_graph_with_curvature(G, name="smoke", save_dir=tmp_path)
        out = tmp_path / "graph_curvature_smoke.png"
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")

    def test_runs_without_save(self, monkeypatch):
        """Without save_dir the function should not raise (mocking plt.show)."""
        monkeypatch.setattr(plt, "show", lambda: None)
        G = _tiny_ricci_graph()
        visualize_graph_with_curvature(G, name="smoke_no_save", save_dir=None)
        plt.close("all")


class TestCurvatureDistribution:
    """Smoke-tests for curvature_distribution (curvature_visualization version)."""

    def test_saves_png(self, tmp_path):
        """With save=True and a save_path, a PNG file should appear."""
        data = [np.linspace(-1.0, 1.0, _N_CURVATURE_VALUES)]
        names = ["test_graph"]
        with patch("matplotlib.pyplot.show"):
            curvature_distribution(data, names, plot_name="curv_dist_test", save_path=tmp_path, save=True)
        out = tmp_path / "curv_dist_test.png"
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")


class TestNodeToPartition:
    """Known-answer tests for node_to_partition (pure function)."""

    def test_disjoint_communities(self):
        """Each node maps to the correct community index for non-overlapping partitions."""
        partitions = [[0, 1, 2], [3, 4], [5]]
        result = node_to_partition(partitions)
        # Community 0: nodes 0, 1, 2
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 0
        # Community 1: nodes 3, 4
        assert result[3] == 1
        assert result[4] == 1
        # Community 2: node 5
        assert result[5] == _COMMUNITY_INDEX_2

    def test_single_community(self):
        """A single community means every node maps to partition 0."""
        partitions = [["a", "b", "c"]]
        result = node_to_partition(partitions)
        assert result == {"a": 0, "b": 0, "c": 0}

    def test_empty_communities(self):
        """An empty list of communities should return an empty dict."""
        assert node_to_partition([]) == {}

    def test_node_count_matches(self):
        """The returned dict should have one entry per unique node."""
        partitions = [[10, 20], [30, 40, 50]]
        result = node_to_partition(partitions)
        assert len(result) == _N_EXPECTED_PARTITIONED


class TestVisualizeGraphPartitionsMarkers:
    """Smoke-tests for visualize_graph_partitions_markers."""

    def test_saves_png_forman_graph(self, tmp_path, complete_graph_k5):
        """Forman-curvature graph with partitions must write a PNG to save_path."""
        G = compute_frc(complete_graph_k5)
        partitions = [sorted(c) for c in nx.connected_components(G)]
        visualize_graph_partitions_markers(
            graph=G,
            partitions=partitions,
            name="markers_smoke",
            save=True,
            save_path=tmp_path,
            show=False,
        )
        out = tmp_path / "markers_smoke.png"
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")

    def test_saves_png_ricci_graph(self, tmp_path):
        """Ricci-curvature graph with multi-partition structure must also write a PNG."""
        G = _tiny_ricci_graph()
        # Two partitions: first three nodes vs. the rest
        partitions = [list(range(_N_NODES_FIRST_PARTITION)), list(range(_N_NODES_FIRST_PARTITION, _N_NODES_TINY))]
        visualize_graph_partitions_markers(
            graph=G,
            partitions=partitions,
            name="ricci_markers",
            save=True,
            save_path=tmp_path,
            show=False,
        )
        out = tmp_path / "ricci_markers.png"
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")


class TestVisualizeGraphPartitionsColors:
    """Smoke-tests for visualize_graph_partitions_colors."""

    def test_runs_without_error(self):
        """Function should run without raising (mocking plt.show and plt.savefig)."""
        G = _tiny_ricci_graph()
        partitions = [list(range(_N_NODES_FIRST_PARTITION)), list(range(_N_NODES_FIRST_PARTITION, _N_NODES_TINY))]
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
            visualize_graph_partitions_colors(G, partitions=partitions, name="colors_smoke", save=False)
        plt.close("all")


# %% gdd_frc_visualization tests


class TestPlotSuccessiveGdd:
    """Smoke-tests for plot_successive_gdd."""

    def test_returns_figure_and_axes(self):
        """Should return a (Figure, Axes) tuple without error."""
        successive = {0: 0.1, 1: 0.25, 2: 0.05}
        fig, _ = plot_successive_gdd(successive)
        assert fig is not None
        plt.close("all")

    def test_raises_on_empty_input(self):
        """Should raise ValueError for an empty mapping."""
        with pytest.raises(ValueError, match="empty"):
            plot_successive_gdd({})


class TestPlotGddHeatmap:
    """Smoke-tests for plot_gdd_heatmap."""

    def test_returns_figure_and_axes(self):
        """Square 2D distance matrix should render without error."""
        rng = np.random.default_rng(_SEED)
        mat = rng.random((_GDD_MATRIX_SIZE, _GDD_MATRIX_SIZE))
        mat = (mat + mat.T) / 2  # symmetric
        np.fill_diagonal(mat, 0.0)
        fig, _ = plot_gdd_heatmap(mat)
        assert fig is not None
        plt.close("all")

    def test_with_labels(self):
        """Labels of correct length should not raise."""
        mat = np.eye(_GDD_MATRIX_SIZE)
        labels = [f"g{i}" for i in range(_GDD_MATRIX_SIZE)]
        fig, _ = plot_gdd_heatmap(mat, labels=labels)
        assert fig is not None
        plt.close("all")

    def test_raises_on_non_square(self):
        """Non-square matrices must raise ValueError."""
        _n_rows = 3
        mat = np.zeros((_n_rows, _GDD_MATRIX_SIZE))
        with pytest.raises(ValueError, match="square 2D"):
            plot_gdd_heatmap(mat)

    def test_raises_on_label_length_mismatch(self):
        """Wrong number of labels must raise ValueError."""
        _n_wrong_labels = 2
        mat = np.eye(_GDD_MATRIX_SIZE)
        with pytest.raises(ValueError, match="labels length"):
            plot_gdd_heatmap(mat, labels=["a"] * _n_wrong_labels)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
