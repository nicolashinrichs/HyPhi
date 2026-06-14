"""
Tests for the viz package.

Render every figure helper on a headless backend and check it returns a Matplotlib Figure (or
animation) and that the curvature normalisation is symmetric and does not clip out-of-range
curvature, the bug the centralised colormap fixes.
"""

# %% Import
import matplotlib

matplotlib.use("Agg")  # headless backend, no display required

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from hyphi.modeling.graph_curvatures import compute_frc
from hyphi.viz import (
    animate_curvature_entropy,
    curvature_norm,
    plot_curvature_entropy,
    plot_dyadic_graph,
    plot_node_topography,
)
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestCurvatureNorm:
    """The shared curvature normalisation is symmetric about 0 and does not clip."""

    def test_symmetric_and_no_clip(self):
        """Limits are symmetric and cover the most extreme value, even beyond [-1, 1]."""
        extreme = 5.0
        norm = curvature_norm([-4.0, -1.0, 0.0, 2.0, extreme])
        assert norm.vcenter == 0.0
        assert norm.vmin == -extreme
        assert norm.vmax == extreme

    def test_all_zero_falls_back_to_unit(self):
        """An all-zero (or empty) input falls back to a unit range rather than erroring."""
        norm = curvature_norm([0.0, 0.0, 0.0])
        assert norm.vmin == -1.0
        assert norm.vmax == 1.0


class TestFigures:
    """Every plotting function returns a Figure on a headless backend."""

    def test_plot_dyadic_graph(self):
        """A dyadic graph with curvature renders to a Figure."""
        graph = compute_frc(nx.complete_graph(6), method="1d")
        fig = plot_dyadic_graph(graph, split=3)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_node_topography(self):
        """Per-node values at 2D positions render to a Figure."""
        rng = np.random.default_rng(0)
        fig = plot_node_topography(rng.standard_normal(8), rng.standard_normal((8, 2)))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_node_topography_rejects_bad_positions(self):
        """Mismatched positions raise a clear error."""
        with pytest.raises(ValueError, match="positions must be"):
            plot_node_topography(np.zeros(4), np.zeros((3, 2)))

    def test_plot_curvature_entropy(self):
        """An entropy series renders to a Figure."""
        fig = plot_curvature_entropy(np.array([0.1, 0.5, 0.3, 0.8]))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestDyadicCurvature:
    """plot_dyadic_graph auto-detects the curvature attribute, draws every edge, and adds a colorbar."""

    def test_orc_attribute_is_auto_detected_and_coloured(self):
        """A graph carrying ricciCurvature (ORC) is coloured by default, not silently drawn grey."""
        graph = nx.complete_graph(6)
        nx.set_edge_attributes(graph, {e: float(i) - 3.0 for i, e in enumerate(graph.edges)}, "ricciCurvature")
        fig = plot_dyadic_graph(graph, split=3)  # curvature=None -> auto-detect
        n_axes_with_colorbar = 2  # the main axes plus the colorbar axes
        assert len(fig.axes) >= n_axes_with_colorbar
        plt.close(fig)

    def test_missing_explicit_attribute_warns(self):
        """Asking for an absent attribute while another curvature is present warns rather than silently greys."""
        graph = compute_frc(nx.complete_graph(6), method="1d")  # carries formanCurvature
        with pytest.warns(UserWarning, match="ricciCurvature"):
            fig = plot_dyadic_graph(graph, curvature="ricciCurvature")
        plt.close(fig)

    def test_draws_every_edge_with_partial_curvature(self):
        """Edges lacking the curvature attribute are still drawn (grey), not dropped."""
        graph = nx.path_graph(6)  # 5 edges
        edges = list(graph.edges)
        nx.set_edge_attributes(graph, dict.fromkeys(edges[:3], 1.0), "formanCurvature")
        fig = plot_dyadic_graph(graph, split=3)
        drawn = sum(len(c.get_segments()) for c in fig.axes[0].collections if isinstance(c, LineCollection))
        assert drawn == graph.number_of_edges()  # all 5 edges drawn, coloured + grey
        plt.close(fig)


class TestAnimation:
    """The animation returns a FuncAnimation that renders frames without error."""

    def test_animate_curvature_entropy(self):
        """The animation object is a FuncAnimation and a frame can be drawn."""
        entropy = np.array([0.1, 0.4, 0.2, 0.9, 0.5])
        anim = animate_curvature_entropy(entropy)
        assert isinstance(anim, FuncAnimation)
        # Drawing the canvas exercises the frame-update function.
        anim._fig.canvas.draw()
        plt.close(anim._fig)

    def test_animate_empty_series_does_not_crash(self):
        """An empty series yields a FuncAnimation (default limits), matching the static plot's graceful handling."""
        anim = animate_curvature_entropy(np.array([]))
        assert isinstance(anim, FuncAnimation)
        plt.close(anim._fig)

    def test_length_mismatch_raises(self):
        """A mismatched x length is rejected up front by both the static plot and the animation."""
        with pytest.raises(ValueError, match="must match"):
            plot_curvature_entropy(np.array([0.1, 0.2, 0.3]), x=np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="must match"):
            animate_curvature_entropy(np.array([0.1, 0.2, 0.3]), x=np.array([0.0, 1.0]))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
