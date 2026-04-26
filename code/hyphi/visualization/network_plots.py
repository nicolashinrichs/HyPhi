"""
Network plotting utilities for HyPhi.

Complements ``curvature_visualisation.py`` with lightweight, layout-aware
plotting helpers for ad-hoc inspection of inter-brain graphs and their
Forman-Ricci curvature values.  Every function returns a ``matplotlib``
figure so downstream code can further customise or save it.

Origins: adapted from the ``network_checks_v2`` exploration notebook used
during Brainhack 2026.

Years: 2026
"""

from __future__ import annotations

# %% Import
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

__all__ = [
    "plot_curvature_distribution",
    "plot_curvature_network",
    "plot_curvature_network_layouts",
    "plot_network",
    "plot_weight_distribution",
]

_VALID_LAYOUTS = ("spring", "kamada_kawai", "spectral", "circular", "shell")


# %% Helpers >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _get_layout_positions(G: nx.Graph, layout: str = "spring", seed: int = 42) -> dict:
    """Dispatch to the chosen ``networkx`` layout; raises on unknown names."""
    if G.number_of_nodes() == 0:
        return {}
    if layout == "spring":
        return nx.spring_layout(G, seed=seed)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if layout == "spectral":
        return nx.spectral_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "shell":
        return nx.shell_layout(G)
    raise ValueError(f"Unknown layout {layout!r}; choose from {_VALID_LAYOUTS}.")


def _safe_color_limits(values: Iterable[float]) -> tuple[float, float]:
    """Return ``(vmin, vmax)`` with a tiny offset when the values are constant."""
    arr = np.asarray(list(values), dtype=float)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


# %% Plot functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def plot_weight_distribution(
    G: nx.Graph,
    bins: int = 20,
    figsize: tuple[float, float] = (7, 4),
    title: str = "Distribution of edge weights",
) -> Figure:
    """
    Histogram of edge weights.

    Parameters
    ----------
    G : nx.Graph
        Weighted graph (``"weight"`` edge attribute).
    bins : int
        Histogram bins.
    figsize : tuple
        Matplotlib figsize.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Raises
    ------
    ValueError
        If the graph carries no ``"weight"`` attributes.

    """
    weights = nx.get_edge_attributes(G, "weight")
    weight_values = np.array(list(weights.values()))
    if len(weight_values) == 0:
        raise ValueError("No edge weights found to plot.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(weight_values, bins=bins)
    ax.set_xlabel("Edge weight")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_network(
    G: nx.Graph,
    layout: str = "spring",
    figsize: tuple[float, float] = (10, 10),
    title: str | None = None,
    with_labels: bool = True,
    node_size: int = 300,
    font_size: int = 8,
    width: float = 1.5,
    seed: int = 42,
) -> Figure:
    """Draw a graph using the requested layout; returns the Figure."""
    pos = _get_layout_positions(G, layout=layout, seed=seed)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_size=node_size,
        font_size=font_size,
        width=width,
        ax=ax,
    )
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_curvature_network(
    G: nx.Graph,
    curvature_attr: str = "formanCurvature",
    layout: str = "spring",
    figsize: tuple[float, float] = (10, 10),
    title: str | None = None,
    with_labels: bool = True,
    node_size: int = 300,
    font_size: int = 8,
    width: float = 2.0,
    seed: int = 42,
    cmap=None,
) -> Figure:
    """
    Draw a graph with edges coloured by a curvature attribute.

    Parameters
    ----------
    G : nx.Graph
        Graph whose edges carry ``curvature_attr``.
    curvature_attr : str
        Edge attribute name (default ``"formanCurvature"``).
    layout : str
        One of :data:`_VALID_LAYOUTS`.
    figsize : tuple
        Matplotlib figsize.
    title : str, optional
        Plot title.
    with_labels : bool
        Whether to draw node labels.
    node_size, font_size, width : numeric
        Styling.
    seed : int
        Seed for stochastic layouts.
    cmap : matplotlib.colors.Colormap, optional
        Defaults to ``plt.cm.coolwarm``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Raises
    ------
    ValueError
        If the graph has no edges to colour.

    """
    cmap = cmap if cmap is not None else plt.cm.coolwarm
    pos = _get_layout_positions(G, layout=layout, seed=seed)

    edge_curvs = [data[curvature_attr] for _, _, data in G.edges(data=True)]
    if len(edge_curvs) == 0:
        raise ValueError("No edges found to colour by curvature.")

    vmin, vmax = _safe_color_limits(edge_curvs)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_curvs,
        edge_cmap=cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
        width=width,
        ax=ax,
    )
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)

    fig.colorbar(edges, ax=ax, label=curvature_attr)
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_curvature_network_layouts(
    G: nx.Graph,
    curvature_attr: str = "formanCurvature",
    layouts: tuple[str, ...] = ("spring", "kamada_kawai", "circular"),
    figsize: tuple[float, float] = (18, 6),
    with_labels: bool = False,
    node_size: int = 300,
    node_alpha: float = 0.6,
    font_size: int = 8,
    width: float = 2.0,
    seed: int = 42,
    cmap=None,
    suptitle: str | None = "Curvature-coloured network in multiple layouts",
) -> Figure:
    """
    Side-by-side curvature-coloured views of the same graph across multiple layouts.

    Useful for quickly checking whether high-curvature edges localise
    structurally (i.e., the same edges stand out regardless of layout).

    """
    cmap = cmap if cmap is not None else plt.cm.coolwarm

    edge_curvs = [data[curvature_attr] for _, _, data in G.edges(data=True)]
    if len(edge_curvs) == 0:
        raise ValueError("No edges found to colour by curvature.")

    vmin, vmax = _safe_color_limits(edge_curvs)

    fig, axes = plt.subplots(1, len(layouts), figsize=figsize, constrained_layout=True)
    if len(layouts) == 1:
        axes = [axes]

    for ax, layout in zip(axes, layouts):
        pos = _get_layout_positions(G, layout=layout, seed=seed)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=node_alpha, ax=ax)
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_curvs,
            edge_cmap=cmap,
            edge_vmin=vmin,
            edge_vmax=vmax,
            width=width,
            ax=ax,
        )
        if with_labels:
            nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
        ax.set_title(layout.replace("_", " ").title())
        ax.set_axis_off()

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=curvature_attr, shrink=0.9)

    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig


def plot_curvature_distribution(
    curvatures: np.ndarray,
    bins: int = 20,
    figsize: tuple[float, float] = (7, 4),
    title: str = "Distribution of Forman-Ricci curvature",
) -> Figure:
    """Histogram of curvature values."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(np.asarray(curvatures), bins=bins)
    ax.set_xlabel("Forman-Ricci curvature")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
