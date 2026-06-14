"""
Node-value topography for sensor or region layouts.

Renders a per-node scalar (for example a node's mean edge curvature or a centrality) as a
coloured scatter at 2D sensor or region positions. With electrode positions this is a scalp
topography; with ROI centroids it is a cortical map. True interpolated scalp maps with head
geometry are MNE's job (``mne.viz.plot_topomap``); this gives the position-aware scatter that
shares HyPhi's curvature colormap and does not depend on a montage.
"""

# %% Import
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .style import CURVATURE_CMAP, curvature_norm

__all__ = [
    "plot_node_topography",
]

_XY_NDIM = 2  # positions are 2D
_N_COORDS = 2  # each position has an x and a y


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def plot_node_topography(
    values: np.ndarray,
    positions: np.ndarray,
    ax: Any = None,
    node_size: int = 200,
    diverging: bool = True,
    title: str = "Node topography",
) -> Any:
    """
    Plot per-node values at 2D positions.

    Parameters
    ----------
    values : array-like
        One scalar per node.
    positions : array-like
        Node positions of shape ``(n_nodes, 2)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    node_size : int
        Marker size.
    diverging : bool
        If True, use the shared diverging curvature colormap centred on 0; otherwise a
        sequential colormap.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the topography.

    """
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != _XY_NDIM or positions.shape[1] != _N_COORDS or positions.shape[0] != values.shape[0]:
        msg = f"positions must be (n_nodes, 2) matching values; got {positions.shape} and {values.shape}"
        raise ValueError(msg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if diverging:
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], c=values, cmap=CURVATURE_CMAP, norm=curvature_norm(values), s=node_size
        )
    else:
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=values, cmap="viridis", s=node_size)
    fig.colorbar(scatter, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
