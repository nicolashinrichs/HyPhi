"""
Dyadic graph layout for hyperscanning networks.

Renders an inter-brain network with the two brains side by side: brain A on the left, brain B on
the right, intra-brain edges within each half and inter-brain edges spanning the gap. Edges are
coloured by curvature when the graph carries it, using the shared diverging curvature colormap.
"""

# %% Import
from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable

from .style import CURVATURE_CMAP, curvature_norm

__all__ = [
    "plot_dyadic_graph",
]

# Edge attributes the curvature functions write: FRC/AFRC -> formanCurvature, ORC -> ricciCurvature.
_KNOWN_CURVATURE_ATTRS = ("formanCurvature", "ricciCurvature")


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def plot_dyadic_graph(
    graph: nx.Graph,
    split: int | None = None,
    curvature: str | None = None,
    node_size: int = 120,
    ax: Any = None,
) -> Any:
    """
    Draw a dyadic inter-brain graph with the two brains side by side.

    Parameters
    ----------
    graph : nx.Graph
        Inter-brain graph. Nodes are ordered so the first ``split`` belong to brain A and the
        rest to brain B.
    split : int, optional
        Number of brain-A nodes; defaults to half the nodes.
    curvature : str, optional
        Edge attribute used to colour edges. When omitted, the first known curvature attribute
        present on the edges is auto-detected (``formanCurvature`` for FRC/AFRC, ``ricciCurvature``
        for ORC), so an ORC graph is not silently drawn uncoloured.
    node_size : int
        Node marker size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the dyadic layout. A colorbar is attached when edges are coloured.

    """
    nodes = sorted(graph.nodes())
    n = len(nodes)
    if split is None:
        split = n // 2
    a_nodes = nodes[:split]
    b_nodes = nodes[split:]

    # Brain A in a left column, brain B in a right column, each spread vertically.
    pos: dict[Any, tuple[float, float]] = {}
    for i, node in enumerate(a_nodes):
        pos[node] = (0.0, -float(i))
    for i, node in enumerate(b_nodes):
        pos[node] = (1.0, -float(i))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure

    # Resolve the curvature attribute: auto-detect a known one when unspecified, and warn if an
    # explicit attribute is absent while the graph actually carries a different known curvature
    # (otherwise an ORC graph asked for "formanCurvature" would be drawn uncoloured with no signal).
    if curvature is None:
        curvature = next((attr for attr in _KNOWN_CURVATURE_ATTRS if nx.get_edge_attributes(graph, attr)), None)
    elif not nx.get_edge_attributes(graph, curvature):
        present = [attr for attr in _KNOWN_CURVATURE_ATTRS if nx.get_edge_attributes(graph, attr)]
        if present:
            warnings.warn(
                f"plot_dyadic_graph: no edge carries {curvature!r}, but the graph has {present}; edges will be "
                "drawn uncoloured. Pass curvature= one of those to colour them.",
                stacklevel=2,
            )

    curvatures = nx.get_edge_attributes(graph, curvature) if curvature else {}
    coloured = list(curvatures)
    coloured_set = set(coloured)
    # Draw EVERY edge: the attributed ones coloured by curvature, the rest in grey, so no topology is
    # lost when only some edges carry curvature (e.g. curved intra-brain edges + uncurved inter-brain).
    other = [edge for edge in graph.edges if edge not in coloured_set]
    cmap = plt.get_cmap(CURVATURE_CMAP)
    if coloured:
        values = np.asarray([curvatures[edge] for edge in coloured], dtype=float)
        norm = curvature_norm(values)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=coloured,
            edge_color=values,
            edge_cmap=cmap,
            edge_vmin=norm.vmin,
            edge_vmax=norm.vmax,
            ax=ax,
        )
        # A colorbar with the same shared norm so the symmetric, data-driven scale is readable and
        # comparable across figures.
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.8, label=curvature)
    if other:
        nx.draw_networkx_edges(graph, pos, edgelist=other, alpha=0.4, ax=ax)

    nx.draw_networkx_nodes(graph, pos, nodelist=a_nodes, node_color="tab:blue", node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=b_nodes, node_color="tab:orange", node_size=node_size, ax=ax)
    ax.set_title("Dyadic inter-brain network")
    ax.set_axis_off()
    return fig


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
