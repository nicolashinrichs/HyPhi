"""
Coherent visualization for HyPhi.

A single, consistent figure layer for dyadic hyperscanning results. Every function returns a
Matplotlib ``Figure`` (or, for the animation, a ``FuncAnimation``) rather than showing or saving
in place, so callers compose and save figures themselves. Curvature is coloured through one
shared diverging, non-clipping normalisation (``style.curvature_norm``), replacing the scattered
fixed ``vmin=-1, vmax=1`` normalisations in the older ``visualization`` modules.

This package provides the dyadic side-by-side layout, the node topography, and the
curvature-entropy time series and its animation. Importing it pulls Matplotlib.
"""

# %% Imports
from .dyadic import plot_dyadic_graph
from .metrics import animate_curvature_entropy, plot_curvature_entropy
from .style import CURVATURE_CMAP, curvature_norm
from .topography import plot_node_topography

__all__ = [
    "CURVATURE_CMAP",
    "animate_curvature_entropy",
    "curvature_norm",
    "plot_curvature_entropy",
    "plot_dyadic_graph",
    "plot_node_topography",
]
