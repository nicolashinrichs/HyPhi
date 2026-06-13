"""
Public API for the visualization subpackage: network, curvature, and GDD plots.

This curates the existing plotting helpers. A coherent ``viz`` package (dyadic layouts, scalp
topography, animations) is planned separately; until then these functions are the supported
entry points. The plotting backends (plotly, seaborn) are optional and imported behind guards,
so importing this subpackage does not require them.
"""

# %% Imports
from .curvature_visualization import (
    curvature_distribution,
    visualize_graph_partitions_colors,
    visualize_graph_partitions_markers,
    visualize_graph_with_curvature,
)
from .GDD_FRc_visualization import (
    plot_gdd_heatmap,
    plot_successive_gdd,
    plot_weight_distributions_by_matrix,
)
from .network_plots import (
    plot_curvature_distribution,
    plot_curvature_network,
    plot_curvature_network_layouts,
    plot_network,
    plot_weight_distribution,
)

__all__ = [
    "curvature_distribution",
    "plot_curvature_distribution",
    "plot_curvature_network",
    "plot_curvature_network_layouts",
    "plot_gdd_heatmap",
    # Network and curvature-network plots
    "plot_network",
    "plot_successive_gdd",
    "plot_weight_distribution",
    # Graph dissimilarity (GDD) plots
    "plot_weight_distributions_by_matrix",
    "visualize_graph_partitions_colors",
    "visualize_graph_partitions_markers",
    # Curvature-on-graph visualizations
    "visualize_graph_with_curvature",
]
