"""
Public API for the modeling subpackage: curvature, entropy, windowing, and density.

This surfaces the canonical pipeline pieces so they can be imported directly, for example
``from hyphi.modeling import compute_frc, vec_entropy``. The lower-level embedding and
Ricci-flow helpers in ``curvatures.py`` are intentionally left out of this curated surface
while the canonical flow API is still being decided; import them by their full path if needed.
"""

# %% Imports
from .density_estimation import fit_kde, select_kde
from .entropies import (
    entropy_correa,
    entropy_ebrahimi,
    entropy_kde_plugin,
    entropy_kozachenko,
    entropy_renyi,
    entropy_tsallis,
    entropy_van_es,
    entropy_vasicek,
    get_quantiles,
    vec_entropy,
    vec_quantiles,
)
from .graph_curvatures import (
    compute_afrc,
    compute_afrc_vec,
    compute_frc,
    compute_frc_vec,
    compute_orc,
    compute_orc_vec,
    extract_curvature_matrices,
    extract_curvatures,
    extract_curvatures_vec,
)
from .transform_curvature import (
    apply_linear_transform,
    attach_edge_weights_to_graph,
    check_all_positive,
    fit_global_positive_linear_transform,
    transform_curvature_collection,
)
from .windowing import build_graphs_from_matrices, compute_plv_matrix, sliding_window_plv

__all__ = [
    "apply_linear_transform",
    "attach_edge_weights_to_graph",
    "build_graphs_from_matrices",
    "check_all_positive",
    "compute_afrc",
    "compute_afrc_vec",
    # Curvature
    "compute_frc",
    "compute_frc_vec",
    "compute_orc",
    "compute_orc_vec",
    # Windowing and connectivity
    "compute_plv_matrix",
    "entropy_correa",
    "entropy_ebrahimi",
    "entropy_kde_plugin",
    "entropy_kozachenko",
    "entropy_renyi",
    "entropy_tsallis",
    "entropy_van_es",
    # Entropy and quantiles
    "entropy_vasicek",
    "extract_curvature_matrices",
    "extract_curvatures",
    "extract_curvatures_vec",
    # Curvature transforms
    "fit_global_positive_linear_transform",
    # Density estimation
    "fit_kde",
    "get_quantiles",
    "select_kde",
    "sliding_window_plv",
    "transform_curvature_collection",
    "vec_entropy",
    "vec_quantiles",
]
