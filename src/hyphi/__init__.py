# HyPhi — Geometric Hyperscanning Analysis Package
"""
Shared library for curvature-based hyperscanning analysis.
Consolidates duplicated logic from the software_module/ scripts.
"""

from .curvatures import (
    compute_frc,
    compute_frc_vec,
    compute_afrc,
    compute_afrc_vec,
    compute_orc,
    compute_orc_vec,
    extract_curvatures,
    extract_curvatures_vec,
    extract_curvature_matrices,
)

from .entropies import (
    entropy_vasicek,
    entropy_van_es,
    entropy_ebrahimi,
    entropy_correa,
    entropy_kde_plugin,
    entropy_kozachenko,
    entropy_renyi,
    entropy_tsallis,
    vec_entropy,
    get_quantiles,
    vec_quantiles,
)

from .windowing import (
    sliding_window_plv,
    build_graphs_from_matrices,
)

from .io_utils import load_config, make_dir, load_network_pkl
from .density import select_kde, fit_kde

__all__ = [
    # curvatures
    "compute_frc", "compute_frc_vec",
    "compute_afrc", "compute_afrc_vec",
    "compute_orc", "compute_orc_vec",
    "extract_curvatures", "extract_curvatures_vec",
    "extract_curvature_matrices",
    # entropies
    "entropy_vasicek", "entropy_van_es", "entropy_ebrahimi", "entropy_correa",
    "entropy_kde_plugin", "entropy_kozachenko", "entropy_renyi", "entropy_tsallis",
    "vec_entropy", "get_quantiles", "vec_quantiles",
    # windowing
    "sliding_window_plv", "build_graphs_from_matrices",
    # io
    "load_config", "make_dir", "load_network_pkl",
    # density
    "select_kde", "fit_kde",
]
