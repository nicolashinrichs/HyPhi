"""
Hyperbolic embeddings for simulated hyperscanning data.

A signal-level, upstream complement to HyPhi's graph-level geometry: where the graph pipeline
measures Ollivier-Ricci curvature on connectivity matrices, this embeds the signal itself into a
negatively curved Poincare ball, the geometry that scalp and inter-brain EEG embeddings are
reported to occupy (HEEGNet, Li et al. 2026). It provides the Poincare-ball geometry utilities
and a lightweight hyperbolic embedding for simulated data, using only NumPy and scikit-learn (both
already core dependencies), so the subpackage is eager-imported and ``hyphi.embeddings`` is reachable.
"""

# %% Imports
from .embed import hyperbolic_embedding
from .hyperbolic import poincare_distance, poincare_distance_matrix, poincare_exp0

__all__ = [
    "hyperbolic_embedding",
    "poincare_distance",
    "poincare_distance_matrix",
    "poincare_exp0",
]
