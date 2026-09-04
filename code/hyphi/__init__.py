"""HyPhi: hyperscanning analysis via the geometry (Ricci curvature and entropy) of inter-brain networks."""

# %% Imports & setup
from importlib.metadata import version

# Submodules are imported so they are reachable as ``hyphi.<name>`` after
# ``import hyphi``. Each submodule declares its own ``__all__`` for its
# function-level public surface. ``__all__`` below is the curated, user-facing
# set; the specialised communities_centrality / io_brainhack / spectral modules
# stay importable but are kept off the default star-import surface.
from . import (
    analyses,
    benchmarks,
    communities_centrality,
    configs,
    io,
    io_brainhack,
    modeling,
    null_models,
    simulation,
    spectral,
    stats,
    visualization,
)

__author__ = "Hinrichs et al."
__version__ = version(distribution_name="hyphi")

__all__ = [
    "__version__",
    "analyses",
    "benchmarks",
    "configs",
    "io",
    "modeling",
    "null_models",
    "simulation",
    "stats",
    "visualization",
]
