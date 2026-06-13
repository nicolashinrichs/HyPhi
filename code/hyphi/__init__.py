"""HyPhi: hyperscanning analysis via the geometry (Ricci curvature and entropy) of inter-brain networks."""

# %% Imports & setup
from importlib.metadata import version

# Submodules are imported so they are reachable as ``hyphi.<name>`` after
# ``import hyphi``. Each subpackage curates its own function-level public surface
# through its ``__all__``; ``analyses`` is the high-level pipeline facade.
# ``__all__`` below is the curated, user-facing set; the specialised
# communities_centrality / io_brainhack / spectral modules stay importable but
# are kept off the default star-import surface.
from . import (
    analyses,
    backends,
    benchmarks,
    communities_centrality,
    configs,
    fnirs,
    io,
    io_brainhack,
    modeling,
    null_models,
    preprocessing,
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
    "backends",
    "benchmarks",
    "configs",
    "fnirs",
    "io",
    "modeling",
    "null_models",
    "preprocessing",
    "simulation",
    "stats",
    "visualization",
]
