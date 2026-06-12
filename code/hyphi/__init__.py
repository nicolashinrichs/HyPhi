"""HyPhi: A Python package for hyperscanning data analysis via geometric entropy."""

# %% Imports & setup
from importlib.metadata import version

__author__ = """Hinrichs et al."""
__version__ = version(distribution_name="hyphi")

# Submodules imported here are accessible directly via ``import hyphi``
# (e.g., ``hyphi.simulation``). Each subpackage curates its own public surface through its
# ``__all__``; ``analyses`` is the high-level pipeline facade.
import hyphi.analyses
import hyphi.benchmarks
import hyphi.communities_centrality
import hyphi.configs
import hyphi.io_brainhack
import hyphi.modeling
import hyphi.null_models
import hyphi.simulation
import hyphi.spectral
import hyphi.stats
import hyphi.visualization

__all__ = [
    "analyses",
    "benchmarks",
    "communities_centrality",
    "configs",
    "io_brainhack",
    "modeling",
    "null_models",
    "simulation",
    "spectral",
    "stats",
    "visualization",
]
