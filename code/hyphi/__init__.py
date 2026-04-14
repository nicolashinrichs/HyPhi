"""HyPhi: A Python package for hyperscanning data analysis via geometric entropy."""

# %% Imports & setup
from importlib.metadata import version

__author__ = """Hinrichs et al."""
__version__ = version(distribution_name="hyphi")

# Submodules imported here are accessible directly via ``import hyphi``
# (e.g., ``hyphi.simulation``).
# TODO: Add or remove imports below as your project evolves — only expose what a package user should use.
import hyphi.configs
import hyphi.visualization
import hyphi.modeling

# import hyphi.preprocessing  # TODO: nothing here yet (remove / add)
import hyphi.simulation
