"""
I/O module for HyPhi: Loading configurations, reading, and writing network data.

Years: 2026
"""

# %% Import
import pickle
import tomllib
from pathlib import Path

__all__ = ["load_config", "load_network_pkl"]

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_config(config_file: str | Path) -> dict:
    """Load configuration from a TOML file."""
    with Path(config_file).open("rb") as fp:
        return tomllib.load(fp)  # configs


def load_network_pkl(pkl_file: str | Path):
    """Load a list or array of networkx graphs from a pickle file."""
    with Path(pkl_file).open("rb") as fp:
        return pickle.load(fp)  # networks


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
