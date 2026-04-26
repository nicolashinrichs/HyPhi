"""
I/O module for HyPhi: Loading configurations, reading and writing network data.

Years: 2026
"""

# %% Import
import os
import pickle
import tomllib
from pathlib import Path

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_config(config_file: str | Path) -> dict:
    """Load configuration from a TOML file."""
    with open(config_file, mode="rb") as fp:
        config = tomllib.load(fp)
    return config


def make_dir(dirpath: str) -> None:
    """Create a directory if it does not exist."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_network_pkl(pkl_file: str):
    """Load a list or array of networkx graphs from a pickle file."""
    with open(pkl_file, mode="rb") as fp:
        networks = pickle.load(fp)
    return networks


def save_network_pkl(data, pkl_file: str) -> None:
    """Save data to a pickle file."""
    make_dir(os.path.dirname(pkl_file) or ".")
    with open(pkl_file, mode="wb") as fp:
        pickle.dump(data, fp)


class CompatUnpickler(pickle.Unpickler):
    """Compatibility unpickler for older NumPy-internal module paths.

    Pickles produced under ``numpy >= 1.25`` reference the private
    ``numpy._core`` namespace.  When loaded under older NumPy releases that
    only expose ``numpy.core``, the standard unpickler raises
    ``ModuleNotFoundError``.  This unpickler rewrites the prefix on the fly so
    such pickles can still be loaded.
    """

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_connectivity_data(pickle_path: str | Path):
    """Load ``connectivity_data.pkl`` and return all expected components.

    Uses :class:`CompatUnpickler` so that pickles written under newer NumPy
    versions remain loadable in environments still on the legacy
    ``numpy.core`` namespace.
    """
    with open(pickle_path, "rb") as fp:
        data = CompatUnpickler(fp).load()
    return data


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
