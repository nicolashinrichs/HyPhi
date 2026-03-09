# ==========================
# File I/O Utilities
# ==========================
"""
Moved from software_module/FileIO.py.
"""

import os
from os import path
import pickle


try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # fallback for 3.10


def load_config(config_file):
    """Load a TOML configuration file and return as dict."""
    with open(config_file, mode="rb") as fp:
        config = tomllib.load(fp)
    return config


def make_dir(dirpath):
    """Create a directory if it does not exist."""
    if not path.exists(dirpath):
        os.makedirs(dirpath)


def load_network_pkl(pkl_file):
    """Load a list of NetworkX graphs from a pickle file."""
    with open(pkl_file, mode="rb") as fp:
        networks = pickle.load(fp)
    return networks
