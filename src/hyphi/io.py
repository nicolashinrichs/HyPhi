import tomllib
import os
from os import path
import pickle
import networkx as nx

def load_config(config_file: str) -> dict:
    """Load configuration from a TOML file."""
    with open(config_file, mode="rb") as fp:
        config = tomllib.load(fp)
    return config

def make_dir(dirpath: str) -> None:
    """Create directory if it does not exist."""
    if not path.exists(dirpath):
        os.makedirs(dirpath)

def load_network_pkl(pkl_file: str):
    """Load a list or array of networkx graphs from a pickle file."""
    with open(pkl_file, mode="rb") as fp:
        networks = pickle.load(fp)
    return networks

def save_network_pkl(data, pkl_file: str) -> None:
    """Save data to a pickle file."""
    make_dir(path.dirname(pkl_file) or ".")
    with open(pkl_file, mode="wb") as fp:
        pickle.dump(data, fp)
