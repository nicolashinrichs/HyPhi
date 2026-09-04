"""Backward-compatible import path for :func:`hyphi.io.load_pickle_adjacency`.

The adjacency-from-pickle loader now lives canonically in :mod:`hyphi.io`. This
thin shim keeps the original ``from hyphi.spectral.adjacency_from_pickle import
load_pickle_adjacency`` import path (used in the contributor notebooks) working.
"""

from hyphi.io import load_pickle_adjacency

__all__ = ["load_pickle_adjacency"]
