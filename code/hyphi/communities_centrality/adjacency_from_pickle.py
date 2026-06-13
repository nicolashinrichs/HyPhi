"""
Thin re-export of the adjacency-from-pickle loader.

The canonical implementation lives in :mod:`hyphi.io_brainhack` as
``load_pickle_adjacency``. This module re-exports it so the original
``from hyphi.communities_centrality.adjacency_from_pickle import load_pickle_adjacency``
import path (used in :mod:`hyphi.communities_centrality.processing`) keeps working
without carrying a duplicate copy.
"""

from hyphi.io_brainhack import load_pickle_adjacency

__all__ = ["load_pickle_adjacency"]
