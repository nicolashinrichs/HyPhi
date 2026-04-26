#!/usr/bin/env python3
"""Minimal helper to load a graph pickle and return its adjacency matrix.

This is the communities_centrality-package copy of the loader.  The same
function is also available in :mod:`hyphi.io_brainhack` (preferred for new
code) and in :mod:`hyphi.spectral.adjacency_from_pickle`; the duplication is
intentional so that the original ``from
hyphi.communities_centrality.adjacency_from_pickle import load_pickle_adjacency``
import path used in :mod:`hyphi.communities_centrality.processing` keeps
working without modification.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np


def load_pickle_adjacency(
    pickle_path: Union[str, Path],
    weight_key: str = "weight",
    symmetrize: bool = False,
    return_nodes: bool = False,
):
    """
    Read a graph pickle and return its adjacency matrix as a NumPy array.

    Parameters
    ----------
    pickle_path
        Path to a pickle containing a NetworkX-like graph object.
        If the pickle contains a list/tuple, the first item is used.
    weight_key
        Edge-attribute key used as edge weight (default: "weight").
    symmetrize
        If True, return max(A, A.T).
    return_nodes
        If True, also return the node order used in the matrix.

    Returns
    -------
    np.ndarray
        Adjacency matrix.
    (np.ndarray, list)
        Returned when return_nodes=True.
    """

    path = Path(pickle_path)

    def _compat_load(path_obj: Path):
        class _CompatGraph:
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)

        class _CompatView:
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                self.__dict__["_state"] = state

        class _CompatUnpickler(pickle.Unpickler):
            _VIEW_NAMES = {
                "NodeView",
                "NodeDataView",
                "EdgeView",
                "OutEdgeView",
                "InEdgeView",
                "MultiEdgeView",
                "OutMultiEdgeView",
                "InMultiEdgeView",
                "EdgeDataView",
                "OutEdgeDataView",
                "InEdgeDataView",
                "MultiEdgeDataView",
                "OutMultiEdgeDataView",
                "InMultiEdgeDataView",
            }

            def find_class(self, module, name):
                if module == "networkx.classes.graph" and name == "Graph":
                    return _CompatGraph
                if module == "networkx.classes.reportviews" and name in self._VIEW_NAMES:
                    return _CompatView
                return super().find_class(module, name)

        with path_obj.open("rb") as f:
            raw = f.read()
        return _CompatUnpickler(io.BytesIO(raw)).load()

    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        obj = _compat_load(path)

    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError("Pickle contains an empty list/tuple.")
        obj = obj[0]

    if isinstance(obj, dict) and "_adj" in obj:
        adj = obj["_adj"]
    else:
        adj = getattr(obj, "_adj", None)

    if not isinstance(adj, dict):
        raise TypeError("Could not find dictionary-like '_adj' in the pickle object.")

    nodes: List[object] = list(adj.keys())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)

    for u, nbrs in adj.items():
        i = idx.get(u)
        if i is None or not isinstance(nbrs, dict):
            continue
        for v, attr in nbrs.items():
            j = idx.get(v)
            if j is None:
                continue
            if isinstance(attr, dict):
                A[i, j] = float(attr.get(weight_key, 1.0))
            else:
                A[i, j] = float(attr)

    if symmetrize:
        A = np.maximum(A, A.T)

    if return_nodes:
        return A, nodes
    return A
