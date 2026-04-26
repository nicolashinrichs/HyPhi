"""
Brainhack I/O helpers for loading adjacency matrices.

This module bundles loaders for the two graph-pickle datasets used by the
GDD / Forman-Ricci transformation notebooks (`GDD_FRc_Kuramoto.ipynb`,
`GDD_FRc_prebase.ipynb`):

- the per-window Kuramoto connectomes shipped in ``data/connectome/``
- the per-dyad prebase graphs (external — see ``data/README.md`` for download
  instructions; these pickles are not committed to the repo)

Each loader returns plain NumPy adjacency matrices keyed by integer index, so
downstream code (``hyphi.modeling.GDD_FRc_helpers``) can build NetworkX graphs
and compute curvatures without re-implementing the unpickling logic.

A small back-compat shim (`_compat_load`) lets us read graph pickles produced
under older NetworkX versions even when the importing environment ships a
different NetworkX, by stubbing out the legacy ``Graph`` / ``*View`` classes
referenced inside the pickle stream and reading the ``_adj`` dict directly.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np

# ------------------------------------------------------------------------------------------
# general
# ------------------------------------------------------------------------------------------


def load_pickle_adjacency(
    pickle_path: Union[str, Path],
    weight_key: str = "weight",
    symmetrize: bool = False,
    return_nodes: bool = False,
):
    """
    Minimal helper to load a graph pickle and return its adjacency matrix.
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


def load_adjacencies_from_paths(
    paths,
    weight_key="weight",
    symmetrize=False,
    return_nodes=False,
    verbose=False,
):
    """
    Load adjacency matrices for a dictionary of paths keyed by index.

    Parameters
    ----------
    paths : dict
        Dictionary like {1: Path(...), 2: Path(...), ...}.
    weight_key : str
        Edge attribute to use as adjacency weight.
    symmetrize : bool
        If True, return max(A, A.T) for each adjacency matrix.
    return_nodes : bool
        If True, also return the node order used for each adjacency matrix.
    verbose : bool
        If True, print basic loading information.

    Returns
    -------
    dict
        {idx: adjacency}
    (dict, dict)
        Returned when return_nodes=True:
        ({idx: adjacency}, {idx: nodes})
    """
    adjacencies = {}
    node_orders = {} if return_nodes else None

    for idx, path in paths.items():
        loaded = load_pickle_adjacency(
            path,
            weight_key=weight_key,
            symmetrize=symmetrize,
            return_nodes=return_nodes,
        )

        if return_nodes:
            adjacency, nodes = loaded
            adjacencies[idx] = adjacency
            node_orders[idx] = nodes
        else:
            adjacencies[idx] = loaded

        if verbose:
            print(f"[{idx}] loaded {path}")
            print("   adjacency shape:", adjacencies[idx].shape)

    if return_nodes:
        return adjacencies, node_orders
    return adjacencies


# ------------------------------------------------------------------------------------------
# kuramoto
# ------------------------------------------------------------------------------------------


def build_kuramoto_paths(
    indices=range(1, 9),
    base_dir=Path("HyPhi") / "data" / "connectome",
    suffix="connectome_kuramoto.pkl",
):
    """
    Build paths like:
    1_connectome_kuramoto.pkl, ..., 8_connectome_kuramoto.pkl
    and check that they exist.
    """
    paths = {idx: Path(base_dir) / f"{idx}_{suffix}" for idx in indices}

    missing = [idx for idx, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(str(idx) for idx in missing)
        raise FileNotFoundError(
            f"Missing Kuramoto pickle file(s) for index/indices: {missing_str}"
        )

    return paths


def load_all_kuramoto_adjacencies(
    indices=range(1, 9),
    base_dir=Path("HyPhi") / "data" / "connectome",
    suffix="connectome_kuramoto.pkl",
    weight_key="weight",
    symmetrize=False,
    return_nodes=False,
    verbose=False,
):
    """
    Load all Kuramoto adjacency matrices only.
    """
    paths = build_kuramoto_paths(
        indices=indices,
        base_dir=base_dir,
        suffix=suffix,
    )

    loaded = load_adjacencies_from_paths(
        paths,
        weight_key=weight_key,
        symmetrize=symmetrize,
        return_nodes=return_nodes,
        verbose=verbose,
    )

    if return_nodes:
        adjacencies, node_orders = loaded
        return {
            "paths": paths,
            "adjacencies": adjacencies,
            "nodes": node_orders,
        }

    return {
        "paths": paths,
        "adjacencies": loaded,
    }


# ------------------------------------------------------------------------------------------
# prebase
# ------------------------------------------------------------------------------------------


def build_prebase_paths(
    indices=range(1, 30),
    base_dir=Path("HyPhi") / "data" / "prebase",
    suffix="_prebase_graph.pkl",
):
    """
    Build paths like:
    01_prebase_graph.pkl, ..., 29_prebase_graph.pkl
    and check that they exist.
    """
    paths = {idx: Path(base_dir) / f"{idx:02d}{suffix}" for idx in indices}

    missing = [idx for idx, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(f"{idx:02d}" for idx in missing)
        raise FileNotFoundError(
            f"Missing prebase pickle file(s) for index/indices: {missing_str}"
        )

    return paths


def load_all_prebase_adjacencies(
    indices=range(1, 30),
    base_dir=Path("HyPhi") / "data" / "prebase",
    suffix="_prebase_graph.pkl",
    weight_key="weight",
    symmetrize=False,
    return_nodes=False,
    verbose=False,
):
    """
    Load all prebase adjacency matrices only.
    """
    paths = build_prebase_paths(
        indices=indices,
        base_dir=base_dir,
        suffix=suffix,
    )

    loaded = load_adjacencies_from_paths(
        paths,
        weight_key=weight_key,
        symmetrize=symmetrize,
        return_nodes=return_nodes,
        verbose=verbose,
    )

    if return_nodes:
        adjacencies, node_orders = loaded
        return {
            "paths": paths,
            "adjacencies": adjacencies,
            "nodes": node_orders,
        }

    return {
        "paths": paths,
        "adjacencies": loaded,
    }
