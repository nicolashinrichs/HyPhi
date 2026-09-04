"""I/O utilities for HyPhi: load configuration files and (NetworkX) graph pickles.

Years: 2026
"""

# %% Import
import pickle
import tomllib
from pathlib import Path

import numpy as np

__all__ = [
    "CompatUnpickler",
    "load_config",
    "load_network_pkl",
    "load_pickle_adjacency",
]

# NetworkX reportview class names tolerated by the compatibility unpickler when the
# graph was pickled under a different NetworkX version than the one installed.
_NETWORKX_VIEW_NAMES = frozenset(
    {
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
)

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_config(config_file: str | Path) -> dict:
    """Load a configuration from a TOML file.

    Parameters
    ----------
    config_file : str or Path
        Path to the TOML configuration file.

    Returns
    -------
    dict
        The parsed configuration.
    """
    with Path(config_file).open("rb") as fp:
        return tomllib.load(fp)


def load_network_pkl(pkl_file: str | Path):
    """Load a list or array of NetworkX graphs from a pickle file.

    Parameters
    ----------
    pkl_file : str or Path
        Path to the pickle file.

    Returns
    -------
    Any
        The unpickled object (typically a list of ``networkx.Graph``).
    """
    with Path(pkl_file).open("rb") as fp:
        return pickle.load(fp)


class CompatUnpickler(pickle.Unpickler):
    """Unpickler that loads newer-NumPy pickles under older NumPy releases.

    Pickles produced under ``numpy >= 1.25`` reference the private ``numpy._core``
    namespace. When loaded under older NumPy releases that only expose
    ``numpy.core``, the standard unpickler raises ``ModuleNotFoundError``. This
    unpickler rewrites the prefix on the fly so such pickles remain loadable.
    """

    def find_class(self, module, name):
        """Rewrite the ``numpy._core`` module prefix to ``numpy.core`` on lookup."""
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


class _CompatGraph:
    """Surrogate for ``networkx.Graph`` used when the original class is unavailable."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        """Restore the surrogate's attributes from the pickled ``state`` dict."""
        if isinstance(state, dict):
            self.__dict__.update(state)


class _CompatView:
    """Surrogate for a NetworkX reportview used when the original class is unavailable."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        """Store the pickled view ``state`` without reconstructing the original view."""
        self.__dict__["_state"] = state


class _NetworkxCompatUnpickler(pickle.Unpickler):
    """Unpickler that substitutes surrogates for NetworkX graph/view classes.

    Used as a fallback when a graph pickle references NetworkX internals that are
    not importable in the current environment; the surrogates expose enough state
    (the ``_adj`` adjacency dict) to recover the adjacency matrix.
    """

    def find_class(self, module, name):
        """Return a surrogate for known NetworkX graph/view classes, else default."""
        if module == "networkx.classes.graph" and name == "Graph":
            return _CompatGraph
        if module == "networkx.classes.reportviews" and name in _NETWORKX_VIEW_NAMES:
            return _CompatView
        return super().find_class(module, name)


def _load_graph_object(path: Path):
    """Load a (possibly cross-version) graph pickle, falling back to surrogates."""
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001 (any unpickling failure -> retry with the compat unpickler)
        with path.open("rb") as f:
            return _NetworkxCompatUnpickler(f).load()


def _adjacency_dict(obj) -> dict:
    """Extract the ``_adj`` adjacency dictionary from a loaded graph-like object."""
    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError("Pickle contains an empty list/tuple.")
        obj = obj[0]
    adj = obj["_adj"] if isinstance(obj, dict) and "_adj" in obj else getattr(obj, "_adj", None)
    if not isinstance(adj, dict):
        raise TypeError("Could not find a dictionary-like '_adj' in the pickle object.")
    return adj


def load_pickle_adjacency(
    pickle_path: str | Path,
    weight_key: str = "weight",
    symmetrize: bool = False,
    return_nodes: bool = False,
):
    """Read a graph pickle and return its adjacency matrix as a NumPy array.

    Loads NetworkX-like graph pickles even across NetworkX versions (falling back
    to lightweight surrogates that expose only the adjacency dict). This is the
    canonical loader; ``hyphi.spectral.adjacency_from_pickle`` and
    ``hyphi.communities_centrality.adjacency_from_pickle`` re-export it for their
    legacy import paths.

    Parameters
    ----------
    pickle_path : str or Path
        Path to a pickle containing a NetworkX-like graph object. If the pickle
        contains a list or tuple, the first item is used.
    weight_key : str, default="weight"
        Edge-attribute key used as the edge weight.
    symmetrize : bool, default=False
        If True, return ``max(A, A.T)``.
    return_nodes : bool, default=False
        If True, also return the node order used in the matrix.

    Returns
    -------
    np.ndarray or tuple of (np.ndarray, list)
        The adjacency matrix, and the node order when ``return_nodes`` is True.
    """
    adj = _adjacency_dict(_load_graph_object(Path(pickle_path)))
    nodes: list[object] = list(adj.keys())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=float)

    for u, nbrs in adj.items():
        i = idx.get(u)
        if i is None or not isinstance(nbrs, dict):
            continue
        for v, attr in nbrs.items():
            j = idx.get(v)
            if j is None:
                continue
            adjacency[i, j] = float(attr.get(weight_key, 1.0)) if isinstance(attr, dict) else float(attr)

    if symmetrize:
        adjacency = np.maximum(adjacency, adjacency.T)
    if return_nodes:
        return adjacency, nodes
    return adjacency


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
