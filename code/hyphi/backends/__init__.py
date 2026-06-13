"""
hyphi.backends: pluggable accelerator backends for discrete-curvature kernels.

One vectorized kernel, several devices. The geometric core of HyPhi (Forman-Ricci
curvature on a time series of inter-brain graphs) reduces, for the 1d notion, to
a closed form that runs unchanged on NumPy (CPU, float64), CuPy (NVIDIA CUDA,
float64) and MLX (Apple Metal, float32). This package exposes a single dispatch
point over those backends with capability-based auto-selection, graceful
fallback to the reference implementation, and per-machine install hints.

Quick start
-----------
>>> from hyphi import backends
>>> import networkx as nx
>>> G = nx.watts_strogatz_graph(200, 6, 0.2, seed=0)
>>> curv = backends.forman_curvature(G, method="1d", backend="auto")  # fastest device here
>>> backends.available_backends()
['numpy', 'networkx', ...]
>>> print(backends.install_hint())

Backend selection precedence for ``backend=None``:
1. the ``HYPHI_BACKEND`` environment variable, if set;
2. otherwise ``"numpy"`` (the safe float64 default).
Pass ``backend="auto"`` to opt into the fastest accelerator available here.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import networkx as nx

from .base import CurvatureBackend
from .capabilities import Capabilities, detect, install_hint, recommend_backend
from .cpu_numpy import NumpyBackend
from .cuda_cupy import CupyBackend
from .graph_io import GraphArrays, graph_to_arrays
from .metal_mlx import MlxBackend
from .native_ext import NativeExtBackend
from .reference_networkx import NetworkxBackend

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "Capabilities",
    "CupyBackend",
    "CurvatureBackend",
    "GraphArrays",
    "MlxBackend",
    "NativeExtBackend",
    "NetworkxBackend",
    "NumpyBackend",
    "available_backends",
    "detect",
    "forman_curvature",
    "get_backend",
    "graph_to_arrays",
    "install_hint",
    "recommend_backend",
]

_REGISTRY: dict[str, type[CurvatureBackend]] = {
    NumpyBackend.name: NumpyBackend,
    NetworkxBackend.name: NetworkxBackend,
    CupyBackend.name: CupyBackend,
    MlxBackend.name: MlxBackend,
    NativeExtBackend.name: NativeExtBackend,
}

#: Safe default: exact float64, always available.
DEFAULT_BACKEND = "numpy"


def available_backends() -> list[str]:
    """Return the names of backends usable on the current machine."""
    return [name for name, cls in _REGISTRY.items() if cls.is_available()]


def get_backend(backend: str | CurvatureBackend | None = None) -> CurvatureBackend:
    """
    Resolve a backend selector to a usable :class:`CurvatureBackend`.

    Parameters
    ----------
    backend : str, CurvatureBackend or None
        ``None`` uses ``HYPHI_BACKEND`` if set, else :data:`DEFAULT_BACKEND`.
        ``"auto"`` picks the fastest accelerator available here
        (:func:`recommend_backend`). A name selects that backend by registry
        key. A :class:`CurvatureBackend` instance is returned unchanged.

    Returns
    -------
    CurvatureBackend
        An instantiated, available backend.

    Raises
    ------
    ValueError
        If the name is unknown.
    RuntimeError
        If the named backend exists but is not available on this machine.

    """
    if isinstance(backend, CurvatureBackend):
        return backend
    if backend is None:
        backend = os.environ.get("HYPHI_BACKEND", DEFAULT_BACKEND)
    if backend == "auto":
        backend = recommend_backend()
    try:
        cls = _REGISTRY[backend]
    except KeyError:
        raise ValueError(f"Unknown backend {backend!r}. Available names: {sorted(_REGISTRY)}") from None
    if not cls.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available on this machine. "
            f"{install_hint()} Usable backends: {available_backends()}"
        )
    return cls()


def _forman_one(
    graph: nx.Graph,
    method: str,
    backend: CurvatureBackend,
    fallback: CurvatureBackend,
    weight: str,
    warn_fallback: bool,
) -> tuple[GraphArrays, np.ndarray]:
    arrays = graph_to_arrays(graph, weight=weight)
    try:
        return arrays, backend.forman_curvature(arrays, method=method)
    except NotImplementedError:
        if warn_fallback:
            warnings.warn(
                f"Backend {backend.name!r} does not implement method={method!r}; falling back to {fallback.name!r}.",
                stacklevel=3,
            )
        return arrays, fallback.forman_curvature(arrays, method=method)


def forman_curvature(
    graph: nx.Graph | list[nx.Graph],
    method: str = "1d",
    backend: str | CurvatureBackend | None = None,
    weight: str = "weight",
    annotate: bool = False,
    warn_fallback: bool = False,
):
    """
    Compute Forman-Ricci curvature for a graph or a series of graphs.

    Parameters
    ----------
    graph : networkx.Graph or list of networkx.Graph
        One graph or a time series of graphs (e.g. sliding windows).
    method : {"1d", "augmented"}
        Curvature notion.
    backend : str, CurvatureBackend or None
        Backend selector; see :func:`get_backend`. ``"auto"`` uses the fastest
        accelerator available here.
    weight : str
        Edge weight attribute (default ``"weight"``).
    annotate : bool
        If ``True``, return copies of the input graphs with a
        ``"formanCurvature"`` edge attribute (mirroring
        :func:`hyphi.modeling.graph_curvatures.compute_frc`); if ``False``,
        return per-edge curvature arrays.
    warn_fallback : bool
        Warn when a backend cannot do ``method`` and the CPU fallback is used.

    Returns
    -------
    numpy.ndarray or list
        For a single graph: a per-edge array (or an annotated graph if
        ``annotate``). For a list: a list of those.

    Notes
    -----
    A backend that does not implement a requested ``method`` (the GPU backends
    implement ``"1d"`` only) transparently falls back to the NumPy backend, so
    callers never have to special-case device support.

    Self-loops are excluded from the curvature computation: a self-loop is not a
    1-simplex in the Forman sense and, for PLV/CCORR graphs, is a diagonal
    artifact (self-correlation 1.0). With ``annotate=True`` the returned copy has
    its self-loops removed and a ``"formanCurvature"`` attribute on every
    remaining edge, with values mapped to edges through the SoA index rather than
    relying on edge-iteration order. This differs from
    :func:`hyphi.modeling.graph_curvatures.compute_frc` on graphs that contain
    self-loops, which annotates the self-loop edges too; see
    ``docs/acceleration.md`` for the measured divergence on the shipped connectome.

    """
    resolved = get_backend(backend)
    fallback = get_backend(DEFAULT_BACKEND)
    # A single graph is an nx.Graph; anything else (list, tuple, generator) is a
    # series, which also materializes a generator exactly once.
    if isinstance(graph, nx.Graph):
        graphs, is_series = [graph], False
    else:
        graphs, is_series = list(graph), True

    results = []
    for g in graphs:
        arrays, curv = _forman_one(g, method, resolved, fallback, weight, warn_fallback)
        if annotate:
            gc = g.copy()
            gc.remove_edges_from([(u, v) for u, v in gc.edges() if u == v])
            labels = arrays.node_order
            ei, ej, cl = arrays.ei.tolist(), arrays.ej.tolist(), curv.tolist()
            for k in range(arrays.n_edges):
                gc[labels[ei[k]]][labels[ej[k]]]["formanCurvature"] = cl[k]
            results.append(gc)
        else:
            results.append(curv)

    return results if is_series else results[0]
