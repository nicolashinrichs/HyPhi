"""
Array boundary between NetworkX graphs and the accelerator kernels.

The curvature backends do not operate on NetworkX objects; they operate on a
flat structure-of-arrays (SoA) representation of an undirected weighted graph,
which is what GPUs and vectorized CPU code want. This module is the single,
explicit conversion point.

A graph with ``N`` nodes and ``E`` edges is represented by

- ``n_nodes`` : int
- ``ei``, ``ej`` : int64 arrays of length ``E``, the endpoints of each edge
  (each undirected edge appears exactly once, ``ei < ej`` is not required)
- ``we`` : float64 array of length ``E``, the edge weights
- ``node_order`` : the node labels in index order, so curvature values can be
  mapped back onto the original NetworkX edges in ``G.edges()`` order

Node weights are assumed to be 1.0 (the ``GraphRicciCurvature`` default and the
universal convention for HyPhi PLV/CCORR graphs). Backends that receive a graph
with non-unit node weights must say so; see :func:`graph_to_arrays`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["GraphArrays", "graph_to_arrays"]


@dataclass(frozen=True)
class GraphArrays:
    """
    Flat structure-of-arrays view of one undirected weighted graph.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    ei, ej : numpy.ndarray
        ``int64`` endpoint indices for each edge, length ``E``.
    we : numpy.ndarray
        ``float64`` edge weights, length ``E``.
    node_order : list
        Node labels in index order (``index -> original label``), so a
        per-edge curvature array lines up with ``G.edges()`` iteration order.

    """

    n_nodes: int
    ei: np.ndarray
    ej: np.ndarray
    we: np.ndarray
    node_order: list

    @property
    def n_edges(self) -> int:
        """Number of edges ``E``."""
        return int(self.we.shape[0])


def graph_to_arrays(graph: nx.Graph, weight: str = "weight") -> GraphArrays:
    """
    Convert a NetworkX graph to its :class:`GraphArrays` SoA representation.

    Parameters
    ----------
    graph : networkx.Graph
        Undirected weighted graph. Missing edge weights default to 1.0.
    weight : str
        Edge attribute holding the weight (default ``"weight"``).

    Returns
    -------
    GraphArrays
        Flat arrays in ``graph.edges()`` order, ready for a kernel.

    Raises
    ------
    ValueError
        If the graph is a ``MultiGraph`` (collapse parallel edges first); if any
        node carries a non-unit ``weight`` attribute (the kernels assume unit node
        weights, matching the HyPhi convention, and a non-unit node weight would
        silently change the curvature relative to ``compute_frc``); or if any edge
        weight is not strictly positive. The 1d Forman kernel takes ``1 / sqrt(w)``,
        so a zero weight gives infinity and a negative weight (a signed CCORR
        value) gives a NaN. Apply ``abs`` and drop zero-coupling edges first,
        rather than letting non-finite curvature propagate silently into the
        entropy.

    Notes
    -----
    Edges are emitted in exactly ``graph.edges()`` order (skipping self-loops) so
    that a returned per-edge curvature array lines up with non-self-loop edge
    iteration. Self-loops are dropped as a deliberate policy: a self-loop is not a
    1-simplex in the Forman sense, and for PLV/CCORR graphs it is a diagonal
    artifact (self-correlation 1.0). This differs from
    ``GraphRicciCurvature.FormanRicci``, which iterates self-loop edges too, so the
    accelerated curvature can diverge from the legacy ``compute_frc`` on graphs
    that carry self-loops (the shipped connectome has one per node).

    A directed graph is converted to undirected first (matching the reference
    library), so antiparallel edges are merged rather than double-counted.

    """
    if graph.is_multigraph():
        raise ValueError(
            "graph_to_arrays: MultiGraph/MultiDiGraph is not supported; collapse parallel "
            "edges into a single weighted edge first."
        )
    if any(data.get(weight, 1.0) != 1.0 for _, data in graph.nodes(data=True)):
        raise ValueError(
            "graph_to_arrays: non-unit node weights are not supported. The kernels assume node "
            f"weight 1.0 (the GraphRicciCurvature default and the HyPhi convention); a node carries "
            f"a non-unit {weight!r} attribute, which would change the curvature versus compute_frc."
        )
    if graph.is_directed():
        graph = graph.to_undirected()

    node_order = list(graph.nodes())
    index = {label: i for i, label in enumerate(node_order)}

    ei_list: list[int] = []
    ej_list: list[int] = []
    we_list: list[float] = []
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        ei_list.append(index[u])
        ej_list.append(index[v])
        we_list.append(float(data.get(weight, 1.0)))

    we_arr = np.asarray(we_list, dtype=np.float64)
    # Require strictly positive AND finite. A naive `w <= 0` misses NaN and +inf (nan <= 0 and
    # inf <= 0 are both False), and `~(w > 0)` still lets +inf through (inf > 0 is True), so both
    # would poison curvature silently. A zero-variance PLV/CCORR window yields a NaN correlation,
    # which reaches here as a NaN edge weight.
    if we_arr.size and (~((we_arr > 0.0) & np.isfinite(we_arr))).any():
        raise ValueError(
            "graph_to_arrays: all edge weights must be strictly positive and finite (the 1d Forman "
            "kernel uses 1/sqrt(weight)). Found a non-positive or non-finite (NaN/inf) weight; apply "
            "abs and remove zero-coupling / undefined edges first (signed CCORR values must be made "
            "positive)."
        )

    ei = np.asarray(ei_list, dtype=np.int64)
    ej = np.asarray(ej_list, dtype=np.int64)
    return GraphArrays(n_nodes=len(node_order), ei=ei, ej=ej, we=we_arr, node_order=node_order)
