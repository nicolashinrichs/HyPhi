"""
Reference NetworkX backend: the parity oracle and the safe default.

This wraps the existing ``GraphRicciCurvature.FormanRicci`` path that the rest
of HyPhi already uses (via :func:`hyphi.modeling.graph_curvatures.compute_frc`).
It is the slowest backend, but it is the ground truth: every accelerated backend
is validated against the numbers this one produces, and the dispatcher falls
back to it for any ``(method, backend)`` combination an accelerator does not
implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from GraphRicciCurvature.FormanRicci import FormanRicci

from .base import CurvatureBackend

if TYPE_CHECKING:
    from .graph_io import GraphArrays

__all__ = ["NetworkxBackend"]


class NetworkxBackend(CurvatureBackend):
    """Ground-truth backend delegating to ``GraphRicciCurvature`` (float64)."""

    name = "networkx"
    device_kind = "cpu"
    compute_dtype = "float64"

    @classmethod
    def is_available(cls) -> bool:
        """Report availability; ``GraphRicciCurvature`` is a hard dependency, so always ``True``."""
        return True

    def forman_curvature(self, graph: GraphArrays, method: str = "1d") -> np.ndarray:
        """Per-edge curvature from the reference loop, in ``graph`` edge order."""
        g = nx.Graph()
        g.add_nodes_from(range(graph.n_nodes))
        for u, v, w in zip(graph.ei.tolist(), graph.ej.tolist(), graph.we.tolist(), strict=True):
            g.add_edge(u, v, weight=w)

        frc = FormanRicci(g, method=method)
        frc.compute_ricci_curvature()
        out = np.empty(graph.n_edges, dtype=np.float64)
        for k, (u, v) in enumerate(zip(graph.ei.tolist(), graph.ej.tolist(), strict=True)):
            out[k] = frc.G[u][v]["formanCurvature"]
        return out
