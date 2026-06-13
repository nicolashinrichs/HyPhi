"""
Vectorized NumPy backend (CPU, float64): the reference-equivalent fast path.

This is the closed-form 1d Forman kernel from :mod:`hyphi.backends.base` plus a
sparse-matrix ``augmented`` kernel, all in pure NumPy/SciPy. It returns the same
numbers as the ``GraphRicciCurvature`` reference loop to floating-point
tolerance (parity tests assert ``< 1e-10``), while replacing the per-edge
Python adjacency-dict traversal with a handful of array operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from .base import CurvatureBackend

if TYPE_CHECKING:
    from .graph_io import GraphArrays

__all__ = ["NumpyBackend"]


def _node_inv_sqrt_sum(graph: GraphArrays) -> np.ndarray:
    """Per-node ``S(u) = sum_{x ~ u} 1 / sqrt(w(u, x))`` via scatter-add."""
    inv_sqrt = 1.0 / np.sqrt(graph.we)
    s = np.zeros(graph.n_nodes, dtype=np.float64)
    np.add.at(s, graph.ei, inv_sqrt)
    np.add.at(s, graph.ej, inv_sqrt)
    return s


def _forman_1d(graph: GraphArrays) -> np.ndarray:
    """Closed-form 1d Forman: ``F = 4 - sqrt(w_e) * (S[u] + S[v])``."""
    s = _node_inv_sqrt_sum(graph)
    return 4.0 - np.sqrt(graph.we) * (s[graph.ei] + s[graph.ej])


def _forman_augmented(graph: GraphArrays) -> np.ndarray:
    """
    Augmented Forman via sparse triangle and face-restricted weight sums.

    ``F_aug = |face| * w_e^2 + 2 - sqrt(w_e) * |T_u + T_v|`` where ``|face|`` is
    the per-edge triangle count and ``T_u`` is the inverse-sqrt weight sum over
    ``u``'s non-triangle, non-``v`` neighbors. Both are obtained from sparse
    matrix products: ``A @ A`` for triangle counts and ``M @ A`` for the
    face-restricted weight sums, with ``M[u, x] = 1 / sqrt(w(u, x))``.
    """
    n = graph.n_nodes
    ei, ej, we = graph.ei, graph.ej, graph.we
    inv_sqrt = 1.0 / np.sqrt(we)

    a = sparse.csr_matrix((np.ones_like(we), (ei, ej)), shape=(n, n))
    a = a + a.T  # binary adjacency (symmetric)
    m = sparse.csr_matrix((inv_sqrt, (ei, ej)), shape=(n, n))
    m = m + m.T  # inverse-sqrt-weight adjacency (symmetric)

    tri = (a @ a).tocsr()  # common-neighbor counts
    p = (m @ a).tocsr()  # P[u, v] = sum_x M[u, x] A[v, x] over common neighbors

    face = np.asarray(tri[ei, ej]).ravel()
    fw_u = np.asarray(p[ei, ej]).ravel()
    fw_v = np.asarray(p[ej, ei]).ravel()

    s = _node_inv_sqrt_sum(graph)
    t_u = (s[ei] - inv_sqrt) - fw_u
    t_v = (s[ej] - inv_sqrt) - fw_v
    return face * we * we + 2.0 - np.sqrt(we) * np.abs(t_u + t_v)


class NumpyBackend(CurvatureBackend):
    """CPU backend computing curvature with vectorized NumPy/SciPy (float64)."""

    name = "numpy"
    device_kind = "cpu"
    compute_dtype = "float64"

    @classmethod
    def is_available(cls) -> bool:
        """Report availability; NumPy is a hard dependency, so always ``True``."""
        return True

    def forman_curvature(self, graph: GraphArrays, method: str = "1d") -> np.ndarray:
        """Per-edge Forman-Ricci curvature; see :class:`CurvatureBackend`."""
        if graph.n_edges == 0:
            return np.zeros(0, dtype=np.float64)
        if method == "1d":
            return _forman_1d(graph)
        if method == "augmented":
            return _forman_augmented(graph)
        raise ValueError(f"Unknown method {method!r}; expected '1d' or 'augmented'.")
