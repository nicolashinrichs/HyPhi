"""
Backend abstraction for accelerated discrete-curvature kernels.

The geometric core of HyPhi (Forman-Ricci curvature on a time series of
inter-brain graphs) is, for the 1-dimensional notion, a per-edge neighborhood
aggregation that reduces to a closed form expressible as a handful of array
operations. That closed form runs unchanged on NumPy (CPU), CuPy (NVIDIA CUDA),
and MLX (Apple Metal); the only thing that differs is the array module. This
module defines the common interface every such backend implements, plus the
exact mathematics so the kernels can be validated against the
``GraphRicciCurvature`` reference.

The math (1d Forman-Ricci, unit node weights)
---------------------------------------------
For an undirected weighted graph with edge weight ``w(e)`` on edge ``e = (u, v)``
and node weights set to 1, the combinatorial 1d Forman-Ricci curvature is

    F(u, v) = w(e) * ( 1/w(e) + 1/w(e)
                       - sum_{x ~ u, x != v} 1 / sqrt(w(e) * w(u, x))
                       - sum_{y ~ v, y != u} 1 / sqrt(w(e) * w(v, y)) )

Defining the per-node quantity ``S(u) = sum_{x ~ u} 1 / sqrt(w(u, x))`` this
telescopes to the vectorizable closed form

    F(u, v) = 4 - sqrt(w(e)) * ( S(u) + S(v) )                         (1d)

which is what every backend computes: one reciprocal-sqrt per edge, one
scatter-add to accumulate ``S`` per node, one gather, one fused multiply-add.
The algebraic identity is verified bit-for-bit against the reference loop in the
parity tests (``code/tests/test_backends.py``) and reproduced in
``experiments/scripts/bench_curvature_backends.py``.

The ``augmented`` notion (AFRC) additionally needs the per-edge triangle count
and a face-restricted weight sum; it is provided on the NumPy backend via sparse
operations and otherwise falls back to the reference, which the dispatcher
handles transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .graph_io import GraphArrays

__all__ = ["CurvatureBackend"]


class CurvatureBackend(ABC):
    """
    Abstract accelerator backend for discrete-curvature kernels.

    A backend is identified by :attr:`name` and a :attr:`device_kind`
    (``"cpu"``, ``"cuda"`` or ``"metal"``). It reports availability on the
    current machine via :meth:`is_available` (a cheap, never-raising probe) and
    computes per-edge curvature via :meth:`forman_curvature`.

    Backends never become the default until they pass the parity suite; the CPU
    NumPy backend is the reference-equivalent default, and the NetworkX backend
    is the ground-truth oracle every other backend is gated against.
    """

    #: Short stable identifier used in the registry and ``HYPHI_BACKEND``.
    name: str = "abstract"
    #: One of ``"cpu"``, ``"cuda"``, ``"metal"``.
    device_kind: str = "cpu"
    #: Floating dtype the kernel computes in (documents the parity regime).
    compute_dtype: str = "float64"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Return ``True`` if this backend can run on the current machine.

        Must never raise: a missing optional dependency or absent device is a
        ``False``, not an exception, so the dispatcher can probe every backend.
        """

    @abstractmethod
    def forman_curvature(self, graph: GraphArrays, method: str = "1d") -> np.ndarray:
        """
        Compute per-edge Forman-Ricci curvature.

        Parameters
        ----------
        graph : GraphArrays
            SoA representation from :func:`hyphi.backends.graph_io.graph_to_arrays`.
        method : {"1d", "augmented"}
            Curvature notion. ``"1d"`` is the closed form above; ``"augmented"``
            adds triangle/face corrections.

        Returns
        -------
        numpy.ndarray
            ``float64`` per-edge curvature in ``graph`` edge order (a host array,
            so callers never handle device memory). GPU backends copy back.

        """

    def __repr__(self) -> str:
        """Return a concise representation naming the backend, device and dtype."""
        return f"<{type(self).__name__} name={self.name!r} device={self.device_kind!r} dtype={self.compute_dtype!r}>"
