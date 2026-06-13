"""
NVIDIA CUDA backend via CuPy (GPU, float64).

CuPy is a drop-in NumPy-compatible array library backed by CUDA, so the 1d
Forman closed form (:mod:`hyphi.backends.base`) is the same code as the NumPy
backend with ``cp`` in place of ``np`` and ``cupyx.scatter_add`` for the
per-node accumulation. Unlike the Metal/MLX path, CuPy supports float64, so this
backend computes in float64 and is bit-for-bit comparable to the reference (the
parity tests assert the same ``< 1e-10`` tolerance as the NumPy backend).

This backend cannot be exercised on a machine without an NVIDIA GPU and the CUDA
toolkit; :meth:`is_available` returns ``False`` there and the dispatcher skips
it. The kernel is identical in form to the validated NumPy and MLX kernels, and
the parity harness in ``experiments/scripts/bench_curvature_backends.py`` runs it
automatically when a CUDA device is present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import CurvatureBackend

if TYPE_CHECKING:
    from .graph_io import GraphArrays

__all__ = ["CupyBackend"]


class CupyBackend(CurvatureBackend):
    """NVIDIA CUDA GPU backend via CuPy (float64). Implements the 1d notion."""

    name = "cupy"
    device_kind = "cuda"
    compute_dtype = "float64"

    @classmethod
    def is_available(cls) -> bool:
        """Report whether CuPy imports and at least one CUDA device is present."""
        try:
            import cupy as cp  # noqa: PLC0415  # ty:ignore[unresolved-import]

            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:  # noqa: BLE001
            return False

    def forman_curvature(self, graph: GraphArrays, method: str = "1d") -> np.ndarray:
        """
        Per-edge 1d Forman curvature on a CUDA GPU (float64).

        ``augmented`` is not implemented on this backend; the dispatcher routes
        it to the NumPy backend.
        """
        if method != "1d":
            raise NotImplementedError(
                f"CupyBackend implements method='1d' only; got {method!r}. "
                "The dispatcher falls back to the NumPy backend for 'augmented'."
            )
        import cupy as cp  # noqa: PLC0415  # ty:ignore[unresolved-import]
        from cupyx import scatter_add  # noqa: PLC0415  # ty:ignore[unresolved-import]

        if graph.n_edges == 0:
            return np.zeros(0, dtype=np.float64)

        ei = cp.asarray(graph.ei)
        ej = cp.asarray(graph.ej)
        we = cp.asarray(graph.we)

        inv_sqrt = 1.0 / cp.sqrt(we)
        s = cp.zeros(graph.n_nodes, dtype=cp.float64)
        scatter_add(s, ei, inv_sqrt)
        scatter_add(s, ej, inv_sqrt)
        curv = 4.0 - cp.sqrt(we) * (s[ei] + s[ej])
        return cp.asnumpy(curv).astype(np.float64)
