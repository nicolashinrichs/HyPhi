"""
Apple Metal backend via MLX (Apple Silicon GPU, float32).

MLX executes array operations on the Apple Metal GPU. The 1d Forman closed form
(:mod:`hyphi.backends.base`) maps directly onto MLX: a reciprocal-sqrt per edge,
two scatter-adds to accumulate the per-node sum, a gather, and a fused
multiply-add.

Precision regime (important for parity)
---------------------------------------
The Metal GPU does not support float64 (``mx.array(..., dtype=mx.float64)`` on
the GPU raises). This backend therefore computes in **float32** and copies the
result back to a float64 host array. Parity against the float64 reference is
exact only to float32 tolerance: expect a relative error around 1e-6 and an
absolute error that scales with curvature magnitude (a few 1e-3 on edges whose
curvature is order 1e2). This is documented, measured in
``experiments/scripts/bench_curvature_backends.py``, and asserted with a
float32-appropriate tolerance in the parity tests. Use this backend when
throughput on large graph series matters more than float64 reproducibility; use
the ``numpy`` backend when bit-level float64 parity is required.

Determinism
-----------
The per-node accumulation is a parallel scatter-add, and float32 addition is not
associative, so repeated runs on the same input can differ in the last bits
(measured around 1e-6 relative, the float32 floor). This backend is therefore not
bit-reproducible. For results that go into a paper's supplementary materials, use
a float64 backend (``numpy`` or ``cupy``), which is deterministic; reserve this
backend for throughput where a float32-level difference is acceptable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import CurvatureBackend

if TYPE_CHECKING:
    from .graph_io import GraphArrays

__all__ = ["MlxBackend"]


class MlxBackend(CurvatureBackend):
    """Apple Metal GPU backend via MLX (float32). Implements the 1d notion."""

    name = "mlx"
    device_kind = "metal"
    compute_dtype = "float32"

    @classmethod
    def is_available(cls) -> bool:
        """Report whether MLX and a usable Apple Metal GPU are present."""
        try:
            import mlx.core as mx  # noqa: PLC0415

            probe = mx.sqrt(mx.array([4.0], dtype=mx.float32))
            mx.eval(probe)
        except Exception:  # noqa: BLE001
            return False
        return True

    def forman_curvature(self, graph: GraphArrays, method: str = "1d") -> np.ndarray:
        """
        Per-edge 1d Forman curvature on the Metal GPU (float32).

        ``augmented`` is not implemented on this backend; the dispatcher routes
        it to a float64 CPU backend. Raising here keeps the contract explicit.
        """
        if method != "1d":
            raise NotImplementedError(
                f"MlxBackend implements method='1d' only; got {method!r}. "
                "The dispatcher falls back to a CPU backend for 'augmented'."
            )
        import mlx.core as mx  # noqa: PLC0415

        if graph.n_edges == 0:
            return np.zeros(0, dtype=np.float64)
        if graph.n_nodes > np.iinfo(np.int32).max:
            raise ValueError(f"MlxBackend: {graph.n_nodes} nodes exceeds the int32 index limit.")

        ei = mx.array(graph.ei.astype(np.int32))
        ej = mx.array(graph.ej.astype(np.int32))
        we = mx.array(graph.we.astype(np.float32))

        inv_sqrt = 1.0 / mx.sqrt(we)
        s = mx.zeros(graph.n_nodes, dtype=mx.float32)
        s = s.at[ei].add(inv_sqrt)
        s = s.at[ej].add(inv_sqrt)
        curv = 4.0 - mx.sqrt(we) * (s[ei] + s[ej])
        mx.eval(curv)
        return np.array(curv, copy=True).astype(np.float64)
