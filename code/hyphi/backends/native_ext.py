"""
Native C++/CUDA/Metal backend via the optional ``hyphi_native`` extension.

This backend wraps the compiled ``hyphi_native`` module (built from the
``native/`` subproject with nanobind + scikit-build-core). It is the
hand-written-kernel path: a C++ CPU reference, a CUDA ``.cu`` kernel (float64),
and a Metal compute shader (float32), selected at call time by ``device``.

The extension is optional and is not built by a plain ``pip install hyphi``. When
it is absent, :meth:`is_available` returns ``False`` and the dispatcher never
selects this backend; the pure-Python array backends (NumPy, CuPy, MLX) cover
the same 1d kernel without a compiler. Build it with::

    pip install ./native                                       # CPU C++ only
    pip install ./native -C cmake.define.HYPHI_NATIVE_CUDA=ON  # + CUDA
    pip install ./native -C cmake.define.HYPHI_NATIVE_METAL=ON # + Metal

See ``native/README.md`` and ``native/NATIVE_BACKEND_DESIGN.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import CurvatureBackend

if TYPE_CHECKING:
    from .graph_io import GraphArrays

__all__ = ["NativeExtBackend"]


def _import_native():
    try:
        import hyphi_native  # type: ignore  # noqa: PLC0415
    except ImportError:
        return None
    return hyphi_native


class NativeExtBackend(CurvatureBackend):
    """
    Hand-written C++/CUDA/Metal kernel backend via ``hyphi_native``.

    Parameters
    ----------
    device : {"cpu", "cuda", "metal"}
        Which compiled device path to dispatch to. ``"cpu"`` is the C++
        reference (float64); ``"cuda"`` requires a CUDA build and an NVIDIA GPU
        (float64); ``"metal"`` requires a Metal build on Apple Silicon (float32).

    """

    name = "native"
    device_kind = "cpu"
    compute_dtype = "float64"

    def __init__(self, device: str = "cpu") -> None:
        """Bind this backend to a compiled device path (``cpu``, ``cuda`` or ``metal``)."""
        self.device = device
        self.device_kind = "cuda" if device == "cuda" else "metal" if device == "metal" else "cpu"
        self.compute_dtype = "float32" if device == "metal" else "float64"

    @classmethod
    def is_available(cls) -> bool:
        """Report whether the compiled ``hyphi_native`` extension is importable."""
        return _import_native() is not None

    def forman_curvature(self, graph: GraphArrays, method: str = "1d") -> np.ndarray:
        """Per-edge 1d Forman curvature via the compiled native kernel."""
        if method != "1d":
            raise NotImplementedError(
                f"NativeExtBackend implements method='1d' only; got {method!r}. "
                "The dispatcher falls back to the NumPy backend for 'augmented'."
            )
        native = _import_native()
        if native is None:
            raise RuntimeError("hyphi_native is not built. See native/README.md.")
        if graph.n_edges == 0:
            return np.zeros(0, dtype=np.float64)
        if graph.n_nodes > np.iinfo(np.int32).max:
            raise ValueError(
                f"NativeExtBackend: {graph.n_nodes} nodes exceeds the int32 index limit of the native kernels."
            )

        ei = graph.ei.astype(np.int32)
        ej = graph.ej.astype(np.int32)
        we = graph.we.astype(np.float64)
        try:
            out = native.forman_1d(graph.n_nodes, ei, ej, we, self.device)
        except RuntimeError:
            # The requested device was not compiled in or is absent; honor the
            # documented fallback to the C++ CPU path rather than propagating.
            if self.device == "cpu":
                raise
            out = native.forman_1d(graph.n_nodes, ei, ej, we, "cpu")
        return np.asarray(out, dtype=np.float64)
