"""
Machine capability detection and per-platform backend selection.

This is the piece that makes acceleration "just work" on any machine: it probes
which array accelerators are usable here (CUDA via CuPy, Metal via MLX), reports
the CPU budget, recommends the best available backend, and prints the exact
``pip`` command to enable acceleration for this specific machine. The probes
never raise; an absent device or missing optional dependency is reported, not
thrown.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

__all__ = ["Capabilities", "detect", "install_hint", "recommend_backend"]


@dataclass(frozen=True)
class Capabilities:
    """
    Snapshot of the accelerators available on the current machine.

    Attributes
    ----------
    system, machine : str
        ``platform.system()`` and ``platform.machine()``.
    cpu_count : int
        Logical core count.
    cuda : bool
        A CUDA device plus CuPy is importable and reports >= 1 device.
    metal : bool
        Apple Silicon with MLX and a usable Metal GPU.
    apple_silicon : bool
        macOS on an ``arm64`` machine (Metal-capable hardware).

    """

    system: str
    machine: str
    cpu_count: int
    cuda: bool
    metal: bool
    apple_silicon: bool


def _cuda_available() -> bool:
    try:
        import cupy as cp  # noqa: PLC0415  # ty:ignore[unresolved-import]

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _metal_available() -> bool:
    try:
        import mlx.core as mx  # noqa: PLC0415

        mx.eval(mx.sqrt(mx.array([4.0], dtype=mx.float32)))
    except Exception:  # noqa: BLE001
        return False
    return True


def detect() -> Capabilities:
    """Probe and return the current machine's accelerator capabilities."""
    system = platform.system()
    machine = platform.machine()
    return Capabilities(
        system=system,
        machine=machine,
        cpu_count=os.cpu_count() or 1,
        cuda=_cuda_available(),
        metal=_metal_available(),
        apple_silicon=(system == "Darwin" and machine == "arm64"),
    )


def recommend_backend(caps: Capabilities | None = None) -> str:
    """
    Return the name of the fastest backend available on this machine.

    Preference order is ``cuda`` (NVIDIA, float64) > ``mlx`` (Apple Metal,
    float32) > ``numpy`` (CPU, float64). The CPU backend is always a valid
    answer, so this never fails.
    """
    caps = caps or detect()
    if caps.cuda:
        return "cupy"
    if caps.metal:
        return "mlx"
    return "numpy"


def install_hint(caps: Capabilities | None = None) -> str:
    """
    Return a one-line, machine-specific instruction to enable acceleration.

    Tells the user exactly which optional dependency to install for their
    hardware, or that an accelerator is already active.
    """
    caps = caps or detect()
    if caps.cuda:
        return "CUDA backend active (CuPy). No action needed; select it with backend='cupy' or 'auto'."
    if caps.metal:
        return "Metal backend active (MLX). No action needed; select it with backend='mlx' or 'auto'."
    if caps.apple_silicon:
        return "Apple Silicon detected. Enable the Metal GPU backend with: pip install 'hyphi[metal]'"
    if caps.system in ("Linux", "Windows"):
        return (
            "If this machine has an NVIDIA GPU, enable the CUDA backend with: "
            "pip install 'hyphi[cuda]'  (installs a cupy-cuda12x wheel)"
        )
    return "No supported accelerator detected; the vectorized NumPy CPU backend is the fastest available here."
