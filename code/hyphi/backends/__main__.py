"""
Command-line entry point: ``python -m hyphi.backends``.

Reports the accelerators detected on this machine, the backends that are usable,
the exact ``pip`` command to enable acceleration here, and an optional quick
self-benchmark and parity check so a user can confirm in one command that their
GPU path works and agrees with the CPU reference.

    python -m hyphi.backends            # report capabilities and install hint
    python -m hyphi.backends --bench    # also run a small parity + timing check
"""

from __future__ import annotations

import argparse
import time

import networkx as nx
import numpy as np

from . import available_backends, detect, forman_curvature, get_backend, install_hint, recommend_backend


def _report() -> None:
    caps = detect()
    print("HyPhi acceleration report")
    print("-------------------------")
    print(f"  system / machine : {caps.system} / {caps.machine}")
    print(f"  logical cores    : {caps.cpu_count}")
    print(f"  CUDA (CuPy)      : {caps.cuda}")
    print(f"  Metal (MLX)      : {caps.metal}")
    print(f"  usable backends  : {available_backends()}")
    print(f"  recommended      : {recommend_backend()}  (use backend='auto')")
    print(f"  install hint     : {install_hint()}")


def _bench() -> None:
    g = nx.watts_strogatz_graph(1000, 25, 0.3, seed=0)
    rng = np.random.default_rng(0)
    for u, v in g.edges():
        g[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    ref = forman_curvature(g, "1d", backend="networkx")
    print("\n  quick parity + timing (WS 1000 nodes, dense):")
    for name in available_backends():
        be = get_backend(name)
        t0 = time.perf_counter()
        curv = forman_curvature(g, "1d", backend=name)
        dt = time.perf_counter() - t0
        abs_err = float(np.max(np.abs(curv - ref)))
        print(f"    {name:9s} [{be.compute_dtype:>7s}]  {dt * 1e3:8.2f} ms   max abs err vs reference {abs_err:.2e}")


def main() -> None:
    """Print the acceleration report and, with ``--bench``, a quick parity/timing check."""
    ap = argparse.ArgumentParser(prog="python -m hyphi.backends")
    ap.add_argument("--bench", action="store_true", help="run a quick parity and timing check")
    args = ap.parse_args()
    _report()
    if args.bench:
        _bench()


if __name__ == "__main__":
    main()
