"""
Reproducible apples-to-apples benchmark and parity report for the curvature backends.

Runs every available backend (NetworkX reference, vectorized NumPy CPU, CuPy
CUDA, MLX Metal) on a grid of graphs and prints, in Markdown:

1. a hardware/software provenance header,
2. a PARITY table (max abs/relative error and Pearson r vs the NetworkX reference),
3. a SINGLE-GRAPH timing table (median wall time per backend per graph size),
4. a SERIES THROUGHPUT table (whole sliding-window series, including
   NumPy across-graph multiprocessing), which is the shape the NeurReps and
   Kuramoto pipelines actually run.

Every number is measured here, not asserted. Same seeds give the same graphs, so
results are reproducible. Timing is the median of repeated runs after a warmup;
correctness is gated against the reference, never against a timing target.

Usage
-----
    python experiments/scripts/bench_curvature_backends.py [--quick]

``--quick`` shrinks the grid for a fast smoke run.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from hyphi import backends


@dataclass
class GraphSpec:
    label: str
    n: int
    k: int
    p: float
    seed: int

    def build(self) -> nx.Graph:
        g = nx.watts_strogatz_graph(self.n, self.k, self.p, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        for u, v in g.edges():
            g[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
        return g


def _median_time(fn, repeats: int) -> float:
    fn()  # warmup (JIT, device upload, allocator)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _hardware_header() -> str:
    caps = backends.detect()
    lines = [
        "## Provenance",
        "",
        f"- Platform: {platform.platform()}",
        f"- Machine: {caps.machine}, logical cores: {caps.cpu_count}",
        f"- Python: {platform.python_version()}, NumPy: {np.__version__}, NetworkX: {nx.__version__}",
        f"- Accelerators detected: CUDA={caps.cuda}, Metal={caps.metal}",
        f"- Available backends: {backends.available_backends()}",
        "",
    ]
    return "\n".join(lines)


def parity_and_single_timing(specs: list[GraphSpec], repeats: int) -> tuple[str, str]:
    names = backends.available_backends()
    ref = backends.get_backend("networkx")

    parity_rows = [
        "| Graph | nodes | edges | backend | dtype | max abs err | max rel err | Pearson r |",
        "|---|---|---|---|---|---|---|---|",
    ]
    timing_rows = ["| Graph | nodes | edges | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]

    for spec in specs:
        g = spec.build()
        arrays = backends.graph_to_arrays(g)
        ref_curv = ref.forman_curvature(arrays, "1d")
        timings = {}
        for name in names:
            be = backends.get_backend(name)
            curv = be.forman_curvature(arrays, "1d")
            abs_err = float(np.max(np.abs(curv - ref_curv)))
            denom = np.maximum(np.abs(ref_curv), 1e-12)
            rel_err = float(np.max(np.abs((curv - ref_curv) / denom)))
            r = float(np.corrcoef(curv, ref_curv)[0, 1])
            parity_rows.append(
                f"| {spec.label} | {arrays.n_nodes} | {arrays.n_edges} | {name} | {be.compute_dtype} | "
                f"{abs_err:.2e} | {rel_err:.2e} | {r:.6f} |"
            )
            timings[name] = _median_time(lambda be=be, arrays=arrays: be.forman_curvature(arrays, "1d"), repeats)
        timing_rows.append(
            f"| {spec.label} | {arrays.n_nodes} | {arrays.n_edges} | "
            + " | ".join(f"{timings[n] * 1e3:.2f} ms" for n in names)
            + " |"
        )

    return "\n".join(parity_rows), "\n".join(timing_rows)


def series_throughput(spec: GraphSpec, n_graphs: int, repeats: int) -> str:
    import multiprocessing as mp

    rng = np.random.default_rng(spec.seed)
    series = []
    for s in range(n_graphs):
        g = nx.watts_strogatz_graph(spec.n, spec.k, spec.p, seed=spec.seed + s)
        for u, v in g.edges():
            g[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
        series.append(g)

    rows = [
        f"Series: {n_graphs} graphs of {spec.n} nodes (Watts-Strogatz k={spec.k}); the NeurReps/Kuramoto shape.",
        "",
        "| Path | total time | per graph | speedup vs reference |",
        "|---|---|---|---|",
    ]

    def run_seq(name):
        be = backends.get_backend(name)
        for g in series:
            be.forman_curvature(backends.graph_to_arrays(g), "1d")

    ref_t = _median_time(lambda: run_seq("networkx"), max(1, repeats // 2))
    rows.append(f"| NetworkX reference (sequential) | {ref_t:.2f} s | {ref_t / n_graphs * 1e3:.1f} ms | 1.0x |")

    np_t = _median_time(lambda: run_seq("numpy"), repeats)
    rows.append(
        f"| NumPy vectorized (sequential) | {np_t:.3f} s | {np_t / n_graphs * 1e3:.2f} ms | {ref_t / np_t:.0f}x |"
    )

    if "mlx" in backends.available_backends():
        mlx_t = _median_time(lambda: run_seq("mlx"), repeats)
        rows.append(
            f"| MLX Metal GPU (sequential) | {mlx_t:.3f} s | {mlx_t / n_graphs * 1e3:.2f} ms | {ref_t / mlx_t:.0f}x |"
        )

    ncpu = min(mp.cpu_count(), 14)
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(ncpu) as pool:
        pool.map(_mp_worker, [backends.graph_to_arrays(g) for g in series])
    mp_t = time.perf_counter() - t0
    rows.append(
        f"| NumPy vectorized + {ncpu}-core multiprocessing | {mp_t:.3f} s | {mp_t / n_graphs * 1e3:.2f} ms | {ref_t / mp_t:.0f}x |"
    )

    return "\n".join(rows)


def _mp_worker(arrays):
    import warnings

    warnings.filterwarnings("ignore")
    return backends.NumpyBackend().forman_curvature(arrays, "1d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        specs = [GraphSpec("WS-small", 200, 6, 0.2, 1), GraphSpec("WS-med", 1000, 10, 0.2, 2)]
        series_spec, n_graphs, repeats = GraphSpec("WS-med", 1000, 10, 0.2, 7), 12, 3
    else:
        specs = [
            GraphSpec("WS-small", 200, 6, 0.2, 1),
            GraphSpec("WS-med", 1000, 10, 0.2, 2),
            GraphSpec("WS-large", 2000, 12, 0.2, 3),
            GraphSpec("WS-dense", 1000, 50, 0.3, 4),
        ]
        series_spec, n_graphs, repeats = GraphSpec("WS-med", 1000, 10, 0.2, 7), 40, 5

    print("# HyPhi curvature backend benchmark\n")
    print(_hardware_header())
    parity, single = parity_and_single_timing(specs, repeats)
    print("## Parity vs NetworkX reference (1d Forman-Ricci)\n")
    print(parity, "\n")
    print("## Single-graph timing (median wall time)\n")
    print(single, "\n")
    print("## Series throughput\n")
    print(series_throughput(series_spec, n_graphs, repeats))
    print("\n_Reproduce: `python experiments/scripts/bench_curvature_backends.py`_")


if __name__ == "__main__":
    sys.exit(main())
