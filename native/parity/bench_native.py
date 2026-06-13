"""Parity and timing harness for the ``hyphi_native`` 1d Forman-Ricci kernel.

This is a runnable gate, not a demo. When the compiled extension ``hyphi_native``
is importable, it builds a panel of weighted Watts-Strogatz graphs, computes the
1d Forman-Ricci curvature on every compiled-in device (always ``cpu``; plus
``cuda`` if :func:`hyphi_native.has_cuda` and ``metal`` if
:func:`hyphi_native.has_metal`), compares each result to a pure-NumPy closed-form
reference computed independently in this file, and ASSERTS the device's parity
tolerance. It then prints a markdown parity and timing table. If the extension is
not importable, it prints a clear "not built" message and exits 0 (a missing
optional accelerator is not a test failure).

The math (node weights = 1), reproduced here as the reference oracle::

    inv_sqrt[k] = 1 / sqrt(we[k])
    S[v]        = sum of inv_sqrt[k] over all edges k incident to node v
    curv[k]     = 4 - sqrt(we[k]) * (S[ei[k]] + S[ej[k]])

This pure-NumPy reference is itself the same expression the validated
``hyphi.backends.cpu_numpy`` backend uses, which is gated bit-for-bit against
``GraphRicciCurvature.FormanRicci(method="1d")`` in the main package tests; this
harness therefore chains the native core to the published reference.

Precision regimes (asserted tolerances):

==========  ===============  ====================================
device      compute dtype    max abs err gate vs the NumPy oracle
==========  ===============  ====================================
cpu         float64          < 1e-10  (exact, ~1e-13 in practice)
cuda        float64          < 1e-10  (exact; double atomicAdd noise)
metal       float32          < 1e-3   (float32, ~1e-6 relative)
==========  ===============  ====================================

The CUDA and Metal paths are a reference implementation: they have not been
compiled or validated on the authoring machine. This harness is what RUNS and
ASSERTS them once the extension is built with the corresponding accelerator on the
target hardware; it never fabricates numbers for a device that is not present.

Usage
-----
Build the extension, then run this file::

    pip install ./native
    python native/parity/bench_native.py

To exercise an accelerator path, build with it enabled (see ``native/README.md``)
on hardware that has the device, then run this file again.
"""

from __future__ import annotations

import sys
import time

import networkx as nx
import numpy as np

#: Per-device maximum-absolute-error gates. float64 paths are exact to ~1e-13; the
#: 1e-10 gate leaves margin for GPU atomic-add reordering. Metal is float32, so its
#: gate is the looser 1e-3 (relative error ~1e-6 scaled by curvature magnitude).
TOLERANCES: dict[str, float] = {"cpu": 1e-10, "cuda": 1e-10, "metal": 1e-3}

#: Watts-Strogatz panel: (n_nodes, k_neighbors, rewire_p, seed). Sizes and rewiring
#: probabilities are varied so parity is checked across edge counts and weight
#: distributions, not a single lucky graph.
PANEL: list[tuple[int, int, float, int]] = [
    (64, 4, 0.1, 0),
    (256, 6, 0.2, 1),
    (1024, 8, 0.3, 2),
    (4096, 10, 0.15, 3),
]


def build_weighted_ws(n: int, k: int, p: float, seed: int) -> nx.Graph:
    """Build a connected weighted Watts-Strogatz graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is joined to its ``k`` nearest neighbors in the ring lattice
        before rewiring (``k`` must be even).
    p : float
        Rewiring probability.
    seed : int
        Seed for both the graph construction and the edge weights, so the panel
        is reproducible.

    Returns
    -------
    networkx.Graph
        Undirected graph with a strictly positive ``"weight"`` on every edge,
        drawn uniformly from ``[0.1, 1.0]`` (the open-ended lower bound keeps
        ``1 / sqrt(w)`` finite and mirrors a normalized connectivity weight).
    """
    g = nx.connected_watts_strogatz_graph(n, k, p, tries=200, seed=seed)
    rng = np.random.default_rng(seed)
    for _, _, data in g.edges(data=True):
        data["weight"] = float(rng.uniform(0.1, 1.0))
    return g


def graph_to_soa(graph: nx.Graph) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a NetworkX graph to the native kernel's SoA inputs.

    Parameters
    ----------
    graph : networkx.Graph
        Undirected weighted graph; missing weights default to 1.0.

    Returns
    -------
    n_nodes : int
        Number of nodes.
    ei, ej : numpy.ndarray
        ``int32`` endpoint indices, length E, in ``graph.edges()`` order.
    we : numpy.ndarray
        ``float64`` edge weights, length E.
    """
    node_index = {label: i for i, label in enumerate(graph.nodes())}
    ei_list: list[int] = []
    ej_list: list[int] = []
    we_list: list[float] = []
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        ei_list.append(node_index[u])
        ej_list.append(node_index[v])
        we_list.append(float(data.get("weight", 1.0)))
    ei = np.asarray(ei_list, dtype=np.int32)
    ej = np.asarray(ej_list, dtype=np.int32)
    we = np.asarray(we_list, dtype=np.float64)
    return len(node_index), ei, ej, we


def forman_1d_reference(
    n_nodes: int, ei: np.ndarray, ej: np.ndarray, we: np.ndarray
) -> np.ndarray:
    """Pure-NumPy float64 closed-form oracle for 1d Forman-Ricci curvature.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (length of the per-node sum ``S``).
    ei, ej : numpy.ndarray
        Endpoint indices, length E.
    we : numpy.ndarray
        Strictly positive edge weights, length E.

    Returns
    -------
    numpy.ndarray
        ``float64`` per-edge curvature ``4 - sqrt(we) * (S[ei] + S[ej])`` where
        ``S[v]`` is the scatter-add of ``1 / sqrt(we)`` over both endpoints of
        every incident edge.
    """
    if ei.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    inv_sqrt = 1.0 / np.sqrt(we)
    s = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(s, ei, inv_sqrt)
    np.add.at(s, ej, inv_sqrt)
    return 4.0 - np.sqrt(we) * (s[ei] + s[ej])


def time_call(fn, repeats: int = 5) -> tuple[np.ndarray, float]:
    """Run ``fn`` once for the result and ``repeats`` times for a best-of timing.

    Parameters
    ----------
    fn : callable
        Zero-argument callable returning a NumPy array.
    repeats : int
        Number of timed repetitions; the minimum wall time is reported (the
        least-noisy estimate of the compute cost).

    Returns
    -------
    result : numpy.ndarray
        The value returned by ``fn``.
    best_ms : float
        Best wall-clock time across ``repeats`` runs, in milliseconds.
    """
    result = fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return result, best * 1e3


def devices_to_test(module) -> list[str]:
    """Return the device names whose paths were compiled into ``module``."""
    devices = ["cpu"]
    if module.has_cuda():
        devices.append("cuda")
    if module.has_metal():
        devices.append("metal")
    return devices


def main() -> int:
    """Run the parity panel, assert tolerances, and print a markdown table.

    Returns
    -------
    int
        Process exit code: 0 when the extension is missing (a non-failure) or when
        every device passes its tolerance gate; 1 if any compiled-in device fails
        parity.
    """
    try:
        import hyphi_native
    except ImportError:
        print(
            "native extension not built; run `pip install ./native` "
            "(and see native/README.md for the CUDA / Metal options). Skipping parity."
        )
        return 0

    print(f"hyphi_native imported (version {getattr(hyphi_native, '__core_version__', '?')})")
    devices = devices_to_test(hyphi_native)
    print(f"compiled-in devices: {', '.join(devices)}")
    if devices == ["cpu"]:
        print(
            "note: cuda / metal paths were not compiled in (has_cuda() and "
            "has_metal() are False); build with the matching cmake.define to test them."
        )
    print()

    rows: list[tuple[str, str, int, float, float, str]] = []
    all_passed = True

    for n, k, p, seed in PANEL:
        graph = build_weighted_ws(n, k, p, seed)
        n_nodes, ei, ej, we = graph_to_soa(graph)
        n_edges = int(we.shape[0])
        reference = forman_1d_reference(n_nodes, ei, ej, we)
        graph_label = f"WS(n={n}, k={k}, p={p})"

        for device in devices:
            tol = TOLERANCES[device]
            try:
                curv, ms = time_call(
                    lambda d=device: hyphi_native.forman_1d(n_nodes, ei, ej, we, device=d)
                )
            except RuntimeError as exc:
                # A compiled-in device whose hardware is absent at runtime raises;
                # report it as a skip rather than a parity failure.
                rows.append((graph_label, device, n_edges, float("nan"), float("nan"),
                             f"SKIP ({exc})"))
                continue

            max_abs_err = float(np.max(np.abs(curv - reference))) if n_edges else 0.0
            passed = max_abs_err < tol
            all_passed = all_passed and passed
            rows.append((graph_label, device, n_edges, max_abs_err, ms,
                         "PASS" if passed else "FAIL"))

    _print_table(rows)

    if not all_passed:
        print("\nPARITY FAILED: at least one device exceeded its tolerance gate.")
        # Hard assertion so a CI invocation fails loudly, never silently.
        assert all_passed, "native parity gate failed; see the table above"
        return 1

    print("\nAll compiled-in devices passed their parity gates.")
    return 0


def _print_table(rows: list[tuple[str, str, int, float, float, str]]) -> None:
    """Print the parity and timing results as a markdown table."""
    print("| graph | device | edges | max abs err | best ms | result |")
    print("|-------|--------|-------|-------------|---------|--------|")
    for graph_label, device, n_edges, err, ms, status in rows:
        err_str = "nan" if err != err else f"{err:.3e}"  # NaN check via self-inequality
        ms_str = "nan" if ms != ms else f"{ms:.3f}"
        print(f"| {graph_label} | {device} | {n_edges} | {err_str} | {ms_str} | {status} |")


if __name__ == "__main__":
    sys.exit(main())
