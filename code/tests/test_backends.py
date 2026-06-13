"""
Tests for the accelerator backends (hyphi.backends).

Three kinds of check, mirroring the project testing philosophy:

- known-answer: the unweighted 1d Forman curvature has a closed form
  ``F(u, v) = 4 - deg(u) - deg(v)`` (star and complete graphs), asserted
  independently of the reference library;
- parity: every available backend agrees with the GraphRicciCurvature reference,
  at float64 tolerance for the float64 backends and float32 tolerance for the
  Metal/MLX backend;
- contract: registry resolution, capability probes that never raise, graph_io
  round-trip, degenerate (empty) input, and transparent fallback for a method a
  backend does not implement.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from hyphi import backends
from hyphi.backends.capabilities import Capabilities, recommend_backend
from hyphi.backends.graph_io import graph_to_arrays


def _weighted_ws(n=120, k=6, p=0.3, seed=0):
    g = nx.watts_strogatz_graph(n, k, p, seed=seed)
    rng = np.random.default_rng(seed)
    for u, v in g.edges():
        g[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    return g


# --- known-answer (independent of the reference library) ---------------------


def test_known_answer_star_unweighted():
    n_leaves = 7
    g = nx.star_graph(n_leaves)  # node 0 is the hub, 1..n_leaves are leaves
    curv = backends.forman_curvature(g, "1d", backend="numpy")
    # every edge is hub(deg n_leaves)-leaf(deg 1): F = 4 - n_leaves - 1
    assert np.allclose(curv, 4 - n_leaves - 1)


def test_known_answer_complete_unweighted():
    g = nx.complete_graph(5)  # every node degree 4
    curv = backends.forman_curvature(g, "1d", backend="numpy")
    assert np.allclose(curv, 4 - 4 - 4)


# --- parity vs the GraphRicciCurvature reference -----------------------------


@pytest.mark.parametrize("method", ["1d", "augmented"])
def test_numpy_parity_with_reference(method):
    g = _weighted_ws()
    ref = backends.forman_curvature(g, method, backend="networkx")
    got = backends.forman_curvature(g, method, backend="numpy")
    assert np.max(np.abs(got - ref)) < 1e-10


def test_mlx_parity_with_reference_float32():
    if "mlx" not in backends.available_backends():
        pytest.skip("MLX/Metal backend not available on this machine")
    g = _weighted_ws()
    ref = backends.forman_curvature(g, "1d", backend="networkx")
    got = backends.forman_curvature(g, "1d", backend="mlx")
    # float32 GPU path: combined absolute + relative tolerance (a pure relative
    # bound blows up at curvature zero-crossings; a pure absolute bound is too
    # loose for large-magnitude curvature).
    assert np.all(np.abs(got - ref) <= 1e-3 + 1e-5 * np.abs(ref))


def test_cupy_parity_when_available():
    if "cupy" not in backends.available_backends():
        pytest.skip("CuPy/CUDA backend not available on this machine")
    g = _weighted_ws()
    ref = backends.forman_curvature(g, "1d", backend="networkx")
    got = backends.forman_curvature(g, "1d", backend="cupy")
    assert np.max(np.abs(got - ref)) < 1e-10


# --- contract ----------------------------------------------------------------


def test_registry_and_default():
    assert "numpy" in backends.available_backends()
    assert "networkx" in backends.available_backends()
    assert backends.get_backend(None).name == "numpy"
    assert backends.get_backend("numpy").name == "numpy"


def test_auto_selects_available_backend():
    assert backends.get_backend("auto").name in backends.available_backends()


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        backends.get_backend("does-not-exist")


def test_capability_probes_never_raise():
    caps = backends.detect()
    assert caps.cpu_count >= 1
    assert isinstance(backends.install_hint(), str)
    for cls in (backends.NumpyBackend, backends.CupyBackend, backends.MlxBackend, backends.NativeExtBackend):
        assert isinstance(cls.is_available(), bool)


def test_graph_io_roundtrip_order():
    g = _weighted_ws(n=40, k=4, seed=2)
    arrays = graph_to_arrays(g)
    assert arrays.n_edges == g.number_of_edges()
    assert arrays.n_nodes == g.number_of_nodes()
    edge_weights = [d["weight"] for _, _, d in g.edges(data=True)]
    assert np.allclose(arrays.we, edge_weights)


def test_empty_graph_returns_empty():
    g = nx.empty_graph(5)  # nodes, no edges
    curv = backends.forman_curvature(g, "1d", backend="numpy")
    assert curv.shape == (0,)


def test_method_fallback_is_transparent():
    # MLX implements 1d only; augmented must fall back to a CPU backend and match
    if "mlx" not in backends.available_backends():
        pytest.skip("MLX/Metal backend not available on this machine")
    g = _weighted_ws()
    ref = backends.forman_curvature(g, "augmented", backend="networkx")
    got = backends.forman_curvature(g, "augmented", backend="mlx")
    assert np.max(np.abs(got - ref)) < 1e-10


def test_series_and_annotate():
    g = _weighted_ws(n=60, seed=5)
    series = [g, g, g]
    arrays_list = backends.forman_curvature(series, "1d", backend="numpy")
    assert isinstance(arrays_list, list) and len(arrays_list) == 3
    annotated = backends.forman_curvature(series, "1d", backend="numpy", annotate=True)
    first_edge = next(iter(annotated[0].edges(data=True)))
    assert "formanCurvature" in first_edge[2]


@pytest.mark.parametrize(
    ("cuda", "metal", "expected"),
    [(True, False, "cupy"), (False, True, "mlx"), (False, False, "numpy")],
)
def test_recommend_returns_a_real_registry_key(cuda, metal, expected):
    # regression: recommend_backend must return a name get_backend can resolve,
    # not a device label. Previously it returned 'cuda' while the key is 'cupy'.
    caps = Capabilities(system="Linux", machine="x86_64", cpu_count=8, cuda=cuda, metal=metal, apple_silicon=False)
    name = recommend_backend(caps)
    assert name == expected
    assert name in backends._REGISTRY


def test_self_loop_graph_does_not_crash_and_drops_loops():
    # regression: shipped PLV/CCORR graphs carry a self-loop per node; the
    # backend must not crash and must compute on the simple graph.
    g = _weighted_ws(n=40, seed=7)
    for n in g.nodes():
        g.add_edge(n, n, weight=1.0)  # add a self-loop on every node
    assert nx.number_of_selfloops(g) == g.number_of_nodes()

    simple = g.copy()
    simple.remove_edges_from([(u, v) for u, v in simple.edges() if u == v])
    ref = backends.forman_curvature(simple, "1d", backend="networkx")
    got = backends.forman_curvature(g, "1d", backend="numpy")
    assert len(got) == simple.number_of_edges()
    assert np.max(np.abs(got - ref)) < 1e-10

    # annotate must not crash, must drop self-loops, and annotate every edge
    annotated = backends.forman_curvature(g, "1d", backend="numpy", annotate=True)
    assert nx.number_of_selfloops(annotated) == 0
    assert annotated.number_of_edges() == simple.number_of_edges()
    assert all("formanCurvature" in d for _, _, d in annotated.edges(data=True))


@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_non_positive_weight_raises(bad):
    # 1/sqrt(w) is inf at 0 and nan for negative; fail loudly instead of
    # letting non-finite curvature leak into the entropy.
    g = nx.path_graph(4)
    for u, v in g.edges():
        g[u][v]["weight"] = 1.0
    g[0][1]["weight"] = bad
    with pytest.raises(ValueError, match="strictly positive"):
        backends.forman_curvature(g, "1d", backend="numpy")


def test_directed_graph_is_undirected_first():
    dg = nx.DiGraph()
    dg.add_edge(0, 1, weight=0.5)
    dg.add_edge(1, 2, weight=0.3)
    dg.add_edge(2, 0, weight=0.7)
    got = backends.forman_curvature(dg, "1d", backend="numpy")
    ref = backends.forman_curvature(dg.to_undirected(), "1d", backend="numpy")
    assert np.array_equal(got, ref)


def test_string_labeled_nodes_map_correctly():
    # the backend tolerates string labels (the legacy compute_frc crashes on them);
    # curvature must map to the right edges via the SoA index, not iteration order.
    g = nx.Graph()
    for u, v, w in [("a", "b", 0.5), ("b", "c", 0.8), ("a", "c", 0.3), ("c", "d", 0.9)]:
        g.add_edge(u, v, weight=w)
    mine = backends.forman_curvature(g, "1d", backend="numpy")
    ref = backends.forman_curvature(g, "1d", backend="networkx")
    assert np.max(np.abs(mine - ref)) < 1e-10


def test_non_unit_node_weights_raise():
    # the kernels assume unit node weights; a non-unit node weight would silently
    # change the curvature versus compute_frc, so it must be rejected loudly.
    g = _weighted_ws(n=20, seed=3)
    first = next(iter(g.nodes()))
    g.nodes[first]["weight"] = 2.0
    with pytest.raises(ValueError, match="node weight"):
        backends.forman_curvature(g, "1d", backend="numpy")


def test_multigraph_rejected():
    mg = nx.MultiGraph()
    mg.add_edge(0, 1, weight=0.5)
    mg.add_edge(0, 1, weight=0.7)  # parallel edge
    with pytest.raises(ValueError, match="MultiGraph"):
        backends.forman_curvature(mg, "1d", backend="numpy")


def test_generator_of_graphs_is_a_series():
    graphs = (_weighted_ws(n=30, seed=s) for s in range(3))
    out = backends.forman_curvature(graphs, "1d", backend="numpy")
    assert isinstance(out, list) and len(out) == 3


def test_map_curvature_series_parallel_matches_serial():
    from hyphi.backends.hpc import map_curvature_series

    series = [_weighted_ws(n=40, seed=s) for s in range(5)]
    serial = map_curvature_series(series, backend="numpy", n_procs=1)
    parallel = map_curvature_series(series, backend="numpy", n_procs=4)
    assert len(parallel) == len(serial)
    for a, b in zip(serial, parallel, strict=True):
        assert np.array_equal(a, b)


def test_gpu_backend_series_does_not_deadlock():
    """A GPU backend requested with n_procs > 1 must NOT enter the process pool: GPU workers inherit a
    broken device context and deadlock (~35% of the time with Metal). It is resolved in the parent and
    run serially. Repeated to catch the intermittent regression (a hang here is a CI timeout)."""
    if "mlx" not in backends.available_backends():
        pytest.skip("mlx (Metal) backend not available")
    from hyphi.backends.hpc import map_curvature_series

    series = [_weighted_ws(n=40, seed=s) for s in range(5)]
    serial = map_curvature_series(series, backend="mlx", n_procs=1)
    for _ in range(3):
        out = map_curvature_series(series, backend="mlx", n_procs=4)
        assert len(out) == len(serial)
        for a, b in zip(serial, out, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-3)  # mlx is float32


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_weight_raises(bad):
    # A zero-variance PLV/CCORR window yields a NaN correlation; nan/inf both slip past a naive
    # `w <= 0` guard, so reject them explicitly rather than leaking non-finite curvature.
    g = nx.path_graph(4)
    for u, v in g.edges():
        g[u][v]["weight"] = 1.0
    g[0][1]["weight"] = bad
    with pytest.raises(ValueError, match="positive and finite"):
        backends.forman_curvature(g, "1d", backend="numpy")
