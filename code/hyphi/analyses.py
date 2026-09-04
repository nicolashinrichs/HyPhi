"""Analyses module for HyPhi: graph curvature and entropy computations."""

# %% Import
from typing import Any

import networkx as nx
import numpy as np
from KDEpy import TreeKDE
from scipy.stats import differential_entropy

from .modeling.graph_curvatures import compute_frc_vec, extract_curvatures

# %% Constants >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Minimum number of distinct values required to fit a KDE or Vasicek estimator.
_MIN_UNIQUE_VALUES = 2
# Number of dimensions for a plain 2-D connectivity matrix (no window axis).
_MATRIX_2D_NDIM = 2


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ========================== #
# Sliding Window Graph Build #
# ========================== #


def build_sliding_window_graphs(connectivity_matrix: np.ndarray) -> list[nx.Graph]:
    """Convert a 3D or 2D connectivity array into a list of NetworkX graphs.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Shape ``(windows, nodes, nodes)`` or ``(nodes, nodes)``.  A 2D array is
        treated as a single window.

    Returns
    -------
    list[nx.Graph]
        One ``nx.Graph`` per window, with edge weights set to the absolute
        values of the connectivity matrix entries.

    """
    if connectivity_matrix.ndim == _MATRIX_2D_NDIM:
        connectivity_matrix = connectivity_matrix[np.newaxis, :, :]

    graphs = []
    for window in range(connectivity_matrix.shape[0]):
        # Graph curvature often expects absolute connectivity values
        w_mat = np.abs(connectivity_matrix[window, :, :])
        G = nx.from_numpy_array(w_mat)
        graphs.append(G)

    return graphs


# ================= #
# Density & Entropy #
# ================= #


def compute_entropy_kde_plugin(
    G: nx.Graph, curvature: str = "formanCurvature", kernel: str = "gaussian", bw: str | float = "ISJ"
) -> float:
    """Compute plugin entropy estimator using kernel density estimation.

    Parameters
    ----------
    G : nx.Graph
        Graph whose edge curvature values are used as samples.
    curvature : str
        Name of the curvature attribute to read from ``G``.
    kernel : str
        KDE kernel passed to :class:`KDEpy.TreeKDE`.
    bw : str or float
        Bandwidth selector; ``"ISJ"`` (Improved Sheather-Jones) is the default.

    Returns
    -------
    float
        Negative mean log-density (plugin entropy estimate).  Returns ``0.0``
        when all curvature values are identical (degenerate distribution).

    """
    curvatures = extract_curvatures(G, curvature=curvature)

    # Needs at least two distinct values to fit a KDE.
    if len(np.unique(curvatures)) < _MIN_UNIQUE_VALUES:
        return 0.0

    f = TreeKDE(kernel=kernel, bw=bw).fit(curvatures)
    fvals = f.evaluate(curvatures)
    epsilon = 1e-10
    log_fvals = np.log(fvals + epsilon)
    return -np.mean(log_fvals)


def compute_entropy_vasicek(
    G: nx.Graph, curvature: str = "formanCurvature", window_length: int | None = None
) -> float:
    """Compute Vasicek entropy estimator from edge curvature values.

    Parameters
    ----------
    G : nx.Graph
        Graph whose edge curvature values are used as samples.
    curvature : str
        Name of the curvature attribute to read from ``G``.
    window_length : int or None
        Window length for the Vasicek spacing estimator.  ``None`` lets
        :func:`scipy.stats.differential_entropy` choose automatically.

    Returns
    -------
    float
        Vasicek differential-entropy estimate.  Returns ``0.0`` when fewer
        than two curvature values are available.

    """
    curvatures = extract_curvatures(G, curvature=curvature)
    if len(curvatures) < _MIN_UNIQUE_VALUES:
        return 0.0
    return differential_entropy(curvatures, method="vasicek", window_length=window_length, nan_policy="omit")


def compute_windowed_curvatures(graphs: list[nx.Graph], method: str = "1d") -> list[nx.Graph]:
    """Compute Forman-Ricci curvatures for a list of windowed graphs.

    Thin convenience wrapper around :func:`compute_frc_vec`.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Sequence of graphs, typically one per sliding window.
    method : str
        FRC computation variant forwarded to :func:`compute_frc_vec`.

    Returns
    -------
    list[nx.Graph]
        Same graphs with curvature attributes attached in place.

    """
    return compute_frc_vec(graphs, method=method)


def compute_entropy(graphs: list[nx.Graph], method: str = "vasicek") -> np.ndarray:
    """Compute entropy for each graph in a list.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Graphs whose curvature distributions are used to estimate entropy.
    method : str
        Entropy estimator to apply: ``"vasicek"`` (default) or ``"kde"``.

    Returns
    -------
    np.ndarray
        1-D array of entropy values, one per input graph.

    Raises
    ------
    ValueError
        If ``method`` is not ``"vasicek"`` or ``"kde"``.

    """
    entropies = []
    for g in graphs:
        if method == "vasicek":
            entropies.append(compute_entropy_vasicek(g))
        elif method == "kde":
            entropies.append(compute_entropy_kde_plugin(g))
        else:
            raise ValueError("Unsupported entropy method")
    return np.array(entropies)


# ========================== #
# Network diagnostic helpers #
# ========================== #


def remove_self_loops_copy(G: nx.Graph) -> nx.Graph:
    """
    Return a copy of ``G`` with all self-loops removed.

    Leaves the input graph untouched.  Useful before plotting or graph-metric
    computation, where self-loops distort layouts and clustering estimates.
    """
    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def prune_graph_by_weight(
    G: nx.Graph,
    threshold: float,
    keep_all_nodes: bool = False,
) -> nx.Graph:
    """
    Drop edges whose ``"weight"`` attribute is below ``threshold``.

    Parameters
    ----------
    G : nx.Graph
        Weighted graph.
    threshold : float
        Edges with ``data["weight"] < threshold`` are dropped; edges at the
        threshold are kept (``>=``).
    keep_all_nodes : bool
        If True, all original nodes are carried over even when they end up
        isolated.  If False (default), isolates resulting from pruning are
        implicitly omitted (only endpoints of surviving edges remain).

    Returns
    -------
    nx.Graph
        New graph of the same class as ``G``.

    """
    H_pruned = G.__class__()
    if keep_all_nodes:
        H_pruned.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if data.get("weight", 0) >= threshold:
            H_pruned.add_edge(u, v, **data)
    return H_pruned


def summarize_network(
    G: nx.Graph,
    show_n: int = 10,
    title: str = "Summary of network",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute descriptive statistics of a (possibly weighted) graph.

    Collects a fixed set of diagnostics useful for sanity-checking a network
    built from a connectivity matrix: node/edge counts, weight distribution,
    top/bottom nodes by degree, self-loops, isolates, connectivity, clustering.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    show_n : int
        Length of "top/bottom N" lists included in the summary.
    title : str
        Heading printed in verbose mode.
    verbose : bool
        If True, pretty-print the summary.  The return value is always the
        full summary dict.

    Returns
    -------
    dict
        Keys include ``directed``, ``n_nodes``, ``n_edges``, ``density``,
        ``weight_stats``, ``is_complete``, ``top_degree``, ``bottom_degree``,
        ``top_weighted_degree``, ``n_self_loops``, ``n_isolates``,
        ``connected``, ``n_components``, ``largest_component_size``,
        ``component_sizes`` (first ``show_n``), and — for undirected graphs —
        ``top_triangles``, ``avg_clustering_unweighted``, ``avg_clustering_weighted``.

    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    summary: dict[str, Any] = {
        "title": title,
        "directed": G.is_directed(),
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G),
    }

    weights = nx.get_edge_attributes(G, "weight")
    weight_values = np.array(list(weights.values()))
    if len(weight_values) > 0:
        summary["weight_stats"] = {
            "count": len(weight_values),
            "min": float(weight_values.min()),
            "max": float(weight_values.max()),
            "mean": float(weight_values.mean()),
            "median": float(np.median(weight_values)),
        }
    else:
        summary["weight_stats"] = None

    deg = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    is_complete = nx.number_of_selfloops(G) == 0 and m == n * (n - 1) // 2
    summary["is_complete"] = bool(is_complete)
    summary["top_degree"] = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:show_n]
    summary["bottom_degree"] = sorted(deg.items(), key=lambda x: x[1])[:show_n]
    summary["top_weighted_degree"] = sorted(strength.items(), key=lambda x: x[1], reverse=True)[:show_n]

    summary["n_self_loops"] = int(nx.number_of_selfloops(G))
    summary["first_self_loops"] = list(nx.selfloop_edges(G, data=True))[:show_n]
    isolates = list(nx.isolates(G))
    summary["n_isolates"] = len(isolates)
    summary["first_isolates"] = isolates[:show_n]

    if n > 0:
        if G.is_directed():
            summary["connected"] = bool(nx.is_weakly_connected(G))
            comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        else:
            summary["connected"] = bool(nx.is_connected(G))
            comps = sorted(nx.connected_components(G), key=len, reverse=True)
        summary["n_components"] = len(comps)
        summary["largest_component_size"] = len(comps[0])
        summary["component_sizes"] = [len(c) for c in comps[:show_n]]
    else:
        summary["connected"] = False
        summary["n_components"] = 0
        summary["largest_component_size"] = 0
        summary["component_sizes"] = []

    if not G.is_directed():
        tri = nx.triangles(G)
        summary["top_triangles"] = sorted(tri.items(), key=lambda x: x[1], reverse=True)[:show_n]
        summary["avg_clustering_unweighted"] = float(nx.average_clustering(G))
        summary["avg_clustering_weighted"] = float(nx.average_clustering(G, weight="weight"))

    if verbose:
        _print_summary(summary, show_n=show_n)
    return summary


def _print_summary(s: dict[str, Any], show_n: int) -> None:
    """Pretty-print a summary dict produced by :func:`summarize_network`."""
    print(s["title"])
    print()
    print(f"directed? {s['directed']}")
    print(f"nodes: {s['n_nodes']}")
    print(f"edges: {s['n_edges']}")
    print(f"density: {s['density']}")

    ws = s["weight_stats"]
    if ws is not None:
        print(f"\nNumber of weighted edges: {ws['count']}")
        print(f"min weight: {ws['min']}")
        print(f"max weight: {ws['max']}")
        print(f"mean weight: {ws['mean']}")
        print(f"median weight: {ws['median']}")
    else:
        print("\nNo edge weights found.")

    print(f"\ncomplete graph? {s['is_complete']}")
    if s["is_complete"]:
        print("This is a complete graph: every node is connected to every other node.")
    else:
        print(f"\nTop {show_n} nodes by degree:")
        print(s["top_degree"])
        print(f"\nBottom {show_n} nodes by degree:")
        print(s["bottom_degree"])
    print(f"\nTop {show_n} nodes by weighted degree:")
    print(s["top_weighted_degree"])

    print(f"\nnumber of self-loops: {s['n_self_loops']}")
    print(f"self-loops (first {show_n}): {s['first_self_loops']}")
    print(f"number of isolates: {s['n_isolates']}")
    print(f"first isolates ({show_n}): {s['first_isolates']}")

    if s["n_nodes"] > 0:
        label = "weakly connected?" if s["directed"] else "connected?"
        print(f"\n{label} {s['connected']}")
        print(f"number of connected components: {s['n_components']}")
        print(f"largest component size: {s['largest_component_size']}")
        print(f"first {show_n} component sizes: {s['component_sizes']}")

    if not s["directed"] and "top_triangles" in s:
        print(f"\nTop {show_n} nodes by triangle count:")
        print(s["top_triangles"])
        print(f"average clustering (unweighted): {s['avg_clustering_unweighted']}")
        print(f"average clustering (weighted): {s['avg_clustering_weighted']}")


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
