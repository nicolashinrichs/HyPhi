"""
Analyzes module for HyPhi: Graph curvature and entropy computations.

Years: 2026
"""

# %% Import
from typing import Any

import networkx as nx
import numpy as np
from KDEpy import TreeKDE
from scipy.stats import differential_entropy

from .modeling.graph_curvatures import compute_frc_vec, extract_curvatures

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ========================== #
# Sliding Window Graph Build #
# ========================== #


def build_sliding_window_graphs(connectivity_matrix: np.ndarray) -> list[nx.Graph]:
    """
    Convert a 3D or 2D connectivity array into a list of NetworkX graphs.

    Args:
        connectivity_matrix: (windows, nodes, nodes) or (nodes, nodes).

    Returns:
        List of nx.Graph instances with edge weights.

    """
    if connectivity_matrix.ndim == 2:
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
    """Compute plugin entropy estimator using KDE."""
    curvatures = extract_curvatures(G, curvature=curvature)

    # Needs some variation in data to compute KDE
    if len(np.unique(curvatures)) <= 1:
        return 0.0

    f = TreeKDE(kernel=kernel, bw=bw).fit(curvatures)
    fvals = f.evaluate(curvatures)
    epsilon = 1e-10
    log_fvals = np.log(fvals + epsilon)
    return -np.mean(log_fvals)


def compute_entropy_vasicek(
    G: nx.Graph, curvature: str = "formanCurvature", window_length: int | None = None
) -> float:
    """Compute Vasicek entropy estimator."""
    curvatures = extract_curvatures(G, curvature=curvature)
    if len(curvatures) < 2:
        return 0.0
    return differential_entropy(curvatures, method="vasicek", window_length=window_length, nan_policy="omit")


def compute_windowed_curvatures(graphs: list[nx.Graph], method: str = "1d") -> list[nx.Graph]:
    """
    Compute windowed curvatures for a list of graphs.

    Convenience pipeline for FRC.
    """
    return compute_frc_vec(graphs, method=method)


def compute_entropy(graphs: list[nx.Graph], method: str = "vasicek") -> np.ndarray:
    """Compute entropy for a list of graphs."""
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
            "count": int(len(weight_values)),
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
