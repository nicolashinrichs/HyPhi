# The functions from here on are just little helpers used in notebooks testing the FRc
# transformation and GDD computation on the resulting network. 
# Some of this functions prepare data for plotting which is done by 
# visualization/GDD_FRc_visualization.py


import numpy as np
import networkx as nx
from hyphi.modeling.graph_curvatures import (
    compute_frc, 
    extract_curvatures, 
)
from hyphi.modeling.curvatures import (
    compute_laplacian_matrix,
    heat_kernel_distance,
)

def compute_global_curvature_stats(curvature_dict):
    """
    Pool curvature arrays from all graphs and compute:
    - mean
    - std
    - median
    - MAD = median absolute deviation from the median
    """
    all_curvatures = np.concatenate(
        [
            np.ravel(np.asarray(curvatures, dtype=float))
            for curvatures in curvature_dict.values()
        ]
    )

    if all_curvatures.size == 0:
        raise ValueError("No curvature values were provided.")

    global_median = np.median(all_curvatures)
    global_mad = np.median(np.abs(all_curvatures - global_median))

    return {
        "all_curvatures": all_curvatures,
        "mean": float(np.mean(all_curvatures)),
        "std": float(np.std(all_curvatures, ddof=0)),
        "median": float(global_median),
        "mad": float(global_mad),
    }


def compute_frc_bundle_from_adjacency(adjacency, method="1d"):
    """
    For one adjacency matrix, compute and return everything that is useful later.
    """
    adjacency = np.asarray(adjacency)

    G = nx.from_numpy_array(adjacency)
    G_frc = compute_frc(G, method=method)

    edge_curvatures = {
        (u, v): data["formanCurvature"]
        for u, v, data in G_frc.edges(data=True)
    }

    curvatures = np.asarray(
        extract_curvatures(G_frc, curvature="formanCurvature"),
        dtype=float,
    )

    return {
        "adjacency": adjacency,
        "graph": G,
        "graph_frc": G_frc,
        "edge_curvatures": edge_curvatures,
        "curvatures": curvatures,
    }


def compute_frc_bundles_from_adjacencies(
    adjacencies,
    method="1d",
    paths=None,
    verbose=False,
):
    """
    Compute FRC bundles for a dictionary of adjacency matrices.

    Parameters
    ----------
    adjacencies : dict
        Dictionary like {1: A1, 2: A2, ...}.
    method : str
        Method passed to compute_frc.
    paths : dict or None
        Optional dictionary like {1: Path(...), 2: Path(...), ...}.
        If provided, each returned bundle gets a "path" entry.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    dict
        Dictionary like {idx: bundle}, where bundle contains:
        adjacency, graph, graph_frc, edge_curvatures, curvatures,
        and optionally path.
    """
    results = {}

    for idx, adjacency in adjacencies.items():
        bundle = compute_frc_bundle_from_adjacency(
            adjacency,
            method=method,
        )

        if paths is not None:
            bundle["path"] = paths[idx]

        results[idx] = bundle

        if verbose:
            print(f"[{idx}] computed FRC bundle")
            if "path" in bundle:
                print("   path:", bundle["path"])
            print("   adjacency shape:", bundle["adjacency"].shape)
            print("   number of curvature values:", bundle["curvatures"].size)

    return results


def compute_successive_gdd(
    weighted_graphs,
    weight_attr="positive_weight",
):
    """
    Compute graph diffusion distance between each graph and the previous one.

    Parameters
    ----------
    weighted_graphs : dict
        Dictionary like {1: G1, 2: G2, ..., 8: G8}, where each graph already
        has the transformed curvature weights stored as an edge attribute.
    weight_attr : str
        Edge attribute to use as Laplacian weight.

    Returns
    -------
    gdd_dict : dict
        Dictionary like {2: d(G2, G1), 3: d(G3, G2), ...}
    """
    indices = sorted(weighted_graphs.keys())

    if len(indices) < 2:
        raise ValueError("Need at least two graphs to compute successive distances.")

    gdd_dict = {}

    for prev_idx, curr_idx in zip(indices[:-1], indices[1:]):
        G_prev = weighted_graphs[prev_idx]
        G_curr = weighted_graphs[curr_idx]

        L_prev = compute_laplacian_matrix(G_prev, g_weight=weight_attr)
        L_curr = compute_laplacian_matrix(G_curr, g_weight=weight_attr)

        if L_prev.shape != L_curr.shape:
            raise ValueError(
                f"Laplacian shape mismatch between graphs {prev_idx} and {curr_idx}: "
                f"{L_prev.shape} vs {L_curr.shape}"
            )

        gdd = heat_kernel_distance(L_prev, L_curr)
        gdd_dict[prev_idx] = float(gdd)

    return gdd_dict

def compute_pairwise_gdd_matrix(
    weighted_graphs,
    weight_attr="positive_weight",
    verbose=False,
):
    """
    Compute the full symmetric pairwise GDD matrix.

    Parameters
    ----------
    weighted_graphs : dict
        Dictionary like {1: G1, 2: G2, ...}, where each graph already
        has the transformed curvature weights stored as an edge attribute.
    weight_attr : str
        Edge attribute to use as Laplacian weight.

    Returns
    -------
    indices : list[int]
        Sorted graph indices used for the matrix ordering.
    distance_matrix : np.ndarray
        Symmetric array with entry (i, j) = GDD(graph_i, graph_j).
    laplacians : dict
        Cached Laplacian matrices keyed by graph index.
    """
    indices = sorted(weighted_graphs.keys())

    if len(indices) < 2:
        raise ValueError("Need at least two graphs to compute pairwise distances.")

    laplacians = {}
    reference_shape = None

    for idx in indices:
        L = compute_laplacian_matrix(weighted_graphs[idx], g_weight=weight_attr)
        laplacians[idx] = L

        if reference_shape is None:
            reference_shape = L.shape
        elif L.shape != reference_shape:
            raise ValueError(
                f"Laplacian shape mismatch for graph {idx}: "
                f"expected {reference_shape}, got {L.shape}"
            )

    n = len(indices)
    distance_matrix = np.zeros((n, n), dtype=float)

    for i, idx_i in enumerate(indices):
        for j in range(i + 1, n):
            idx_j = indices[j]
            d = float(heat_kernel_distance(laplacians[idx_i], laplacians[idx_j]))
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

            if verbose:
                print(f"GDD({idx_i:02d}, {idx_j:02d}) = {d}")

    return indices, distance_matrix, laplacians

