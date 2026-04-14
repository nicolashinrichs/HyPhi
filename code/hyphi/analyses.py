"""
Analyzes module for HyPhi: Graph curvature and entropy computations.

Years: 2026
"""

# %% Import
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


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
