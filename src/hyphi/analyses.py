import networkx as nx
import numpy as np
import numpy.typing as npt
from typing import List, Callable, Union
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from KDEpy import NaiveKDE, TreeKDE, FFTKDE
from scipy.stats import differential_entropy

# ========================== #
# Sliding Window Graph Build #
# ========================== #

def build_sliding_window_graphs(connectivity_matrix: np.ndarray) -> List[nx.Graph]:
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

# ================ #
# Graph Curvatures #
# ================ #

def compute_frc(G: nx.Graph, method: str = "1d") -> nx.Graph:
    """Compute Forman-Ricci Curvature."""
    frc = FormanRicci(G, method=method)
    frc.compute_ricci_curvature()
    return frc.G

def compute_frc_vec(Gt: List[nx.Graph], method: str = "1d") -> List[nx.Graph]:
    return [compute_frc(G, method=method) for G in Gt]

def compute_orc(G: nx.Graph, alpha: float = 0.5, method: str = "OTDSinkhornMix") -> nx.Graph:
    """Compute Ollivier-Ricci Curvature."""
    orc = OllivierRicci(G, alpha=alpha, method=method)
    orc.compute_ricci_curvature()
    return orc.G

def extract_curvatures(G: nx.Graph, curvature: str = "formanCurvature") -> np.ndarray:
    return np.array([ddict[curvature] for u, v, ddict in G.edges(data=True)])

# ================= #
# Density & Entropy #
# ================= #

def select_kde(kernel: str = "gaussian", bw: Union[str, float] = "ISJ", method: str = "FFT"):
    if method == "naive":
        return NaiveKDE(kernel=kernel, bw=bw)
    elif method == "tree":
        return TreeKDE(kernel=kernel, bw=bw)
    elif method == "FFT":
        return FFTKDE(kernel=kernel, bw=bw)
    else:
        raise ValueError("Method must be naive, tree, or FFT")

def compute_entropy_kde_plugin(G: nx.Graph, curvature: str = "formanCurvature", kernel: str = "gaussian", bw: Union[str, float] = "ISJ") -> float:
    curvatures = extract_curvatures(G, curvature=curvature)
    
    # Needs some variation in data to compute KDE
    if len(np.unique(curvatures)) <= 1:
        return 0.0

    f = TreeKDE(kernel=kernel, bw=bw).fit(curvatures)
    fvals = f.evaluate(curvatures)
    epsilon = 1e-10
    log_fvals = np.log(fvals + epsilon)
    return -np.mean(log_fvals)

def compute_entropy_vasicek(G: nx.Graph, curvature: str = "formanCurvature", window_length: int = None) -> float:
    curvatures = extract_curvatures(G, curvature=curvature)
    if len(curvatures) < 2:
        return 0.0
    return differential_entropy(curvatures, method="vasicek", window_length=window_length, nan_policy="omit")

def compute_windowed_curvatures(graphs: List[nx.Graph], method: str = "1d") -> List[nx.Graph]:
    """Convenience pipeline for FRC."""
    return compute_frc_vec(graphs, method=method)

def compute_entropy(graphs: List[nx.Graph], method: str = "vasicek") -> np.ndarray:
    entropies = []
    for g in graphs:
        if method == "vasicek":
            entropies.append(compute_entropy_vasicek(g))
        elif method == "kde":
            entropies.append(compute_entropy_kde_plugin(g))
        else:
            raise ValueError("Unsupported entropy method")
    return np.array(entropies)
