"""TODO: add description here."""

# %% Import
import math

import networkx as nx
import numpy as np
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_frc(G: nx.Graph, method: str = "1d") -> nx.Graph:  # noqa: N803
    """
    Compute Forman-Ricci curvature on graph G.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    method : str
        FormanRicci method parameter (default "1d").

    Returns
    -------
    nx.Graph
        Graph with 'formanCurvature' edge attribute.

    """
    frc = FormanRicci(G, method=method)
    frc.compute_ricci_curvature()
    return frc.G


def compute_frc_vec(list_of_graphs: list[nx.Graph], method: str = "1d") -> list[nx.Graph]:
    """Get Forman-Ricci curvatures as a vector for a list of graphs."""
    return [compute_frc(G, method=method) for G in list_of_graphs]


def get_orc(
    G: nx.Graph,
    alpha_val: float = 0.5,
    base_val: float = math.e,
    power_val: int = 0,
    method_val: str = "OTDSinkhornMix",
) -> nx.Graph:
    """Get Ollivier-Ricci curvature."""
    # Initialize Ollivier-Ricci Curvature class
    orc = OllivierRicci(G, alpha=alpha_val, base=base_val, exp_power=power_val, method=method_val)
    # Compute the Ollivier-Ricci curvature
    orc.compute_ricci_curvature()
    # Return updated graph with curvature property on edges
    return orc.G


def get_orc_vec(
    list_of_graphs: list[nx.Graph],
    alpha_val: float = 0.5,
    base_val: float = math.e,
    power_val: int = 0,
    method_val: str = "OTDSinkhornMix",
) -> list[nx.Graph]:
    """Get Ollivier-Ricci curvatures as a vector for a list of graphs."""
    return [get_orc(G, alpha_val, base_val, power_val, method_val=method_val) for G in list_of_graphs]


def extract_curvatures(G: nx.Graph, curvature: str = "formanCurvature") -> np.ndarray:
    """Extract curvatures from a graph G."""
    return np.array([ddict[curvature] for u, v, ddict in G.edges(data=True)])


def extract_curvatures_vec(list_of_graphs: list[nx.Graph], curvature: str = "formanCurvature"):
    """Extract curvatures from a graph G as a vector for a list of graphs."""
    return [extract_curvatures(G, curvature=curvature) for G in list_of_graphs]


def compute_afrc(G: nx.Graph) -> nx.Graph:
    """Compute Augmented Forman-Ricci curvature on graph G."""
    return compute_frc(G, method="augmented")


def compute_afrc_vec(list_of_graphs: list[nx.Graph]) -> list[nx.Graph]:
    """Compute AFRC on a list of graphs for a list of graphs."""
    return compute_frc_vec(list_of_graphs, method="augmented")


# ---------------------
# Ollivier-Ricci Curvature
# ---------------------


def compute_orc(
    G: nx.Graph, alpha: float = 0.5, base: float = math.e, exp_power: float = 0, method: str = "OTDSinkhornMix"
) -> nx.Graph:
    """
    Compute Ollivier-Ricci curvature on graph G.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    alpha : float
        Laziness parameter.
    base : float
        Base for the ORC computation.
    exp_power : float
        Exponent power.
    method : str
        ORC method.

    Returns
    -------
    nx.Graph
        Graph with 'ricciCurvature' edge attribute.

    """
    orc = OllivierRicci(G, alpha=alpha, base=base, exp_power=exp_power, method=method)
    orc.compute_ricci_curvature()
    return orc.G


def compute_orc_vec(
    list_of_graphs: list[nx.Graph],
    alpha: float = 0.5,
    base: float = math.e,
    exp_power: float = 0,
    method: str = "OTDSinkhornMix",
) -> list[nx.Graph]:
    """Compute ORC on a list of graphs."""
    return [compute_orc(G, alpha, base, exp_power, method=method) for G in list_of_graphs]


def extract_curvature_matrices(list_of_graphs: list[nx.Graph], curvature: str = "formanCurvature") -> np.ndarray:
    """
    Extract curvature adjacency matrices from a list of curvature-annotated graphs.

    This centralizes the duplicated `nx.attr_matrix` logic from
    HyperCCORRFRC.py, HyperCCORRAugFRC.py, and KuramotoFRC.py.

    Parameters
    ----------
    list_of_graphs : list[nx.Graph]
        Graphs with curvature-edge attributes.
    curvature : str
        Name of the edge attribute to extract.

    Returns
    -------
    np.ndarray
        Array of shape (len(graphs), n_nodes, n_nodes).

    """
    matrices = []
    for G in list_of_graphs:
        mat, _ = nx.attr_matrix(G, edge_attr=curvature)
        matrices.append(np.asarray(mat))
    return np.array(matrices)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
