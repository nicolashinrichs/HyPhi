# ============================
# Curvature Computation Module
# ============================
"""
Consolidated curvature functions extracted from software_module/GraphCurvatures.py.
Supports Forman-Ricci (FRC), Augmented Forman-Ricci (AFRC), and Ollivier-Ricci (ORC).
"""

import math
import numpy as np
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci


# ---------------------
# Forman-Ricci Curvature
# ---------------------

def compute_frc(G, method="1d"):
    """Compute Forman-Ricci curvature on graph G.

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


def compute_frc_vec(graphs, method="1d"):
    """Compute FRC on a list of graphs."""
    return [compute_frc(G, method=method) for G in graphs]


def compute_afrc(G):
    """Compute Augmented Forman-Ricci curvature on graph G."""
    return compute_frc(G, method="augmented")


def compute_afrc_vec(graphs):
    """Compute AFRC on a list of graphs."""
    return compute_frc_vec(graphs, method="augmented")


# ---------------------
# Ollivier-Ricci Curvature
# ---------------------

def compute_orc(G, alpha=0.5, base=math.e, exp_power=0, method="OTDSinkhornMix"):
    """Compute Ollivier-Ricci curvature on graph G.

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


def compute_orc_vec(graphs, alpha=0.5, base=math.e, exp_power=0, method="OTDSinkhornMix"):
    """Compute ORC on a list of graphs."""
    return [compute_orc(G, alpha, base, exp_power, method=method) for G in graphs]


# ---------------------
# Curvature Extraction
# ---------------------

def extract_curvatures(G, curvature="formanCurvature"):
    """Extract curvature values from graph edges as a numpy array."""
    return np.array([ddict[curvature] for u, v, ddict in G.edges(data=True)])


def extract_curvatures_vec(graphs, curvature="formanCurvature"):
    """Extract curvatures from a list of graphs."""
    return [extract_curvatures(G, curvature=curvature) for G in graphs]


def extract_curvature_matrices(graphs, curvature="formanCurvature"):
    """Extract curvature adjacency matrices from a list of curvature-annotated graphs.

    This centralises the duplicated `nx.attr_matrix` logic from
    HyperCCORRFRC.py, HyperCCORRAugFRC.py, and KuramotoFRC.py.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Graphs with curvature edge attributes.
    curvature : str
        Name of the edge attribute to extract.

    Returns
    -------
    np.ndarray
        Array of shape (len(graphs), n_nodes, n_nodes).
    """
    matrices = []
    for G in graphs:
        mat, _ = nx.attr_matrix(G, edge_attr=curvature)
        matrices.append(np.asarray(mat))
    return np.array(matrices)
