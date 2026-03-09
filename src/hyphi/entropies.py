# ===========================
# Entropy Estimation Module
# ===========================
"""
Consolidated entropy functions extracted from software_module/Entropies.py.
Provides multiple differential entropy estimators for graph curvature distributions.
"""

import numpy as np
import numpy.typing as npt
import networkx as nx
from typing import List, Callable, Optional

from scipy.stats import differential_entropy
from KDEpy import TreeKDE

from .curvatures import extract_curvatures


# ---------------------
# Spacing-based Entropy Estimators
# ---------------------

def entropy_vasicek(G, curvature="formanCurvature", window_length=None):
    """Vasicek entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    kwargs = {"method": "vasicek", "nan_policy": "omit"}
    if window_length is not None:
        kwargs["window_length"] = window_length
    return differential_entropy(curvatures, **kwargs)


def entropy_van_es(G, curvature="formanCurvature"):
    """Van Es entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="van es", nan_policy="omit")


def entropy_ebrahimi(G, curvature="formanCurvature"):
    """Ebrahimi entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="ebrahimi", nan_policy="omit")


def entropy_correa(G, curvature="formanCurvature"):
    """Correa entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="correa", nan_policy="omit")


# ---------------------
# KDE Plugin Entropy
# ---------------------

def entropy_kde_plugin(G, curvature="formanCurvature", kernel_type="gaussian",
                       bw="ISJ", norm=2):
    """Plugin entropy estimate using TreeKDE.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.
    kernel_type : str
        KDE kernel type.
    bw : str or float
        Bandwidth parameter.
    norm : int
        Norm for TreeKDE.

    Returns
    -------
    float
        Plugin entropy estimate: -E[log f(X)].
    """
    curvatures = extract_curvatures(G, curvature=curvature)
    f = TreeKDE(kernel=kernel_type, bw=bw, norm=norm).fit(curvatures)
    fvals = f.evaluate(curvatures)
    epsilon = 1e-10
    log_fvals = np.log(fvals + epsilon)
    return -np.mean(log_fvals)


# ---------------------
# kNN-based Entropy Estimators
# ---------------------

def entropy_kozachenko(G, curvature="formanCurvature", k=4):
    """Kozachenko-Leonenko kNN entropy estimator."""
    import infomeasure as im
    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="metric", k=k)


def entropy_renyi(G, curvature="formanCurvature", order=2, k=4):
    """Rényi entropy estimator via kNN."""
    import infomeasure as im
    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="renyi", alpha=order, k=k)


def entropy_tsallis(G, curvature="formanCurvature", order=2, k=4):
    """Tsallis entropy estimator via kNN."""
    import infomeasure as im
    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="tsallis", q=order, k=k)


# ---------------------
# Vectorised Helpers
# ---------------------

def vec_entropy(graphs, estimator=None):
    """Compute entropy over a list of curvature-annotated graphs.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Graphs with curvature edge attributes.
    estimator : callable, optional
        Entropy estimator function taking a graph. Defaults to entropy_kozachenko.

    Returns
    -------
    np.ndarray
        Array of entropy values.
    """
    if estimator is None:
        estimator = entropy_kozachenko
    return np.array([estimator(G) for G in graphs])


def get_quantiles(G, qs, curvature="formanCurvature"):
    """Get quantiles of the curvature distribution on a single graph."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return np.quantile(curvatures, qs)


def vec_quantiles(graphs, qs, curvature="formanCurvature"):
    """Get quantiles for a list of graphs."""
    return np.array([get_quantiles(G, qs=qs, curvature=curvature) for G in graphs])
