"""Compute entropy estimates using various methods."""

# %% Import
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from KDEpy import TreeKDE
from scipy.stats import differential_entropy

from hyphi.modeling.graph_curvatures import extract_curvatures

if TYPE_CHECKING:
    import networkx as nx
    import numpy.typing as npt

import numpy as np


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ---------------------
# Spacing-based Entropy Estimators
# ---------------------


def entropy_vasicek(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", window_length: int | None = None
) -> npt.number | npt.ndarray:
    """Vasicek entropy estimator on graph curvatures.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.
    window_length : int or None
        Window length for Vasicek estimator.

    Returns
    -------
    float
        Vasicek entropy estimate.
    """
    curvatures = extract_curvatures(G, curvature=curvature)
    kwargs: dict = {"method": "vasicek", "nan_policy": "omit"}
    if window_length is not None:
        kwargs["window_length"] = window_length
    return differential_entropy(curvatures, **kwargs)


def entropy_van_es(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Van Es entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="van es", nan_policy="omit")


def entropy_ebrahimi(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Ebrahimi entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="ebrahimi", nan_policy="omit")


def entropy_correa(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Correa entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="correa", nan_policy="omit")


# ---------------------
# KDE Plugin Entropy
# ---------------------


def entropy_kde_plugin(
    G: nx.classes.graph.Graph,
    curvature: str = "formanCurvature",
    kernel_type: str = "gaussian",
    bw: str | float | int = "ISJ",
    norm: int = 2,
) -> float:
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


def entropy_kozachenko(G: nx.classes.graph.Graph, curvature: str = "formanCurvature", k: int = 4) -> float:
    """Kozachenko-Leonenko kNN entropy estimator."""
    import infomeasure as im

    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="metric", k=k)


def entropy_renyi(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Rényi entropy estimator via kNN."""
    import infomeasure as im

    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="renyi", alpha=order, k=k)


def entropy_tsallis(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Tsallis entropy estimator via kNN."""
    import infomeasure as im

    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="tsallis", q=order, k=k)


# ---------------------
# Vectorised Helpers
# ---------------------


def vec_entropy(
    graphs: npt.NDArray[nx.classes.graph.Graph] | list[nx.classes.graph.Graph],
    estimator: Callable | None = None,
    parallel: bool = False,
) -> npt.NDArray[float]:
    """Compute entropy over a list of curvature-annotated graphs.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Graphs with curvature edge attributes.
    estimator : callable, optional
        Entropy estimator function taking a graph. Defaults to ``entropy_kozachenko``.
    parallel : bool
        If True, use Ray for parallel computation.

    Returns
    -------
    np.ndarray
        Array of entropy values.
    """
    if estimator is None:
        estimator = entropy_kozachenko

    if parallel:
        import ray

        @ray.remote
        def _par_estim(g):
            return estimator(g)

        h_refs = [_par_estim.remote(G) for G in graphs]
        h_map = ray.get(h_refs)
        ray.shutdown()
        return np.array(list(h_map))

    return np.array([estimator(G) for G in graphs])


def get_quantiles(
    G: nx.classes.graph.Graph, qs: npt.NDArray[float] | list[float], curvature: str = "formanCurvature"
) -> npt.NDArray[float]:
    """Get quantiles of the curvature distribution on a single graph."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return np.quantile(curvatures, qs)


def vec_quantiles(
    graphs: npt.NDArray[nx.classes.graph.Graph] | list[nx.classes.graph.Graph],
    qs: npt.NDArray[float] | list[float],
    curvature: str = "formanCurvature",
) -> npt.NDArray[float]:
    """Get quantiles for a list of graphs."""
    return np.array([get_quantiles(G, qs=qs, curvature=curvature) for G in graphs])


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
