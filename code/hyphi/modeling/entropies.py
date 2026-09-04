"""Compute entropy estimates using various methods."""

# %% Import
from __future__ import annotations

from typing import TYPE_CHECKING

from KDEpy import TreeKDE
from scipy.stats import differential_entropy

from hyphi.modeling.graph_curvatures import extract_curvatures

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx
    import numpy.typing as npt

import numpy as np

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ---------------------
# Spacing-based Entropy Estimators
# ---------------------


def entropy_vasicek(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", window_length: int | None = None
) -> np.floating | np.ndarray:
    """
    Vasicek entropy estimator on graph curvatures.

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


def entropy_van_es(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> np.floating | np.ndarray:
    """Compute Van Es entropy estimate on graph curvatures.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.

    Returns
    -------
    float
        Van Es entropy estimate.

    """
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="van es", nan_policy="omit")


def entropy_ebrahimi(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> np.floating | np.ndarray:
    """Compute Ebrahimi entropy estimate on graph curvatures.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.

    Returns
    -------
    float
        Ebrahimi entropy estimate.

    """
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="ebrahimi", nan_policy="omit")


def entropy_correa(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> np.floating | np.ndarray:
    """Compute Correa entropy estimate on graph curvatures.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.

    Returns
    -------
    float
        Correa entropy estimate.

    """
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
    """
    Plugin entropy estimate using TreeKDE.

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
    """Compute Kozachenko-Leonenko kNN entropy estimate on graph curvatures.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.
    k : int
        Number of nearest neighbours.

    Returns
    -------
    float
        Kozachenko-Leonenko entropy estimate.

    """
    import infomeasure as im  # noqa: PLC0415 (optional heavy dependency, deferred to call time)

    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="metric", k=k)


def entropy_renyi(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Compute Renyi entropy estimate via kNN on graph curvatures.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.
    order : float or int
        Order of the Renyi entropy (alpha parameter).
    k : int
        Number of nearest neighbours.

    Returns
    -------
    float
        Renyi entropy estimate.

    """
    import infomeasure as im  # noqa: PLC0415 (optional heavy dependency, deferred to call time)

    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="renyi", alpha=order, k=k)


def entropy_tsallis(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Compute Tsallis entropy estimate via kNN on graph curvatures.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.
    order : float or int
        Order of the Tsallis entropy (q parameter).
    k : int
        Number of nearest neighbours.

    Returns
    -------
    float
        Tsallis entropy estimate.

    """
    import infomeasure as im  # noqa: PLC0415 (optional heavy dependency, deferred to call time)

    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="tsallis", q=order, k=k)


# ---------------------
# Vectorised Helpers
# ---------------------


def vec_entropy(
    graphs: np.ndarray | list[nx.classes.graph.Graph],
    estimator: Callable | None = None,
    parallel: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Compute entropy over a list of curvature-annotated graphs.

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
        import ray  # noqa: PLC0415 (optional heavy dependency, deferred to call time)

        @ray.remote
        def _par_estim(g):
            return estimator(g)

        h_refs = [_par_estim.remote(G) for G in graphs]
        h_map = ray.get(h_refs)
        ray.shutdown()
        return np.array(list(h_map))

    return np.array([estimator(G) for G in graphs])


def get_quantiles(
    G: nx.classes.graph.Graph, qs: npt.NDArray[np.float64] | list[float], curvature: str = "formanCurvature"
) -> npt.NDArray[np.float64]:
    """Return quantiles of the curvature distribution on a single graph.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    qs : array-like of float
        Quantile levels in [0, 1].
    curvature : str
        Name of the curvature edge attribute.

    Returns
    -------
    np.ndarray
        Array of quantile values corresponding to ``qs``.

    """
    curvatures = extract_curvatures(G, curvature=curvature)
    return np.quantile(curvatures, qs)


def vec_quantiles(
    graphs: np.ndarray | list[nx.classes.graph.Graph],
    qs: npt.NDArray[np.float64] | list[float],
    curvature: str = "formanCurvature",
) -> npt.NDArray[np.float64]:
    """Return quantiles of curvature distributions for a list of graphs.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Graphs with curvature edge attributes.
    qs : array-like of float
        Quantile levels in [0, 1].
    curvature : str
        Name of the curvature edge attribute.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(graphs), len(qs))`` with quantile values.

    """
    return np.array([get_quantiles(G, qs=qs, curvature=curvature) for G in graphs])


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
