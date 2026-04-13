"""Compute entropy estimates using various methods."""

# %% Import
from typing import TYPE_CHECKING, Callable

import infomeasure as im  # TODO: not listed as deps
import ray  # TODO: not listed as deps
from scipy.stats import differential_entropy

from hyphi.modeling.density_estimation import *
from hyphi.modeling.graph_curvatures import extract_curvatures

if TYPE_CHECKING:
    import numpy.typing as npt
    import networkx as nx

# Compute the entropy distribution of the FRC
# or quantify the diversity or spread of curvature


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_entropy_vasicek(
    G: nx.classes.graph.Graph, window_length: int | None, curvature: str = "formanCurvature"
) -> npt.number | npt.ndarray:
    """Get entropy estimate using Vasicek."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="vasicek", window_length=window_length, nan_policy="omit")


def get_entropy_van_es(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Get entropy estimate using van Es."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="van es", nan_policy="omit")


def get_entropy_ebrahimi(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Get entropy estimate using Ebrahimi."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="ebrahimi", nan_policy="omit")


def get_entropy_corr(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Get entropy correlation."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="correa", nan_policy="omit")


def get_entropy_kde_plugin(
    G: nx.classes.graph.Graph,
    curvature: str = "formanCurvature",
    kernel_type: str = "gaussian",
    bw: str | float | int = "ISJ",
    norm: int = 2,
) -> float:
    """
    Compute plugin entropy estimate using TreeKDE.

    # TODO: update the Parameters (does not match kwargs)
    Parameters:
    - data: array-like, the sample data
    - bw: bandwidth parameter for KDE
    - kernel: kernel function name

    Returns:
    - entropy: plugin entropy estimate
    """
    # Extract the graph curvatures
    curvatures = extract_curvatures(G, curvature=curvature)

    # Fit the TreeKDE estimator
    # We use the TreeKDE because it is faster than naive
    # But unlike the FFTKDE, we can evaluate at arbitrary points
    f = TreeKDE(kernel=kernel_type, bw=bw, norm=norm).fit(curvatures)

    # Evaluate KDE at the original data points
    fvals = f.evaluate(curvatures)

    # Compute plugin entropy: -E[log f(X)]
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    log_fvals = np.log(fvals + epsilon)

    return -np.mean(log_fvals)


# def getEntropyWaveletPlugin(G: nx.classes.graph.Graph):


def get_entropy_kozachenko(G: nx.classes.graph.Graph, curvature: str = "formanCurvature", num_nn: int = 4) -> float:
    """Get entropy estimate using Kozachenko."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="metric", k=num_nn)


def get_entropy_renyi(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, num_nn: int = 4
) -> float:
    """Get entropy estimate using Renyi's algorithm."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="renyi", alpha=order, k=num_nn)


def get_entropy_tsallis(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, num_nn: int = 4
) -> float:
    """Get entropy estimate using Tsallis algorithm."""
    curvatures = extract_curvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="tsallis", q=order, k=num_nn)


def vec_entropy(
    Gt: npt.NDArray[nx.classes.graph.Graph] | list[nx.classes.graph.Graph], estim: Callable = get_entropy_kozachenko
) -> npt.NDArray[float]:
    # TODO: add docstring
    # Define remote function for Ray
    @ray.remote
    def par_estim(g):
        return estim(g)

    # Get Ray futures (object references)
    h_refs = [par_estim.remote(G) for G in Gt]

    # Get entropy results
    h_map = ray.get(h_refs)

    ray.shutdown()

    return np.array(list(h_map))


def get_quantiles(
    G: nx.classes.graph.Graph, qs: npt.NDArray[float] | list[float], curvature: str = "formanCurvature"
) -> npt.NDArray[float]:
    # TODO: add docstring
    curvatures = extract_curvatures(G, curvature=curvature)
    return np.quantile(curvatures, qs)


def vec_quantiles(
    Gt: npt.NDArray[nx.classes.graph.Graph] | list[nx.classes.graph.Graph],
    qs: npt.NDArray[float] | list[float],
    curvature: str = "formanCurvature",
) -> npt.NDArray[float]:
    # TODO: add docstring
    return np.array([get_quantiles(G, qs=qs, curvature=curvature) for G in Gt])


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
