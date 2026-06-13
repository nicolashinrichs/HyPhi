"""Compute entropy estimates of curvature distributions, with a name-keyed registry."""

# %% Import
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from KDEpy import TreeKDE
from scipy.stats import differential_entropy

from hyphi.modeling.graph_curvatures import extract_curvatures

if TYPE_CHECKING:
    import networkx as nx
    import numpy.typing as npt

import numpy as np

# %% Degenerate-input contract >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Sentinel returned for an EXACTLY-degenerate curvature distribution (empty, a single value, or
# all identical). Such a distribution has no meaningful differential entropy, and an unguarded
# estimator either raises or returns NaN/inf (the KDE bandwidth selection divides by zero).
# Returning 0.0 keeps the entropy-over-time trace finite and matches the prior guarded behaviour
# in analyses.py.
#
# KNOWN LIMITATIONS (tracked for the entropy-suite follow-up, issues #11 / #28):
#  1. This guard keys on EXACT distinctness (np.unique < 2), so a NEAR-constant or
#     few-distinct-with-ties distribution slips past it. Forman-Ricci curvature is
#     integer-valued, so near-lattice graphs across the small-world transition still reach the
#     estimators, where the spacing estimators (vasicek/van_es/ebrahimi/correa) return -inf/NaN
#     on tied order statistics and kde_plugin can still raise. A variance/tolerance test (or a
#     nonzero-spacing count) is the proper fix.
#  2. 0.0 reads as "no disorder" only for the bounded estimators (renyi/tsallis/kde_plugin). For
#     the unbounded kNN/spacing estimators (including the default kozachenko) real values are
#     strongly negative, so 0.0 is the HIGH-entropy extreme, not a low reading. Whether the
#     sentinel should be NaN, an explicit error, or estimator-aware is an open scientific choice.
_DEGENERATE_ENTROPY = 0.0

# Minimum number of distinct curvature values required to estimate entropy. Fewer than this
# (an empty array, a single value, or an all-constant distribution) is degenerate. See the
# KNOWN LIMITATIONS above: this exact-distinctness test does not catch near-constant input.
_MIN_DISTINCT_VALUES = 2


def _entropy_guard(curvatures: npt.ArrayLike) -> float | None:
    """
    Return the degenerate-input sentinel, or None when the input is well-formed.

    Parameters
    ----------
    curvatures : array-like
        The curvature values extracted from a graph.

    Returns
    -------
    float or None
        ``_DEGENERATE_ENTROPY`` when the input has fewer than two distinct values (empty,
        a single value, or constant); otherwise None, meaning estimate normally.

    """
    arr = np.asarray(curvatures, dtype=float)
    if np.unique(arr).size < _MIN_DISTINCT_VALUES:
        return _DEGENERATE_ENTROPY
    return None


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ---------------------
# Spacing-based Entropy Estimators
# ---------------------


def entropy_vasicek(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", window_length: int | None = None
) -> npt.number | npt.ndarray:
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
        Vasicek entropy estimate, or the degenerate-input sentinel.

    """
    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
    kwargs: dict = {"method": "vasicek", "nan_policy": "omit"}
    if window_length is not None:
        kwargs["window_length"] = window_length
    return differential_entropy(curvatures, **kwargs)


def entropy_van_es(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Van Es entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
    return differential_entropy(curvatures, method="van es", nan_policy="omit")


def entropy_ebrahimi(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Ebrahimi entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
    return differential_entropy(curvatures, method="ebrahimi", nan_policy="omit")


def entropy_correa(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Correa entropy estimator on graph curvatures."""
    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
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
        Plugin entropy estimate ``-E[log f(X)]``, or the degenerate-input sentinel.

    """
    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
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
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
    return im.entropy(curvatures, approach="metric", k=k)


def entropy_renyi(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Rényi entropy estimator via kNN."""
    import infomeasure as im

    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
    return im.entropy(curvatures, approach="renyi", alpha=order, k=k)


def entropy_tsallis(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Tsallis entropy estimator via kNN."""
    import infomeasure as im

    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
    return im.entropy(curvatures, approach="tsallis", q=order, k=k)


# ---------------------
# Estimator registry
# ---------------------

# Name-keyed registry of every entropy estimator, so the high-level API can dispatch by name
# instead of a hard-coded if/elif. "kde" is kept as an alias for "kde_plugin" for backwards
# compatibility with the previous compute_entropy interface.
ESTIMATORS: dict[str, Callable] = {
    "vasicek": entropy_vasicek,
    "van_es": entropy_van_es,
    "ebrahimi": entropy_ebrahimi,
    "correa": entropy_correa,
    "kde_plugin": entropy_kde_plugin,
    "kde": entropy_kde_plugin,
    "kozachenko": entropy_kozachenko,
    "renyi": entropy_renyi,
    "tsallis": entropy_tsallis,
}

# The single default estimator shared by compute_entropy and vec_entropy. This is the estimator
# the analysis pipeline (run_ws_sweep, the experiment scripts) already used through vec_entropy.
DEFAULT_ENTROPY_METHOD = "kozachenko"


def get_estimator(method: str) -> Callable:
    """
    Resolve an entropy estimator by name.

    Parameters
    ----------
    method : str
        One of the keys of ``ESTIMATORS``.

    Returns
    -------
    callable
        The estimator function.

    Raises
    ------
    ValueError
        If ``method`` is not a registered estimator name.

    """
    try:
        return ESTIMATORS[method]
    except KeyError:
        valid = ", ".join(sorted(ESTIMATORS))
        msg = f"Unknown entropy method {method!r}; choose from: {valid}"
        raise ValueError(msg) from None


# ---------------------
# Vectorised Helpers
# ---------------------


def vec_entropy(
    graphs: npt.NDArray[nx.classes.graph.Graph] | list[nx.classes.graph.Graph],
    estimator: Callable | None = None,
    parallel: bool = False,
) -> npt.NDArray[float]:
    """
    Compute entropy over a list of curvature-annotated graphs.

    Parameters
    ----------
    graphs : list[nx.Graph]
        Graphs with curvature edge attributes.
    estimator : callable, optional
        Entropy estimator function taking a graph. Defaults to the registry's
        ``DEFAULT_ENTROPY_METHOD``.
    parallel : bool
        If True, use Ray for parallel computation.

    Returns
    -------
    np.ndarray
        Array of entropy values.

    """
    if estimator is None:
        estimator = ESTIMATORS[DEFAULT_ENTROPY_METHOD]

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
