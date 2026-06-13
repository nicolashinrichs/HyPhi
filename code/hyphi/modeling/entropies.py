"""Compute entropy estimates of curvature distributions, with a name-keyed registry."""

# %% Import
from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from KDEpy import TreeKDE
from scipy.stats import differential_entropy

from hyphi.modeling.graph_curvatures import extract_curvatures

if TYPE_CHECKING:
    import networkx as nx
    import numpy.typing as npt

import numpy as np

# %% Degenerate-input contract and tie dithering >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Forman-Ricci curvature is DISCRETE (integer-valued for unweighted graphs), but the continuous
# entropy estimators below assume continuous data. Ties collapse order-statistic spacings (the
# spacing estimators -> log(0) -> -inf), collapse kNN distances (the kNN estimators -> artefacts),
# and a near-constant distribution breaks the KDE bandwidth selection. We resolve this by DITHERING:
# add uniform jitter of half the smallest nonzero gap between distinct curvature values. For
# unit-spaced integer curvature this is the standard quantization dither U(-1/2, 1/2), under which
# the SHANNON-type differential-entropy estimate recovers the discrete Shannon entropy of the
# curvature histogram (a constant distribution -> ~0, more disorder -> larger). So an ordered/lattice
# window reads as MINIMUM entropy and the transition peak is preserved, while the estimators no longer
# raise or return -inf/NaN. For genuinely continuous data the gap is tiny, so the jitter is negligible
# and estimates are essentially unchanged. The jitter is seeded from the data, so results are reproducible.
#
# Two limits on the Shannon-recovery reading: (1) the kNN Renyi/Tsallis estimators recover their own
# order-2 analogue (Renyi-2 / Tsallis-2), not Shannon; (2) recovery is specific to UNIT label spacing.
# Non-unit integer gaps shift the dither width with the gap and bias the differential-entropy estimate
# by ~log(gap) (unreachable on unweighted FRC, whose spacing is 1). In all cases the orientation
# (ordered < disordered) and the transition detection hold.
#
# NOTE: this changes the numeric entropy values versus the pre-dithering implementation, which
# returned -inf / NaN / artefacts on tied integer curvature. The downstream impact of the estimator
# suite on previously-recorded [MEASURED] results is tracked in issue #11.

# Returned for CONTENTLESS input: too few curvature samples (or non-finite curvature) to estimate a
# distribution. There is no meaningful entropy, so it is defined as 0.0 (the low end, consistent with
# a dithered constant ~ 0).
_CONTENTLESS_ENTROPY = 0.0

# Default neighbour count of the kNN estimators (kozachenko/renyi/tsallis); they need at least k + 1
# samples. Fewer than this many curvature samples cannot be scored by every estimator, so it is
# treated as contentless. (Assumes the default k; a larger custom k on a tiny graph is the caller's
# responsibility.)
_DEFAULT_KNN_K = 4
_MIN_SAMPLES = _DEFAULT_KNN_K + 1

# At least this many DISTINCT values are needed to measure a nonzero gap for the dither.
_MIN_DISTINCT = 2

# Fallback spacing for the dither when the distribution is constant (no nonzero gap between distinct
# values); 1.0 matches the unit spacing of integer Forman-Ricci curvature.
_DEFAULT_GAP = 1.0


def _dither(curvatures: npt.ArrayLike) -> np.ndarray:
    """
    Break ties in a discrete curvature distribution with uniform half-gap jitter.

    Adds ``U(-gap/2, +gap/2)`` noise, where ``gap`` is the smallest nonzero spacing between distinct
    curvature values (``_DEFAULT_GAP`` when the distribution is constant). The jitter is seeded from
    the data so the result is reproducible. For unit-spaced integer curvature this is the standard
    quantization dither (recovering the discrete Shannon entropy); for continuous data the jitter is
    negligible.

    Parameters
    ----------
    curvatures : array-like
        Curvature values extracted from a graph (assumed to have at least two samples).

    Returns
    -------
    np.ndarray
        The dithered curvature values.

    """
    arr = np.asarray(curvatures, dtype=float)
    uniq = np.unique(arr)
    gap = float(np.min(np.diff(uniq))) if uniq.size >= _MIN_DISTINCT else _DEFAULT_GAP
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(arr).tobytes()))
    return arr + rng.uniform(-0.5 * gap, 0.5 * gap, size=arr.shape)


def _entropy_guard(curvatures: npt.ArrayLike) -> float | None:
    """
    Return the contentless-input sentinel (0.0), or None when there is a distribution to estimate.

    Input is contentless when it has fewer than ``_MIN_SAMPLES`` samples (too few for the kNN
    estimators, which need k + 1) or contains non-finite values. An all-constant but sufficiently
    large distribution is NOT special-cased here: :func:`_dither` gives the estimators a unit-width
    uniform to score (~0 entropy), so the ordered case reads as minimum entropy rather than raising.

    Parameters
    ----------
    curvatures : array-like
        The curvature values extracted from a graph.

    Returns
    -------
    float or None
        ``_CONTENTLESS_ENTROPY`` for too-few-sample or non-finite input; otherwise None.

    """
    arr = np.asarray(curvatures, dtype=float)
    if arr.size < _MIN_SAMPLES or not np.isfinite(arr).all():
        return _CONTENTLESS_ENTROPY
    return None


def _prepare_curvatures(curvatures: npt.ArrayLike) -> tuple[float | None, np.ndarray | None]:
    """
    Guard contentless input and dither ties.

    Returns ``(sentinel, None)`` for contentless input (estimate is the sentinel), or
    ``(None, dithered_array)`` when there is a distribution to estimate.
    """
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel, None
    return None, _dither(curvatures)


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
    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    kwargs: dict = {"method": "vasicek", "nan_policy": "omit"}
    if window_length is not None:
        kwargs["window_length"] = window_length
    return differential_entropy(curvatures, **kwargs)


def entropy_van_es(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Van Es entropy estimator on graph curvatures."""
    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    return differential_entropy(curvatures, method="van es", nan_policy="omit")


def entropy_ebrahimi(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Ebrahimi entropy estimator on graph curvatures."""
    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    return differential_entropy(curvatures, method="ebrahimi", nan_policy="omit")


def entropy_correa(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> npt.number | npt.ndarray:
    """Correa entropy estimator on graph curvatures."""
    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
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
    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    try:
        fvals = TreeKDE(kernel=kernel_type, bw=bw, norm=norm).fit(curvatures).evaluate(curvatures)
    except ValueError:
        # Data-driven bandwidth selection (e.g. ISJ) can fail to converge, and the numeric support
        # solver can fail at evaluate(), on a low-spread distribution even after dithering; fall back
        # to a closed-form rule that never root-finds (re-running both fit and evaluate).
        fvals = TreeKDE(kernel=kernel_type, bw="silverman", norm=norm).fit(curvatures).evaluate(curvatures)
    epsilon = 1e-10
    log_fvals = np.log(fvals + epsilon)
    return -np.mean(log_fvals)


# ---------------------
# kNN-based Entropy Estimators
# ---------------------


def entropy_kozachenko(G: nx.classes.graph.Graph, curvature: str = "formanCurvature", k: int = 4) -> float:
    """Kozachenko-Leonenko kNN entropy estimator."""
    import infomeasure as im

    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    return im.entropy(curvatures, approach="metric", k=k)


def entropy_renyi(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Rényi entropy estimator via kNN."""
    import infomeasure as im

    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    return im.entropy(curvatures, approach="renyi", alpha=order, k=k)


def entropy_tsallis(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, k: int = 4
) -> float:
    """Tsallis entropy estimator via kNN."""
    import infomeasure as im

    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
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
