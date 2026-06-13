"""
Compute entropy estimates of curvature distributions, with a name-keyed registry.

Estimator suite (issue #11), selectable by name through ``ESTIMATORS`` / ``get_estimator``:

Curvature-distribution estimators (score the edge-curvature histogram of one graph):
  - ``vasicek``, ``van_es``, ``ebrahimi``, ``correa`` -- spacing-based differential entropy (scipy)
  - ``kde_plugin`` (alias ``kde``) -- KDE-plugin Shannon, ``-E[log f]`` (KDEpy)
  - ``kde_renyi``, ``kde_tsallis`` -- KDE-plugin Renyi / Tsallis of a given order (KDEpy)
  - ``kozachenko`` -- Kozachenko-Leonenko kNN differential entropy (infomeasure)
  - ``renyi``, ``tsallis`` -- kNN Renyi / Tsallis of a given order (infomeasure)
  - ``histogram`` -- binned Shannon with Miller-Madow bias correction (numpy)
Spectral estimator (scores the graph itself, not its curvatures):
  - ``von_neumann`` -- entropy of the normalized Laplacian spectrum
Trace-complexity markers (score the windowed entropy series ``H(t)``, NOT registry estimators):
  - ``permutation_entropy``, ``sample_entropy``

All curvature estimators DITHER discrete (integer) curvature to break ties (see the contract below).

DOWNSTREAM IMPACT of the estimator choice (tracked in issue #11): the estimators are NOT
interchangeable, so the choice is a load-bearing methodological parameter, not a free knob.
  - Reproducibility: a ``[MEASURED]`` entropy-over-time result is comparable only to another computed
    with the SAME estimator (and order / k). Record the estimator name with every result.
  - The default (``DEFAULT_ENTROPY_METHOD``) is what ``compute_entropy`` / ``vec_entropy`` use when
    unspecified; changing it silently changes every default-path number.
  - The curvature-vs-paradigm comparison matrix (issue #16) should hold the estimator fixed per row,
    or report it per cell.
  - Orders differ: ``renyi`` / ``tsallis`` (order 2) emphasize the bulk and saturate as a magnitude
    proxy; the Shannon-type estimators are the calibrated entropy. ``von_neumann`` measures a
    DIFFERENT quantity (network structure) and is not magnitude-comparable to the curvature estimators.
  - Sample size: the KDE estimators (``kde_plugin`` / ``kde_renyi`` / ``kde_tsallis``) carry a small
    downward bias at small N, and their dithered-constant floor is a small NEGATIVE N-dependent value
    (a differential-entropy boundary effect), not exactly 0. Record N alongside the estimator name so
    the bias can be bounded.
"""

# %% Import
from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from KDEpy import TreeKDE
from scipy.stats import differential_entropy

from hyphi.modeling.graph_curvatures import extract_curvatures
from hyphi.spectral.laplace import laplace

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

# Tolerance for treating a Renyi/Tsallis order as the Shannon limit (order == 1).
_ORDER_TOL = 1e-9


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
# Spectral (graph) Entropy
# ---------------------


def entropy_von_neumann(G: nx.classes.graph.Graph) -> float:
    """
    Von Neumann (spectral) graph entropy of the normalized Laplacian.

    Unlike the curvature-distribution estimators, this measures the entropy of the NETWORK itself:
    the eigenvalue distribution of the density matrix ``rho = L / tr(L)``, where ``L`` is the
    combinatorial Laplacian (edge weights used if present). It complements the curvature estimators
    and, being spectral, is unaffected by the integer-tie problem. For the complete graph ``K_n`` it
    equals ``log(n - 1)``.

    NOTE: across the Watts-Strogatz rewiring sweep this is NON-MONOTONE (it peaks at the small-world
    regime, so a fully-random graph can read LOWER than the lattice). It is not a monotone disorder
    indicator: pair it with a curvature estimator for transition orientation. Self-loops cancel out of
    ``L`` and are ignored; the measure is defined only for non-negative edge weights.

    Parameters
    ----------
    G : nx.Graph
        Graph; edge weights are used if present.

    Returns
    -------
    float
        ``-sum_i lambda_i log lambda_i`` over the normalized Laplacian spectrum, or the contentless
        sentinel for an edgeless graph.

    """
    import networkx as nx

    if G.number_of_edges() == 0:
        return _CONTENTLESS_ENTROPY
    adjacency = nx.to_numpy_array(G, weight="weight")
    eigenvalues, _, _ = laplace(adjacency)
    total = float(eigenvalues.sum())
    if total <= 0:
        return _CONTENTLESS_ENTROPY
    probs = np.clip(eigenvalues / total, 0.0, None)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ---------------------
# Histogram Entropy
# ---------------------


def entropy_histogram(
    G: nx.classes.graph.Graph, curvature: str = "formanCurvature", bins: str | int = "auto"
) -> float:
    """
    Binned Shannon entropy of the curvature distribution with Miller-Madow bias correction.

    A fast, transparent discretized baseline that handles integer ties natively (no dithering needed).
    The Miller-Madow term ``(K - 1) / (2 N)`` corrects the downward bias of the plugin Shannon entropy
    (``K`` = occupied bins, ``N`` = sample count). For a distribution uniform over ``k`` values this
    returns ``~log k``.

    With the default ``bins="auto"`` and INTEGER-valued curvature (the unweighted Forman-Ricci case),
    each distinct integer is given its own bin, so a uniform over ``k`` integers recovers ``log k``.
    (Numpy's "auto" rule otherwise caps the bin count via the Sturges floor and would merge distinct
    integers, under-reading the entropy on a wide integer range.) For non-integer data, or an explicit
    ``bins``, the specification is passed straight to ``numpy.histogram``. Miller-Madow is calibrated
    for the undersampled regime and slightly over-estimates a fully-sampled uniform.

    Parameters
    ----------
    G : nx.Graph
        Graph with curvature edge attributes.
    curvature : str
        Name of the curvature edge attribute.
    bins : str or int
        Bin specification (default ``"auto"``); see above for the integer-curvature handling.

    Returns
    -------
    float
        Bias-corrected discrete Shannon entropy in nats, or the contentless sentinel.

    """
    curvatures = extract_curvatures(G, curvature=curvature)
    sentinel = _entropy_guard(curvatures)
    if sentinel is not None:
        return sentinel
    arr = np.asarray(curvatures, dtype=float)
    if bins == "auto" and np.allclose(arr, np.round(arr)):
        # Integer-valued curvature: one bin per integer so each distinct value is resolved.
        bins = np.arange(np.floor(arr.min()) - 0.5, np.ceil(arr.max()) + 1.5, 1.0)
    counts, _ = np.histogram(arr, bins=bins)
    counts = counts[counts > 0]
    n_samples = int(counts.sum())
    probs = counts / n_samples
    shannon = float(-np.sum(probs * np.log(probs)))
    miller_madow = (counts.size - 1) / (2 * n_samples)
    return shannon + miller_madow


# ---------------------
# KDE-plugin Renyi / Tsallis
# ---------------------


def _kde_density_at_samples(curvatures: np.ndarray, kernel_type: str, bw: str | float | int, norm: int) -> np.ndarray:
    """
    Fit a TreeKDE to the (dithered) curvatures and evaluate it at the sample points.

    Falls back from the data-driven bandwidth (e.g. ISJ) to silverman when either the fit or the
    evaluate fails to converge on a low-spread distribution.
    """
    try:
        return TreeKDE(kernel=kernel_type, bw=bw, norm=norm).fit(curvatures).evaluate(curvatures)
    except ValueError:
        return TreeKDE(kernel=kernel_type, bw="silverman", norm=norm).fit(curvatures).evaluate(curvatures)


def entropy_kde_renyi(
    G: nx.classes.graph.Graph,
    curvature: str = "formanCurvature",
    order: float | int = 2,
    kernel_type: str = "gaussian",
    bw: str | float | int = "ISJ",
    norm: int = 2,
) -> float:
    """
    KDE-plugin Renyi entropy of order ``order`` on graph curvatures.

    The continuous-density companion of the kNN ``entropy_renyi``:
    ``H_a = 1/(1 - a) * log E_f[f^(a - 1)]``, estimated from a kernel density. At ``order == 1`` it
    reduces to the Shannon plugin (``entropy_kde_plugin``).

    Returns
    -------
    float
        Renyi-``order`` entropy estimate, or the contentless sentinel.

    """
    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    fvals = _kde_density_at_samples(curvatures, kernel_type, bw, norm) + 1e-10
    if abs(order - 1.0) < _ORDER_TOL:
        return float(-np.mean(np.log(fvals)))
    integral = float(np.mean(fvals ** (order - 1)))  # E_f[f^(a-1)] = int f^a dx
    return float(np.log(integral) / (1.0 - order))


def entropy_kde_tsallis(
    G: nx.classes.graph.Graph,
    curvature: str = "formanCurvature",
    order: float | int = 2,
    kernel_type: str = "gaussian",
    bw: str | float | int = "ISJ",
    norm: int = 2,
) -> float:
    """
    KDE-plugin Tsallis entropy of order ``order`` on graph curvatures.

    The continuous-density companion of the kNN ``entropy_tsallis``:
    ``H_q = (1 - E_f[f^(q - 1)]) / (q - 1)``. At ``order == 1`` it reduces to the Shannon plugin.

    Returns
    -------
    float
        Tsallis-``order`` entropy estimate, or the contentless sentinel.

    """
    sentinel, curvatures = _prepare_curvatures(extract_curvatures(G, curvature=curvature))
    if sentinel is not None:
        return sentinel
    fvals = _kde_density_at_samples(curvatures, kernel_type, bw, norm) + 1e-10
    if abs(order - 1.0) < _ORDER_TOL:
        return float(-np.mean(np.log(fvals)))
    integral = float(np.mean(fvals ** (order - 1)))
    return float((1.0 - integral) / (order - 1.0))


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
    "von_neumann": entropy_von_neumann,
    "histogram": entropy_histogram,
    "kde_renyi": entropy_kde_renyi,
    "kde_tsallis": entropy_kde_tsallis,
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


# ---------------------
# Trace-complexity markers (second-order, on the entropy-over-time series H(t))
# ---------------------
#
# These operate on the 1-D entropy-over-time TRACE produced by vec_entropy, not on a single graph, so
# they are NOT registry estimators. They mark a phase transition by the complexity of H(t) rather than
# its level.


def permutation_entropy(series: npt.ArrayLike, order: int = 3) -> float:
    """
    Permutation entropy of a 1-D series: the Shannon entropy of its ordinal-pattern distribution.

    A monotone series has a single ordinal pattern (entropy 0); an i.i.d. series approaches
    ``log(order!)``. Use on the windowed entropy trace ``H(t)`` as a second-order transition marker.

    Parameters
    ----------
    series : array-like
        1-D time series (e.g. the windowed entropy trace).
    order : int
        Embedding dimension (ordinal-pattern length), ``>= 2``.

    Returns
    -------
    float
        Permutation entropy in nats, or 0.0 if the series is too short.

    """
    arr = np.asarray(series, dtype=float).ravel()
    n_patterns = arr.size - order + 1
    if order < _MIN_DISTINCT or n_patterns < 1:
        return 0.0
    pattern_counts: dict[tuple[int, ...], int] = {}
    for i in range(n_patterns):
        key = tuple(int(j) for j in np.argsort(arr[i : i + order], kind="stable"))
        pattern_counts[key] = pattern_counts.get(key, 0) + 1
    counts = np.array(list(pattern_counts.values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def sample_entropy(series: npt.ArrayLike, m: int = 2, r: float | None = None) -> float:
    """
    Sample entropy of a 1-D series: the negative log conditional probability of staying alike.

    ``-log(A / B)``: two subsequences alike for ``m`` points stay alike at the next point
    (self-matches excluded). Low for regular/periodic series, higher for irregular ones. Use on the
    windowed entropy trace.

    Parameters
    ----------
    series : array-like
        1-D time series.
    m : int
        Embedding length.
    r : float, optional
        Chebyshev-distance tolerance; defaults to ``0.2 * std(series)``.

    Returns
    -------
    float
        Sample entropy in nats, or 0.0 if undefined (series too short, or no matches).

    """
    arr = np.asarray(series, dtype=float).ravel()
    n = arr.size
    if n < m + 2:
        return 0.0
    tol = 0.2 * float(np.std(arr)) if r is None else float(r)
    if tol <= 0:
        return 0.0

    def _count_matches(length: int) -> int:
        templates = np.array([arr[i : i + length] for i in range(n - length + 1)])
        total = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            total += int(np.sum(dist <= tol) - 1)  # exclude the self-match
        return total

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0 or a == 0:
        return 0.0
    return float(-np.log(a / b))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
