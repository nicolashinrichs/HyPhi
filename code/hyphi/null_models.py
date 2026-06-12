"""
Null-model / surrogate generators for HyPhi.

Three families of controls matched to the reviewer's request (2026-04):

1. **Signal-level surrogates** — amplitude-preserving phase randomization and
   channel-wise circular time-shift.  Preserve each channel's spectral /
   autocorrelation structure while destroying cross-channel / cross-brain
   phase alignment.  Use these when raw phase/signal is available (simulation
   path).
2. **Dyad-level surrogates** — subject swapping across dyads and dyad-label
   shuffling.  Break the real dyadic pairing / assignment to produce a
   pseudo-dyad null.  Use these to show curvature effects reflect genuine
   inter-brain coupling, not within-brain structure.
3. **Condition-level surrogates** — within-dyad, trial-level condition-label
   shuffle.  Remove the condition signal while preserving dyad- and
   trial-level structure.  Matches the permutation scheme in
   :func:`hyphi.stats.hierarchical_permutation_test`.

Every generator accepts an optional :class:`numpy.random.Generator` so runs
are reproducible from a seed.

Years: 2026
"""

from __future__ import annotations

# %% Import
import logging

import networkx as nx
import numpy as np

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
logger = logging.getLogger(__name__)

__all__ = [
    "circular_time_shift",
    "condition_label_shuffle_within_dyad",
    "configuration_model_null",
    "degree_preserving_rewire",
    "dyad_label_shuffle",
    "dyad_subject_swap",
    "generate_surrogate_stack",
    "phase_randomize",
]

_VALID_METHODS = {"phase_randomize", "circular_time_shift", "dyad_subject_swap"}


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _random_derangement(n: int, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    """Random derangement of ``[0, n)`` — a permutation with no fixed points."""
    if n < 2:
        return np.arange(n)
    base = np.arange(n)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if not np.any(perm == base):
            return perm
    # Rejection sampling exhausted. Fall back to a deterministic rotation and warn: the
    # rotation is a valid derangement but not a uniformly random one, so the resulting null
    # is slightly biased. In practice this only triggers for pathologically small n.
    logger.warning(
        "_random_derangement: no random derangement found in %d tries for n=%d; "
        "falling back to a deterministic rotation (surrogate not uniformly random).",
        max_tries,
        n,
    )
    return np.roll(base, 1)


# ---------------------
# Signal-level surrogates
# ---------------------


def phase_randomize(
    signal: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Amplitude-preserving phase-randomization surrogate (Prichard & Theiler, 1994).

    Randomises the Fourier phases channel-wise while preserving each channel's
    power spectrum.  Destroys cross-channel / cross-brain phase relationships
    and any temporal structure that depends on phase.

    Parameters
    ----------
    signal : np.ndarray
        ``(T,)`` or ``(n_channels, T)``.
    rng : np.random.Generator, optional
        RNG.

    Returns
    -------
    np.ndarray
        Surrogate with the input's shape and power spectrum.

    """
    rng = _as_rng(rng)
    one_d = signal.ndim == 1
    arr = signal[np.newaxis, :] if one_d else signal
    n_channels, T = arr.shape

    out = np.empty_like(arr, dtype=float)
    for ch in range(n_channels):
        fft_vals = np.fft.rfft(arr[ch])
        amp = np.abs(fft_vals)
        random_phases = rng.uniform(0.0, 2 * np.pi, size=len(fft_vals))
        random_phases[0] = 0.0  # preserve DC
        if T % 2 == 0:
            random_phases[-1] = 0.0  # preserve Nyquist bin (real)
        out[ch] = np.fft.irfft(amp * np.exp(1j * random_phases), n=T)

    return out[0] if one_d else out


def circular_time_shift(
    signal: np.ndarray,
    min_shift: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Channel-wise circular time-shift surrogate.

    Each channel is rotated by a random offset in
    ``[min_shift, T - min_shift)``.  Preserves each channel's autocorrelation
    and amplitude distribution but destroys cross-channel temporal alignment.

    Parameters
    ----------
    signal : np.ndarray
        ``(T,)`` or ``(n_channels, T)``.
    min_shift : int
        Minimum absolute shift (avoids trivially small rotations).
    rng : np.random.Generator, optional
        RNG.

    Returns
    -------
    np.ndarray
        Shifted surrogate.

    """
    rng = _as_rng(rng)
    one_d = signal.ndim == 1
    arr = signal[np.newaxis, :] if one_d else signal
    n_channels, T = arr.shape
    if 2 * min_shift >= T:
        raise ValueError(f"Signal length {T} too short for min_shift={min_shift}.")

    out = np.empty_like(arr)
    for ch in range(n_channels):
        shift = int(rng.integers(min_shift, T - min_shift))
        out[ch] = np.roll(arr[ch], shift)

    return out[0] if one_d else out


# ---------------------
# Dyad-level surrogates
# ---------------------


def dyad_subject_swap(
    data_matrix: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate pseudo-dyads by deranging subject-B across real dyads.

    The B-subject of each dyad is replaced by the B-subject of a different
    dyad, chosen from a random derangement of dyad indices so no dyad keeps
    its original pairing.

    Parameters
    ----------
    data_matrix : np.ndarray
        Shape ``(n_dyads, 2, T)`` or ``(n_dyads, 2, n_channels, T)``.  Axis 1
        indexes subject A (0) and subject B (1).
    rng : np.random.Generator, optional
        RNG.

    Returns
    -------
    np.ndarray
        Same shape as input, with subject-B deranged.

    """
    rng = _as_rng(rng)
    n_dyads = data_matrix.shape[0]
    if n_dyads < 2:
        logger.warning("dyad_subject_swap: n_dyads=%d — nothing to swap.", n_dyads)
        return data_matrix.copy()

    perm = _random_derangement(n_dyads, rng)
    out = data_matrix.copy()
    out[:, 1] = data_matrix[perm, 1]
    return out


def dyad_label_shuffle(
    dyad_labels: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Shuffle dyad assignment across observations.

    Returns a permutation of the input labels, preserving each dyad's total
    count but breaking the observation → dyad association.  Useful as a
    sanity null: a mixed-effects model with ``groups=shuffled_labels`` should
    yield a near-zero group variance component, while the true labels should
    not.

    Parameters
    ----------
    dyad_labels : np.ndarray
        Shape ``(n_observations,)``.
    rng : np.random.Generator, optional
        RNG.

    Returns
    -------
    np.ndarray
        Shape ``(n_observations,)`` with labels permuted uniformly at random.

    """
    rng = _as_rng(rng)
    return rng.permutation(np.asarray(dyad_labels))


def condition_label_shuffle_within_dyad(
    condition_labels: np.ndarray,
    dyad_labels: np.ndarray,
    trial_labels: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Shuffle condition labels within each dyad.

    If ``trial_labels`` is provided, shuffling happens at the trial level:
    all rows sharing a ``(dyad, trial)`` keep a single, jointly permuted
    condition label — preserving the window-within-trial block structure,
    as required by the hierarchical permutation scheme.  Otherwise labels
    are permuted freely across all observations of each dyad.

    Parameters
    ----------
    condition_labels : np.ndarray
        Per-observation condition labels.
    dyad_labels : np.ndarray
        Per-observation dyad labels (same length).
    trial_labels : np.ndarray, optional
        Per-observation trial ids (same length).  Strongly recommended for
        windowed hyperscanning data.
    rng : np.random.Generator, optional
        RNG.

    Returns
    -------
    np.ndarray
        Shuffled condition labels, same shape as input.

    """
    rng = _as_rng(rng)
    cond = np.asarray(condition_labels).copy()
    dyad = np.asarray(dyad_labels)

    if trial_labels is None:
        for d in np.unique(dyad):
            mask = dyad == d
            cond[mask] = rng.permutation(cond[mask])
        return cond

    trial = np.asarray(trial_labels)
    for d in np.unique(dyad):
        mask = dyad == d
        sub_trials = trial[mask]
        sub_cond = cond[mask]
        uniq_trials, first_idx = np.unique(sub_trials, return_index=True)
        trial_cond = sub_cond[first_idx]
        perm = rng.permutation(len(uniq_trials))
        mapping = dict(zip(uniq_trials, trial_cond[perm]))
        cond[mask] = np.array([mapping[t] for t in sub_trials])
    return cond


# ---------------------
# Convenience: generate a stack of surrogates
# ---------------------


def generate_surrogate_stack(
    data: np.ndarray,
    method: str = "phase_randomize",
    n_surrogates: int = 100,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate ``n_surrogates`` stacked copies of ``data`` under the requested null.

    Parameters
    ----------
    data : np.ndarray
        Input; the required shape depends on ``method``.
    method : {"phase_randomize", "circular_time_shift", "dyad_subject_swap"}
        Which signal/dyad-level surrogate to apply.
    n_surrogates : int
        Number of surrogates to generate.
    rng : np.random.Generator, optional
        RNG.
    **kwargs
        Forwarded to the underlying generator (e.g. ``min_shift``).

    Returns
    -------
    np.ndarray
        Stacked surrogates, shape ``(n_surrogates, *data.shape)``.

    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(_VALID_METHODS)}; got {method!r}")
    rng = _as_rng(rng)
    dispatch = {
        "phase_randomize": phase_randomize,
        "circular_time_shift": circular_time_shift,
        "dyad_subject_swap": dyad_subject_swap,
    }
    fn = dispatch[method]
    return np.stack([fn(data, rng=rng, **kwargs) for _ in range(n_surrogates)], axis=0)


# ---------------------
# Intra-brain (single-graph) surrogates
# ---------------------
# The dyad-level surrogates above break a real dyadic pairing, so they need at least two brains.
# For a single-brain (intra-brain) network there is no pairing to break; the appropriate null
# keeps each node's degree (the connection budget) while randomising which nodes are connected.
# These two generators provide that degree-matched null, so a curvature effect on one brain can
# be tested against a null that controls for the degree sequence.

# A double-edge swap needs at least two edges to swap and four nodes to keep it simple.
_MIN_SWAP_EDGES = 2
_MIN_SWAP_NODES = 4


def degree_preserving_rewire(
    graph: nx.Graph,
    n_swaps: int | None = None,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    """
    Degree-preserving edge-rewiring surrogate of a single graph.

    Randomises the topology by repeated double-edge swaps, which keep every node's degree
    fixed. Operates on the connection structure (degree sequence); edge weights are not
    preserved. Returns a copy; the input graph is untouched.

    Parameters
    ----------
    graph : nx.Graph
        The single-brain connectivity graph.
    n_swaps : int, optional
        Number of double-edge swaps to attempt. Defaults to ``10 * n_edges``.
    rng : np.random.Generator, optional
        RNG; pass a seeded generator for reproducibility.

    Returns
    -------
    nx.Graph
        A degree-matched rewired copy.

    """
    rng = _as_rng(rng)
    rewired = graph.copy()
    n_edges = rewired.number_of_edges()
    if n_edges < _MIN_SWAP_EDGES or rewired.number_of_nodes() < _MIN_SWAP_NODES:
        # Too few edges or nodes for a valid double-edge swap; nothing to randomise.
        return rewired
    n_swaps = n_swaps if n_swaps is not None else 10 * n_edges
    seed = int(rng.integers(0, 2**32 - 1))
    nx.double_edge_swap(rewired, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
    return rewired


def configuration_model_null(
    graph: nx.Graph,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    """
    Configuration-model surrogate of a single graph.

    Builds a new simple graph with the same degree sequence as the input, randomising the
    edges. Self-loops and parallel edges from the multigraph construction are removed, so the
    realised degree sequence can be slightly below the target for dense graphs. Operates on
    the degree sequence; edge weights are not preserved.

    Parameters
    ----------
    graph : nx.Graph
        The single-brain connectivity graph.
    rng : np.random.Generator, optional
        RNG; pass a seeded generator for reproducibility.

    Returns
    -------
    nx.Graph
        A simple graph drawn from the configuration model of the degree sequence.

    """
    rng = _as_rng(rng)
    degree_sequence = [d for _, d in graph.degree()]
    seed = int(rng.integers(0, 2**32 - 1))
    multigraph = nx.configuration_model(degree_sequence, seed=seed)
    simple = nx.Graph(multigraph)  # collapse parallel edges
    simple.remove_edges_from(nx.selfloop_edges(simple))
    return simple


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
