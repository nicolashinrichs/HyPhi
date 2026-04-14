"""
Null models module for HyPhi: Surrogate data generation for statistical testing.

Years: 2026
"""

# %% Import
from __future__ import annotations

import numpy as np

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ---------------------
# Signal-level Surrogates
# ---------------------


def phase_randomize(signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Phase randomization (amplitude-preserving surrogate).

    Randomises the Fourier phases of a signal while preserving its
    power spectrum.  Works on each channel independently.

    *(preserves power spectrum, destroys phase relationships)*

    Parameters
    ----------
    signal : np.ndarray
        Signal of shape ``(n_channels, T)`` or ``(T,)``.
    rng : np.random.Generator, optional
        Random number generator.  Defaults to ``np.random.default_rng()``.

    Returns
    -------
    np.ndarray
        Surrogate signal with same shape and power spectrum as input.

    """
    if rng is None:
        rng = np.random.default_rng()

    one_d = signal.ndim == 1
    if one_d:
        signal = signal[np.newaxis, :]

    n_channels, T = signal.shape
    surrogates = np.zeros_like(signal)

    for ch in range(n_channels):
        fft_vals = np.fft.rfft(signal[ch])
        amplitudes = np.abs(fft_vals)
        random_phases = rng.uniform(0, 2 * np.pi, size=len(fft_vals))
        # Preserve DC and Nyquist (if T is even)
        random_phases[0] = 0
        if T % 2 == 0:
            random_phases[-1] = 0
        surrogates[ch] = np.fft.irfft(amplitudes * np.exp(1j * random_phases), n=T)

    return surrogates[0] if one_d else surrogates


def circular_time_shift(signal: np.ndarray, min_shift: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Circular time-shift surrogate.

    Applies a random circular shift to each channel independently,
    destroying temporal alignment while preserving autocorrelation.

    Parameters
    ----------
    signal : np.ndarray
        Signal of shape ``(n_channels, T)`` or ``(T,)``.
    min_shift : int
        Minimum absolute shift (avoids trivially small shifts).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Circularly shifted surrogate.

    """
    if rng is None:
        rng = np.random.default_rng()

    one_d = signal.ndim == 1
    if one_d:
        signal = signal[np.newaxis, :]

    n_channels, T = signal.shape
    surrogates = np.zeros_like(signal)

    for ch in range(n_channels):
        shift = rng.integers(min_shift, T - min_shift)
        surrogates[ch] = np.roll(signal[ch], shift)

    return surrogates[0] if one_d else surrogates


# ---------------------
# Dyad-level Null Models
# ---------------------


def dyad_subject_swap(data_matrix: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate pseudo-dyads by shuffling subjects across real dyads.

    Swaps the second subject of each dyad with the second subject of
    a randomly chosen other dyad, breaking real dyad pairings while
    keeping each subject's data intact.

    Parameters
    ----------
    data_matrix : np.ndarray
        Array of shape ``(n_dyads, 2, n_timepoints)``.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Shuffled data matrix with same shape.

    """
    if rng is None:
        rng = np.random.default_rng()

    n_dyads = data_matrix.shape[0]
    shuffled = data_matrix.copy()

    for i in range(n_dyads):
        swap_idx = rng.integers(0, n_dyads)
        if swap_idx != i:
            temp = shuffled[i, 1].copy()
            shuffled[i, 1] = shuffled[swap_idx, 1]
            shuffled[swap_idx, 1] = temp

    return shuffled


def dyad_label_shuffle(
    entropy_series: np.ndarray, dyad_labels: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Dyad-label shuffling control.

    Shuffles dyad assignment labels while keeping the entropy time series
    intact, breaking the dyad-specific structure.

    Parameters
    ----------
    entropy_series : np.ndarray # TODO: kwarg not used
        Entropy values of shape ``(n_observations,)`` or ``(n_observations, n_features)``.
        Not modified; provided for context on the observation count.
    dyad_labels : np.ndarray
        Dyad identifiers of shape ``(n_observations,)``.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Shuffled dyad labels (same shape as *dyad_labels*).

    """
    if rng is None:
        rng = np.random.default_rng()

    unique_dyads = np.unique(dyad_labels)
    permuted_dyads = rng.permutation(unique_dyads)
    mapping = dict(zip(unique_dyads, permuted_dyads))
    return np.array([mapping[d] for d in dyad_labels])


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
