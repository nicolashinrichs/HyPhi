# =====================================
# Null Model Generators
# =====================================
"""
Utility functions for generating surrogate / null-model data
to test the specificity of curvature-entropy findings.
"""

import numpy as np
from typing import Optional


def phase_randomize(signal, rng=None):
    """Phase randomization (amplitude-preserving surrogate).

    Randomises the Fourier phases of a signal while preserving its
    power spectrum. Works on each channel independently.

    Parameters
    ----------
    signal : np.ndarray
        Signal of shape (n_channels, T) or (T,).
    rng : np.random.Generator, optional
        Random number generator. Defaults to np.random.default_rng().

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


def circular_time_shift(signal, rng=None):
    """Circular time-shift surrogate.

    Applies a random circular shift to each channel independently,
    destroying temporal alignment while preserving autocorrelation.

    Parameters
    ----------
    signal : np.ndarray
        Signal of shape (n_channels, T) or (T,).
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
        shift = rng.integers(1, T)  # avoid shift=0
        surrogates[ch] = np.roll(signal[ch], shift)

    return surrogates[0] if one_d else surrogates


def dyad_shuffle(entropy_series, dyad_labels, rng=None):
    """Dyad-shuffling control.

    Shuffles dyad assignment labels while keeping the entropy time series
    intact, breaking the dyad-specific structure.

    Parameters
    ----------
    entropy_series : np.ndarray
        Entropy values of shape (n_observations,) or (n_observations, n_features).
    dyad_labels : np.ndarray
        Dyad identifiers of shape (n_observations,).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Shuffled dyad labels (same shape as dyad_labels).
    """
    if rng is None:
        rng = np.random.default_rng()

    shuffled = dyad_labels.copy()
    # Shuffle unique dyad labels and re-map
    unique_dyads = np.unique(shuffled)
    permuted_dyads = rng.permutation(unique_dyads)
    mapping = dict(zip(unique_dyads, permuted_dyads))
    return np.array([mapping[d] for d in shuffled])
