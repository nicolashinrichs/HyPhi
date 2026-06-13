"""
Connectivity for fNIRS haemoglobin time series.

fNIRS measures slow haemodynamics (HbO/HbR), not band-limited oscillations, so the phase-based
PLV used for EEG and MEG does not transfer. The modality-appropriate connectivity is
correlation based (or wavelet-transform coherence). This module provides windowed Pearson
correlation, producing the ``(windows, nodes, nodes)`` tensor that
:func:`hyphi.analyses.build_sliding_window_graphs` consumes.

Wavelet-transform coherence (WTC) is the other standard fNIRS connectivity choice; it needs a
continuous wavelet transform (an extra dependency) and is left as a follow-up. Which of the two
is canonical for fNIRS is a methodological choice for the maintainers.
"""

# %% Import
from __future__ import annotations

import numpy as np

__all__ = [
    "windowed_correlation",
]

_CHANNELS_TIME_NDIM = 2  # connectivity expects (n_channels, n_times)
_MIN_WIN_SIZE = 2  # Pearson correlation needs at least two samples per window


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def windowed_correlation(data: np.ndarray, win_size: int, win_stride: int) -> np.ndarray:
    """
    Windowed Pearson-correlation connectivity for a multichannel time series.

    The returned correlations are SIGNED, in [-1, 1]. Note that the downstream graph builder
    ``hyphi.analyses.build_sliding_window_graphs`` takes the absolute value of the connectivity
    (Forman curvature requires non-negative edge weights), so anti-correlation and correlation of
    equal magnitude become the same edge weight; how to treat the sign for fNIRS is an open
    methodological choice. This function also does NOT detrend or band-pass the input: systemic
    fNIRS confounds (Mayer waves, instrument drift, motion) must be removed upstream (haemodynamic
    band-pass, short-channel regression), or they will dominate the correlation.

    Parameters
    ----------
    data : np.ndarray
        Haemoglobin time series of shape ``(n_channels, n_times)``. Drop dead/saturated channels
        first: a zero-variance channel makes its correlation undefined and is rejected (see below).
    win_size : int
        Number of time steps per window (must be at least 2 and at most ``n_times``).
    win_stride : int
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Connectivity tensor of shape ``(n_windows, n_channels, n_channels)``: one symmetric
        correlation matrix per window, with unit diagonal and values in [-1, 1].

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, if ``win_size`` is outside ``[2, n_times]`` or ``win_stride`` is
        below 1, or if a channel has zero variance in some window (a dead/saturated optode), which
        would otherwise yield NaN correlations that silently collapse the downstream entropy series.

    """
    data = np.asarray(data, dtype=float)
    if data.ndim != _CHANNELS_TIME_NDIM:
        msg = f"Expected (n_channels, n_times) data, got shape {data.shape}"
        raise ValueError(msg)

    n_times = data.shape[1]
    if win_size < _MIN_WIN_SIZE:
        msg = f"win_size must be at least {_MIN_WIN_SIZE} to compute a correlation, got {win_size}."
        raise ValueError(msg)
    if win_size > n_times:
        msg = f"win_size ({win_size}) exceeds the series length ({n_times}); no window fits."
        raise ValueError(msg)
    if win_stride < 1:
        msg = f"win_stride must be at least 1, got {win_stride}."
        raise ValueError(msg)

    mats = []
    for start in range(0, n_times - win_size + 1, win_stride):
        window = data[:, start : start + win_size]
        # A dead or saturated optode is constant over the window. Detect it by ZERO peak-to-peak
        # range (exact and scale-invariant): a constant-NONZERO channel still carries roundoff
        # variance ~1e-31, so np.corrcoef returns finite (not NaN) values and would slip a NaN-only
        # check, then feed a spurious ~0-weight (near-disconnected) node into the graph. Reject
        # loudly so the bad channels are dropped first.
        flat = np.flatnonzero(np.ptp(window, axis=1) == 0)
        if flat.size:
            msg = (
                f"windowed_correlation: channel(s) {flat.tolist()} have zero variance (are constant) in "
                f"the window starting at sample {start} (a dead or saturated optode), so their correlation "
                "is undefined. Drop the bad channels before computing connectivity."
            )
            raise ValueError(msg)
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.corrcoef(window)
        if not np.isfinite(mat).all():
            # A non-constant channel can still yield non-finite correlations from inf/NaN in the input.
            msg = (
                f"windowed_correlation: the correlation in the window starting at sample {start} is "
                "non-finite; check the input for inf or NaN values."
            )
            raise ValueError(msg)
        mats.append(mat)
    return np.stack(mats, axis=0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
