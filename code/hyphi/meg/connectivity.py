"""
Connectivity for MEG.

MEG records fast oscillations, so the phase-locking-value path used for EEG transfers directly:
use the shared adapter (:func:`hyphi.preprocessing.epochs_to_plv_graphs`) to go from a
band-limited MEG recording to PLV graphs. MEG sensors are also prone to field spread (volume
conduction), so this module additionally provides the weighted phase-lag index (wPLI), which is
insensitive to zero-lag (instantaneous, field-spread) coupling.

Both ``meg_to_plv_graphs`` and ``windowed_wpli`` produce inputs to HyPhi's graph pipeline: the
first returns graphs directly; the second returns a ``(windows, nodes, nodes)`` tensor for
:func:`hyphi.analyses.build_sliding_window_graphs`.
"""

# %% Import
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import hilbert

from hyphi.preprocessing import epochs_to_plv_graphs

if TYPE_CHECKING:
    import networkx as nx

__all__ = [
    "meg_to_plv_graphs",
    "windowed_wpli",
]

_CHANNELS_TIME_NDIM = 2  # connectivity expects (n_channels, n_times)
_MIN_WIN_SIZE = 2  # wPLI needs at least two samples per window


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def meg_to_plv_graphs(inst: Any, win_size: int, win_stride: int, picks: Any = None) -> list[nx.Graph]:
    """
    Sensor-space PLV graph series from a band-limited MEG recording.

    Thin wrapper over the shared phase adapter, so MEG uses the same phase-to-PLV path as EEG.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs, object with ``get_data``, or np.ndarray
        A band-limited MEG recording.
    win_size, win_stride : int
        Sliding-window size and stride.
    picks : optional
        Channel selection forwarded to the adapter (for example ``"mag"`` or ``"grad"``).

    Returns
    -------
    list[nx.Graph]
        One PLV graph per window.

    """
    return epochs_to_plv_graphs(inst, win_size, win_stride, picks=picks)


def windowed_wpli(signal: np.ndarray, win_size: int, win_stride: int) -> np.ndarray:
    """
    Windowed weighted phase-lag index (wPLI) connectivity.

    wPLI weights phase-lag contributions by the magnitude of the imaginary cross-spectrum, so it
    is insensitive to zero-lag coupling (volume conduction / field spread). For each window,
    ``wPLI_ij = |E[Im(X_i conj(X_j))]| / E[|Im(X_i conj(X_j))|]`` from the analytic signal, with a
    zero diagonal (the imaginary auto-spectrum is zero).

    Parameters
    ----------
    signal : np.ndarray
        Band-limited signal of shape ``(n_channels, n_times)``.
    win_size, win_stride : int
        Sliding-window size and stride.

    Returns
    -------
    np.ndarray
        Connectivity tensor ``(n_windows, n_channels, n_channels)`` with values in [0, 1].

    """
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != _CHANNELS_TIME_NDIM:
        msg = f"Expected (n_channels, n_times) data, got shape {arr.shape}"
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = "MEG signal contains inf or NaN values; clean or drop them first."
        raise ValueError(msg)
    n_times = arr.shape[1]
    if win_size < _MIN_WIN_SIZE:
        msg = f"win_size must be at least {_MIN_WIN_SIZE}, got {win_size}."
        raise ValueError(msg)
    if win_size > n_times:
        msg = f"win_size ({win_size}) exceeds the series length ({n_times}); no window fits."
        raise ValueError(msg)
    if win_stride < 1:
        msg = f"win_stride must be at least 1, got {win_stride}."
        raise ValueError(msg)

    analytic = hilbert(arr, axis=-1)
    mats = []
    for start in range(0, n_times - win_size + 1, win_stride):
        # A constant (dead/saturated) sensor has no phase; its constant factor cancels in the wPLI
        # ratio, producing SPURIOUS (not zero) connectivity, so reject it loudly (zero peak-to-peak).
        real_window = arr[:, start : start + win_size]
        flat = np.flatnonzero(np.ptp(real_window, axis=1) == 0)
        if flat.size:
            msg = (
                f"windowed_wpli: channel(s) {flat.tolist()} are constant in the window starting at sample "
                f"{start} (a dead or saturated sensor), so their phase is undefined. Drop the bad channels first."
            )
            raise ValueError(msg)
        window = analytic[:, start : start + win_size]
        cross_imag = np.imag(window[:, None, :] * np.conj(window[None, :, :]))
        numerator = np.abs(np.mean(cross_imag, axis=-1))
        denominator = np.mean(np.abs(cross_imag), axis=-1)
        wpli = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        # The input is checked finite above, but the cross-spectrum product squares the magnitude, so a
        # finite-but-extreme amplitude can overflow float64 to inf -> NaN here (and the divide guard does
        # not clean a NaN numerator). Reject loudly rather than emit a silent NaN that zeroes out downstream.
        if not np.isfinite(wpli).all():
            msg = (
                f"windowed_wpli: non-finite wPLI in the window starting at sample {start}; the cross-spectrum "
                "overflowed float64. Rescale the signal to physical units (MEG is ~1e-13 T) before computing it."
            )
            raise ValueError(msg)
        np.fill_diagonal(wpli, 0.0)
        mats.append(wpli)
    return np.stack(mats, axis=0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
