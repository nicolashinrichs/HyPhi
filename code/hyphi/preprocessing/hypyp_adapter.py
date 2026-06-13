"""
Adapter from HyPyP / MNE preprocessed data to HyPhi phase arrays.

HyPhi is complementary to HyPyP and MNE: those tools clean and band-limit the recordings, and
this adapter turns their output into the input HyPhi's geometric pipeline expects. It takes a
preprocessed, band-limited signal (an MNE ``Raw`` or ``Epochs``, a HyPyP-preprocessed object
that exposes ``get_data``, or a plain array), extracts the instantaneous phase via the analytic
signal, and returns a ``(n_channels, n_times)`` phase array that feeds
``hyphi.modeling.windowing.sliding_window_plv``.

Phase is only meaningful for a narrowband signal, so the input should already be filtered to a
single band upstream by HyPyP or MNE; this adapter does not filter.
"""

# %% Import
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import hilbert

from hyphi.modeling.windowing import sliding_window_plv

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

__all__ = [
    "epochs_to_phase",
    "epochs_to_plv_graphs",
]

# An MNE Epochs array is (n_epochs, n_channels, n_times); Raw and 2D inputs are (n_channels, n_times).
_EPOCHED_NDIM = 3
_CHANNELS_TIME_NDIM = 2


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _to_data_array(inst: Any, picks: Any = None) -> np.ndarray:
    """Resolve an MNE/HyPyP object or an array to a NumPy data array."""
    if hasattr(inst, "get_data"):  # MNE Raw / Epochs, or a HyPyP wrapper exposing the same API
        data = inst.get_data(picks=picks) if picks is not None else inst.get_data()
        return np.asarray(data)
    return np.asarray(inst)


def _instantaneous_phase(data_2d: np.ndarray) -> np.ndarray:
    """Instantaneous phase (radians) of a ``(n_channels, n_times)`` signal via the analytic signal."""
    return np.angle(hilbert(data_2d, axis=-1))


def epochs_to_phase(inst: Any, picks: Any = None) -> np.ndarray:
    """
    Extract the instantaneous phase of a band-limited signal as a ``(n_channels, n_times)`` array.

    The input MUST already be band-limited to a single band upstream (by HyPyP/MNE): analytic-signal
    phase is only meaningful for a narrowband signal, and this adapter does not filter. A wideband
    input yields a meaningless phase (and spurious PLV downstream) with no error, so band-pass first.

    Parameters
    ----------
    inst : mne.Epochs, mne.io.Raw, object with ``get_data``, or np.ndarray
        A preprocessed, band-limited signal. An epoched input (3D) has its phase computed PER EPOCH
        (so the analytic signal is never Hilbert-transformed across an epoch join) and the per-epoch
        phase concatenated along time. NOTE: the concatenation still leaves a phase DISCONTINUITY at
        each join, so a sliding window placed across a join mixes two epochs. Use
        ``epochs_to_plv_graphs`` (which windows within each epoch) rather than windowing this flat
        array yourself when windows could straddle the epoch length.
    picks : optional
        Channel selection, forwarded to ``get_data`` for MNE/HyPyP objects; ignored for arrays.

    Returns
    -------
    np.ndarray
        Instantaneous phase in radians, shape ``(n_channels, n_times)``, values in [-pi, pi].

    """
    data = _to_data_array(inst, picks=picks)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim == _EPOCHED_NDIM:
        return np.concatenate([_instantaneous_phase(data[e]) for e in range(data.shape[0])], axis=-1)
    if data.ndim == _CHANNELS_TIME_NDIM:
        return _instantaneous_phase(data)
    msg = f"Expected 1D, 2D, or 3D data, got shape {data.shape}"
    raise ValueError(msg)


def epochs_to_plv_graphs(inst: Any, win_size: int, win_stride: int, picks: Any = None) -> list[nx.Graph]:
    """
    Run the full adapter path: preprocessed signal to a series of sliding-window PLV graphs.

    This is the one-call plug-and-play path for users who want "preprocessed data to HyPhi
    graphs" without handling the phase array themselves. For an epoched (3D) input the sliding
    window runs WITHIN each epoch and the per-epoch graph series are concatenated, so no PLV
    window ever straddles an epoch join (which would mix two epochs across a phase discontinuity
    and bias connectivity). The input must already be band-limited upstream (see
    ``epochs_to_phase``).

    Parameters
    ----------
    inst : mne.Epochs, mne.io.Raw, object with ``get_data``, or np.ndarray
        A preprocessed, band-limited signal.
    win_size : int
        Number of time steps per window.
    win_stride : int
        Stride between consecutive windows.
    picks : optional
        Channel selection forwarded to the adapter.

    Returns
    -------
    list[nx.Graph]
        One weighted PLV graph per window (per-epoch windows concatenated for an epoched input).

    """
    data = _to_data_array(inst, picks=picks)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    # sliding_window_plv infers orientation by assuming channels <= times and silently transposes
    # otherwise; a window/epoch with more channels than time samples would be transposed (turning
    # time samples into graph nodes). Reject it loudly rather than return a corrupted graph.
    if data.ndim >= _CHANNELS_TIME_NDIM and data.shape[-2] > data.shape[-1]:
        msg = (
            f"more channels ({data.shape[-2]}) than time samples ({data.shape[-1]}) per epoch/window; "
            "sliding_window_plv cannot disambiguate the orientation. Provide more time samples than channels."
        )
        raise ValueError(msg)
    if data.ndim == _CHANNELS_TIME_NDIM:
        return sliding_window_plv(_instantaneous_phase(data), win_size, win_stride)
    if data.ndim == _EPOCHED_NDIM:
        if win_size > data.shape[-1]:
            logger.warning(
                "epochs_to_plv_graphs: win_size=%d exceeds the epoch length %d, so no window fits "
                "within an epoch and no graphs are produced. Reduce win_size, or pass continuous "
                "(2-D) data if you intend to window across the recording.",
                win_size,
                data.shape[-1],
            )
        graphs: list[nx.Graph] = []
        for e in range(data.shape[0]):
            graphs.extend(sliding_window_plv(_instantaneous_phase(data[e]), win_size, win_stride))
        return graphs
    msg = f"Expected 1D, 2D, or 3D data, got shape {data.shape}"
    raise ValueError(msg)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
