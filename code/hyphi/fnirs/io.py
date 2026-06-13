"""
fNIRS input and haemoglobin conversion, built on MNE core.

Reads a SNIRF recording and converts raw continuous-wave amplitude to haemoglobin
concentration (HbO/HbR) via optical density and the modified Beer-Lambert law, with optional
temporal-derivative-distribution-repair (TDDR) motion correction. All of this is provided by
MNE itself, so HyPhi orchestrates MNE rather than reimplementing the optics. The result feeds
the windowed connectivity in :mod:`hyphi.fnirs.connectivity`.
"""

# %% Import
from __future__ import annotations

from typing import Any

import numpy as np

# mne is imported lazily inside the functions below: it is a heavy import (~seconds) and only the
# fNIRS path needs it, so importing this module (and hence ``import hyphi``) stays fast and mne-free.

__all__ = [
    "haemoglobin_data",
    "load_snirf",
    "to_haemoglobin",
]

_DEFAULT_PPF = 6.0  # partial pathlength factor for the modified Beer-Lambert law


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def to_haemoglobin(raw: Any, ppf: float = _DEFAULT_PPF, tddr: bool = True) -> Any:
    """
    Convert a raw continuous-wave fNIRS recording to haemoglobin concentration.

    Applies optical density, optional TDDR motion correction, and the modified Beer-Lambert
    law. Operates on an in-memory MNE ``Raw`` so it can be used without a file on disk.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw fNIRS recording with ``fnirs_cw_amplitude`` channels.
    ppf : float
        Partial pathlength factor for the Beer-Lambert law.
    tddr : bool
        If True, apply temporal-derivative-distribution-repair motion correction.

    Returns
    -------
    mne.io.Raw
        A recording with HbO/HbR channels.

    """
    from mne.preprocessing.nirs import (  # noqa: PLC0415 (lazy: keep mne out of import hyphi)
        beer_lambert_law,
        optical_density,
        temporal_derivative_distribution_repair,
    )

    od = optical_density(raw)
    if tddr:
        od = temporal_derivative_distribution_repair(od)
    return beer_lambert_law(od, ppf=ppf)


def load_snirf(path: Any, ppf: float = _DEFAULT_PPF, tddr: bool = True) -> Any:
    """
    Load a SNIRF file and convert it to haemoglobin concentration.

    Parameters
    ----------
    path : str or path-like
        Path to a ``.snirf`` recording.
    ppf : float
        Partial pathlength factor for the Beer-Lambert law.
    tddr : bool
        If True, apply TDDR motion correction.

    Returns
    -------
    mne.io.Raw
        A recording with HbO/HbR channels.

    """
    from mne.io import read_raw_snirf  # noqa: PLC0415 (lazy: keep mne out of import hyphi)

    raw = read_raw_snirf(path, preload=True)
    return to_haemoglobin(raw, ppf=ppf, tddr=tddr)


def haemoglobin_data(raw: Any, chroma: str = "hbo") -> np.ndarray:
    """
    Extract one chromophore's channels as a ``(n_channels, n_times)`` array.

    Parameters
    ----------
    raw : mne.io.Raw
        A haemoglobin recording (output of :func:`to_haemoglobin`).
    chroma : {"hbo", "hbr"}
        Which chromophore to extract.

    Returns
    -------
    np.ndarray
        The selected channels, shape ``(n_channels, n_times)``.

    """
    return np.asarray(raw.copy().pick(chroma).get_data())


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
