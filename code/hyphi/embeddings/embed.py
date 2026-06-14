"""
Hyperbolic embedding of multichannel signals.

Embeds a multichannel signal (for example simulated Kuramoto phases or Watts-Strogatz-derived
activity) into the Poincare ball: it builds a dissimilarity matrix between channels, finds a
low-dimensional Euclidean configuration with metric MDS, then maps that configuration into the
negatively curved Poincare ball through the exponential map at the origin.

This is the lightweight, dependency-free embedding for simulated data. The full HEEGNet model
(Li et al. 2026), with its two-stage DSMDBN moment alignment and Hyperbolic Horospherical
Sliced-Wasserstein matching, is a deep network that needs a learning framework and is out of
scope here; how far to take that integration (a replacement for the connectivity-to-graph step,
a parallel alternative geometry, or a validation layer) is an open design decision.
"""

# %% Import
from __future__ import annotations

import numpy as np
from sklearn.manifold import MDS

from .hyperbolic import poincare_exp0

__all__ = [
    "hyperbolic_embedding",
]

_SIGNAL_NDIM = 2  # (n_channels, n_times)
_MIN_CHANNELS = 2  # a dissimilarity / MDS embedding needs at least two channels


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def hyperbolic_embedding(
    data: np.ndarray,
    n_components: int = 2,
    metric: str = "correlation",
    random_state: int = 0,
) -> np.ndarray:
    """
    Embed channels of a multichannel signal into the Poincare ball.

    Parameters
    ----------
    data : np.ndarray
        Signal of shape ``(n_channels, n_times)``.
    n_components : int
        Dimension of the embedding (2 for the Poincare disk).
    metric : {"correlation", "euclidean"}
        Channel dissimilarity. ``"correlation"`` uses ``1 - |corr|``, so the magnitude of coupling
        sets closeness (a strongly anti-correlated pair is treated as close, matching the
        absolute-value convention the connectivity layer uses); ``"euclidean"`` runs MDS on the raw
        channel vectors and is therefore sensitive to per-channel amplitude (standardize the signal
        first if you want a scale-invariant embedding).
    random_state : int
        Seed for the MDS initialisation, for reproducibility (a fixed seed gives a fixed embedding
        for a given scikit-learn version).

    Returns
    -------
    np.ndarray
        Embedding of shape ``(n_channels, n_components)``, with every point inside the open
        Poincare ball.

    Raises
    ------
    ValueError
        If ``data`` is not 2D, has fewer than two channels, or contains inf/NaN; if ``n_components``
        is not in ``[1, n_channels]``; if ``metric`` is unknown; or (for ``"correlation"``) if a
        channel is constant (zero variance), whose correlation is undefined.

    """
    data = np.asarray(data, dtype=float)
    if data.ndim != _SIGNAL_NDIM:
        msg = f"data must be (n_channels, n_times), got shape {data.shape}."
        raise ValueError(msg)
    n_channels = data.shape[0]
    if n_channels < _MIN_CHANNELS:
        msg = f"need at least {_MIN_CHANNELS} channels to embed, got {n_channels}."
        raise ValueError(msg)
    if not 1 <= n_components <= n_channels:
        msg = f"n_components must be in [1, n_channels={n_channels}], got {n_components}."
        raise ValueError(msg)
    if not np.isfinite(data).all():
        msg = "data contains inf or NaN; clean or drop those samples/channels first."
        raise ValueError(msg)

    if metric == "correlation":
        # A constant (dead/flat) channel has zero variance, so its correlation is undefined (NaN);
        # reject it loudly rather than let the NaN reach MDS as a cryptic "Input X contains NaN".
        flat = np.flatnonzero(np.ptp(data, axis=1) == 0)
        if flat.size:
            msg = (
                f"channel(s) {flat.tolist()} are constant (zero variance), so their correlation is "
                "undefined; drop the dead channels first."
            )
            raise ValueError(msg)
        dissimilarity = 1.0 - np.abs(np.corrcoef(data))
        np.fill_diagonal(dissimilarity, 0.0)
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress="auto",
        )
        euclidean = mds.fit_transform(dissimilarity)
    elif metric == "euclidean":
        mds = MDS(n_components=n_components, random_state=random_state, normalized_stress="auto")
        euclidean = mds.fit_transform(data)
    else:
        msg = f"Unknown metric {metric!r}; choose 'correlation' or 'euclidean'"
        raise ValueError(msg)
    return poincare_exp0(euclidean)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
