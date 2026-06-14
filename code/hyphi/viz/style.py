"""
Shared visual style for HyPhi figures.

The single source of truth for how curvature is coloured. Curvature is signed and centred at 0,
so it is shown with a diverging colormap and a normalisation that is symmetric about 0 with limits
taken from the data. This avoids the bug of a fixed ``vmin=-1, vmax=1`` range, which silently
clips Forman or augmented Forman curvature on weighted graphs (whose values can exceed 1 in
magnitude) and so misleads interpretation.
"""

# %% Import
from __future__ import annotations

import numpy as np
from matplotlib.colors import TwoSlopeNorm

__all__ = [
    "CURVATURE_CMAP",
    "curvature_norm",
]

# Diverging colormap: negative curvature one side, positive the other, neutral at 0.
CURVATURE_CMAP = "coolwarm"


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def curvature_norm(values: np.ndarray) -> TwoSlopeNorm:
    """
    Build a diverging normalisation centred on 0 with symmetric limits from the data.

    Parameters
    ----------
    values : array-like
        Curvature values to be coloured.

    Returns
    -------
    matplotlib.colors.TwoSlopeNorm
        Centred at 0 with ``vmin = -m`` and ``vmax = m`` where ``m = max(|finite values|)``, so
        no value is clipped. Falls back to a unit range for empty or all-zero input.

    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    magnitude = float(np.max(np.abs(finite))) if finite.size else 1.0
    if magnitude == 0.0:
        magnitude = 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-magnitude, vmax=magnitude)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
