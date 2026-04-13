# ==========================
# Density Estimation
# ==========================
"""
Moved from software_module/DensityEstimation.py.
"""

import numpy as np
from KDEpy import NaiveKDE, TreeKDE, FFTKDE


def select_kde(kernel_type="gaussian", bw="ISJ", norm=2, method="FFT"):
    """Create a KDE estimator (unfitted).

    Parameters
    ----------
    kernel_type : str
        Kernel function name.
    bw : str or float
        Bandwidth parameter ('scott', 'silverman', 'ISJ', or a number).
    norm : int
        Norm for the KDE.
    method : str
        One of 'naive', 'tree', 'FFT'.

    Returns
    -------
    KDE estimator object (unfitted).
    """
    assert (bw in ["scott", "silverman", "ISJ"]) or isinstance(bw, (int, float)), \
        f"BW {bw} not an approved type!"
    assert isinstance(norm, int)

    match method:
        case "naive":
            return NaiveKDE(kernel=kernel_type, bw=bw, norm=norm)
        case "tree":
            return TreeKDE(kernel=kernel_type, bw=bw, norm=norm)
        case "FFT":
            return FFTKDE(kernel=kernel_type, bw=bw, norm=norm)
        case _:
            raise ValueError(
                f"KDE method {method} not supported! Must be one of (FFT, naive, tree)."
            )


def fit_kde(data, kernel_type="gaussian", bw="ISJ", norm=2, method="FFT"):
    """Fit a KDE to data and return the fitted estimator."""
    return select_kde(kernel_type, bw, norm, method).fit(data)
