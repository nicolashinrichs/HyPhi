"""Density estimation utilities based on kernel density estimation (KDE)."""

# %% Import

import numpy as np
from KDEpy import FFTKDE, NaiveKDE, TreeKDE

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def select_kde(kernel_type="gaussian", bw="ISJ", norm=2, method="FFT"):
    """
    Create a KDE estimator (unfitted).

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
    NaiveKDE or TreeKDE or FFTKDE
        An unfitted KDE estimator object.

    Raises
    ------
    ValueError
        If ``bw`` is not one of the approved string values and not numeric,
        or if ``method`` is not one of 'FFT', 'naive', 'tree'.
    TypeError
        If ``norm`` is not an integer.

    """
    if not ((bw in ["scott", "silverman", "ISJ"]) or isinstance(bw, (int, float))):
        raise ValueError(f"BW {bw} not an approved type!")
    if not isinstance(norm, int):
        raise TypeError(f"norm must be an int, got {type(norm).__name__!r}")

    match method:
        case "naive":
            return NaiveKDE(kernel=kernel_type, bw=bw, norm=norm)
        case "tree":
            return TreeKDE(kernel=kernel_type, bw=bw, norm=norm)
        case "FFT":
            return FFTKDE(kernel=kernel_type, bw=bw, norm=norm)
        case _:
            raise ValueError(f"KDE method {method} not supported! Must be one of (FFT, naive, tree).")


def fit_kde(
    data: np.ndarray | list,
    kernel_type: str = "gaussian",
    bw: str | float | int = "ISJ",
    norm: int = 2,
    method: str = "FFT",
) -> NaiveKDE | TreeKDE | FFTKDE:
    """
    Fit a kernel density estimate to data.

    Parameters
    ----------
    data : np.ndarray or list
        Input data to fit the KDE on.
    kernel_type : str, optional
        Kernel function name. Default is 'gaussian'.
    bw : str or float or int, optional
        Bandwidth parameter ('scott', 'silverman', 'ISJ', or a number).
        Default is 'ISJ'.
    norm : int, optional
        Norm for the KDE. Default is 2.
    method : str, optional
        KDE method to use: one of 'FFT', 'naive', or 'tree'. Default is 'FFT'.

    Returns
    -------
    NaiveKDE or TreeKDE or FFTKDE
        A fitted KDE estimator object.

    """
    return select_kde(kernel_type, bw, norm, method).fit(data)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
