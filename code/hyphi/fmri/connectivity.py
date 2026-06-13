"""
Connectivity for fMRI ROI time series.

fMRI BOLD is very slow and is analysed on parcellated ROI time series, so connectivity is
correlation based, not phase based. This module provides windowed Pearson correlation (the
simple default) and windowed partial correlation (which controls for the other ROIs via the
precision matrix), each producing the ``(windows, ROIs, ROIs)`` tensor that
:func:`hyphi.analyses.build_sliding_window_graphs` consumes.

Which of the two is canonical for a given fMRI analysis (Pearson is simpler and more stable;
partial correlation is sparser and more specific) is a methodological choice for the maintainers.
"""

# %% Import
from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "roi_correlation",
    "roi_partial_correlation",
]

_ROI_TIME_NDIM = 2  # connectivity expects (n_rois, n_times)
_MIN_WIN_SIZE = 2  # correlation needs at least two samples per window
_PCORR_TOL = 1e-6  # roundoff slack on the |partial correlation| <= 1 bound


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _check_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != _ROI_TIME_NDIM:
        msg = f"Expected (n_rois, n_times) data, got shape {arr.shape}"
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = "ROI time series contains inf or NaN values; clean or drop them first."
        raise ValueError(msg)
    return arr


def _windows(n_times: int, win_size: int, win_stride: int) -> range:
    if win_size < _MIN_WIN_SIZE:
        msg = f"win_size must be at least {_MIN_WIN_SIZE} to compute a correlation, got {win_size}."
        raise ValueError(msg)
    if win_size > n_times:
        msg = f"win_size ({win_size}) exceeds the series length ({n_times}); no window fits."
        raise ValueError(msg)
    if win_stride < 1:
        msg = f"win_stride must be at least 1, got {win_stride}."
        raise ValueError(msg)
    return range(0, n_times - win_size + 1, win_stride)


def _reject_constant_rois(window: np.ndarray, start: int) -> None:
    """
    Reject an exactly-constant (dead) ROI as undefined connectivity.

    A constant ROI has zero variance, so its correlation is undefined (NaN). Detecting it by zero
    peak-to-peak range fails early and names the offending ROI and window (rather than waiting for a
    downstream NaN). This catches an EXACTLY-constant ROI at any value; a NEAR-constant ROI (tiny
    jitter) passes here and is caught downstream (roi_partial_correlation rejects the resulting
    out-of-range partial correlation; Pearson stays bounded in [-1, 1]).
    """
    flat = np.flatnonzero(np.ptp(window, axis=1) == 0)
    if flat.size:
        msg = (
            f"ROI(s) {flat.tolist()} have zero variance (are constant) in the window starting at sample "
            f"{start}, so their correlation is undefined. Drop the bad ROIs before computing connectivity."
        )
        raise ValueError(msg)


def roi_correlation(roi_timeseries: np.ndarray, win_size: int, win_stride: int) -> np.ndarray:
    """
    Windowed Pearson-correlation connectivity for ROI time series.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        ROI time series of shape ``(n_rois, n_times)``.
    win_size : int
        Number of time steps per window.
    win_stride : int
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Connectivity tensor ``(n_windows, n_rois, n_rois)``: one symmetric correlation matrix
        per window, unit diagonal, values in [-1, 1].

    Raises
    ------
    ValueError
        If ``roi_timeseries`` is not 2-D, if ``win_size`` is outside ``[2, n_times]`` or
        ``win_stride`` is below 1, or if a ROI is constant in some window (a dead ROI).

    Notes
    -----
    The returned correlations are SIGNED, but the downstream graph builder
    ``hyphi.analyses.build_sliding_window_graphs`` takes their absolute value (Forman curvature
    requires non-negative edge weights), so anti-correlation and correlation of equal magnitude
    become the same edge weight. This function also does not regress motion/global-signal or
    band-pass; those fMRI confounds must be removed upstream (fMRIPrep-style) or they dominate the
    correlation.

    """
    data = _check_2d(roi_timeseries)
    mats = []
    for s in _windows(data.shape[1], win_size, win_stride):
        window = data[:, s : s + win_size]
        _reject_constant_rois(window, s)
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.corrcoef(window)
        if not np.isfinite(mat).all():
            msg = (
                f"roi_correlation: the correlation in the window starting at sample {s} is non-finite; "
                "check the input for inf or NaN values."
            )
            raise ValueError(msg)
        mats.append(mat)
    return np.stack(mats, axis=0)


def roi_partial_correlation(roi_timeseries: np.ndarray, win_size: int, win_stride: int) -> np.ndarray:
    """
    Windowed partial-correlation connectivity for ROI time series.

    Partial correlation between two ROIs controls for all the others, computed from the
    precision (inverse covariance) matrix: ``pcorr_ij = -P_ij / sqrt(P_ii * P_jj)``. The
    pseudo-inverse is used for numerical robustness on short or collinear windows.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        ROI time series of shape ``(n_rois, n_times)``.
    win_size : int
        Number of time steps per window.
    win_stride : int
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Connectivity tensor ``(n_windows, n_rois, n_rois)``: one symmetric partial-correlation
        matrix per window, unit diagonal.

    Raises
    ------
    ValueError
        If ``roi_timeseries`` is not 2-D, if ``win_size`` is outside ``[2, n_times]`` or
        ``win_stride`` is below 1, if a ROI is constant in some window, or if a degenerate window
        yields non-finite partial correlations.

    Warns
    -----
    UserWarning
        If ``win_size <= n_rois``: the per-window covariance is then rank-deficient, so the
        pseudo-inverse avoids a crash but the partial correlations are statistically unreliable.

    Notes
    -----
    As with :func:`roi_correlation`, the sign is discarded downstream by
    ``build_sliding_window_graphs`` (it takes the absolute value). For a reliable estimate use a
    window much longer than the ROI count (``win_size >> n_rois``).

    """
    data = _check_2d(roi_timeseries)
    n_rois = data.shape[0]
    starts = _windows(data.shape[1], win_size, win_stride)  # validates win_size/win_stride first
    if win_size <= n_rois:
        warnings.warn(
            f"roi_partial_correlation: win_size ({win_size}) is not greater than the number of ROIs "
            f"({n_rois}), so the per-window covariance is rank-deficient and the partial correlations are "
            "statistically unreliable (the pseudo-inverse avoids a crash but does not fix this). Use a "
            "window longer than the ROI count.",
            stacklevel=2,
        )
    mats = []
    for s in starts:
        window = data[:, s : s + win_size]
        _reject_constant_rois(window, s)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.linalg.pinv(np.cov(window))
            scale = np.sqrt(np.diag(precision))
            pcorr = -precision / np.outer(scale, scale)
        np.fill_diagonal(pcorr, 1.0)
        if not np.isfinite(pcorr).all():
            msg = (
                f"roi_partial_correlation: non-finite partial correlation in the window starting at sample "
                f"{s} (a degenerate or rank-deficient window); use a longer window or fewer ROIs."
            )
            raise ValueError(msg)
        peak = float(np.abs(pcorr).max())
        if peak > 1.0 + _PCORR_TOL:
            # Partial correlation is bounded in [-1, 1]; |pcorr| > 1 means the per-window covariance
            # was near-singular (a near-constant ROI that slips the exactly-constant guard, or
            # near-collinear ROIs), so the pinv estimate is invalid. Reject rather than feed an
            # out-of-range edge weight into the curvature pipeline.
            msg = (
                f"roi_partial_correlation: partial correlation {peak:.3g} exceeds [-1, 1] in the window "
                f"starting at sample {s}; the window is degenerate (a near-constant or near-collinear ROI "
                "makes the covariance near-singular). Drop the bad ROI or use a longer window."
            )
            raise ValueError(msg)
        mats.append(pcorr)
    return np.stack(mats, axis=0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
