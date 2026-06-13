"""
fMRI input: parcellate a preprocessed NIfTI into ROI time series.

HyPhi assumes fMRIPrep-style preprocessed input and delegates the heavy lifting (NIfTI reading,
masking, parcellation with standard atlases) to nilearn, which it orchestrates rather than
reimplements. The result is an ``(n_rois, n_times)`` ROI time series that feeds the windowed
connectivity in :mod:`hyphi.fmri.connectivity`.

nilearn is an optional dependency: it is imported lazily inside :func:`parcellate`, so importing
this module does not require it. Install the extra (``pip install 'hyphi[fmri]'``) to use the
parcellation path.
"""

# %% Import
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "parcellate",
]


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def parcellate(func_img: Any, atlas_img: Any, confounds: Any = None, **masker_kwargs: Any) -> np.ndarray:
    """
    Parcellate a preprocessed functional NIfTI into ROI time series.

    Wraps nilearn's ``NiftiLabelsMasker``: it applies the label atlas to the functional image,
    averages voxels within each ROI, and (optionally) regresses out confounds, returning one
    time series per ROI.

    Parameters
    ----------
    func_img : niimg-like
        A preprocessed 4D functional image (path or nibabel image).
    atlas_img : niimg-like
        A deterministic (label) atlas image, for example Schaefer or AAL.
    confounds : array-like or path, optional
        Confounds to regress out, forwarded to ``NiftiLabelsMasker.fit_transform``.
    **masker_kwargs
        Extra keyword arguments forwarded to ``NiftiLabelsMasker`` (for example
        ``standardize``, ``low_pass``, ``high_pass``, ``t_r``).

    Returns
    -------
    np.ndarray
        ROI time series of shape ``(n_rois, n_times)``, the orientation HyPhi's connectivity
        functions expect.

    Raises
    ------
    ImportError
        If nilearn is not installed.

    """
    try:
        from nilearn.maskers import NiftiLabelsMasker  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as exc:  # pragma: no cover - exercised only without nilearn installed
        msg = "fMRI parcellation requires nilearn. Install the extra with: pip install 'hyphi[fmri]'"
        raise ImportError(msg) from exc

    masker = NiftiLabelsMasker(labels_img=atlas_img, **masker_kwargs)
    # nilearn returns (n_times, n_rois); HyPhi connectivity wants (n_rois, n_times).
    roi_time_by_roi = masker.fit_transform(func_img, confounds=confounds)
    return np.asarray(roi_time_by_roi).T


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
