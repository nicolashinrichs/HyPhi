"""
fMRI hyperscanning support: NIfTI parcellation and ROI connectivity.

This subpackage turns a preprocessed functional NIfTI into the ``(windows, ROIs, ROIs)``
connectivity tensor that HyPhi's graph pipeline consumes. Parcellation (NIfTI reading, masking,
standard atlases) is delegated to nilearn, an optional dependency imported lazily inside
``parcellate``; the correlation and partial-correlation connectivity use only NumPy. Importing
this subpackage (and ``import hyphi``) does not require nilearn; it is needed only to parcellate.
"""

# %% Imports
from .connectivity import roi_correlation, roi_partial_correlation
from .io import parcellate

__all__ = [
    "parcellate",
    "roi_correlation",
    "roi_partial_correlation",
]
