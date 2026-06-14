"""
MEG hyperscanning support: FIF input and phase-based connectivity.

MEG records fast oscillations like EEG, so the phase-to-PLV path transfers directly through the
shared preprocessing adapter; this subpackage adds FIF input (via MNE) and the volume-conduction
robust weighted phase-lag index. Source-space node definitions are supported through MNE's
inverse pipeline (feed source-estimate time courses to the connectivity functions like sensor
channels). MNE is imported lazily inside ``load_fif``, so importing this subpackage (and
``import hyphi``) stays fast; MNE loads only when a FIF file is read.
"""

# %% Imports
from .connectivity import meg_to_plv_graphs, windowed_wpli
from .io import load_fif

__all__ = [
    "load_fif",
    "meg_to_plv_graphs",
    "windowed_wpli",
]
