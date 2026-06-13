"""
Preprocessing interoperability: adapters from HyPyP / MNE preprocessed data into HyPhi.

HyPhi does not implement its own preprocessing (filtering, ICA, artifact rejection); those are
solved by HyPyP and MNE. This subpackage provides the thin adapter that turns their cleaned,
band-limited output into the phase arrays HyPhi's graph pipeline consumes.
"""

# %% Imports
from .hypyp_adapter import epochs_to_phase, epochs_to_plv_graphs

__all__ = [
    "epochs_to_phase",
    "epochs_to_plv_graphs",
]
