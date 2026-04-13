"""TODO: add description here"""

# %% Import
import numpy as np
import math
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_frc(G, method_val: str = "1d"):
    """Get Forman-Ricci curvature."""
    # Initialize Forman-Ricci Curvature class
    frc = FormanRicci(G, method=method_val)
    # Compute the Forman-Ricci curvature
    frc.compute_ricci_curvature()
    # Return updated graph with curvature property on edges
    return frc.G


def get_frc_vec(Gt, method_val: str = "1d"):
    """Get Forman-Ricci curvatures as a vector."""
    return list(map(lambda G: get_frc(G, method_val=method_val), Gt))


def get_orc(G, alpha_val=0.5, base_val=math.e, power_val=0, method_val: str = "OTDSinkhornMix"):
    """Get Ollivier-Ricci curvature."""
    # Initialize Ollivier-Ricci Curvature class
    orc = OllivierRicci(G, alpha=alpha_val, base=base_val, exp_power=power_val, method=method_val)
    # Compute the Ollivier-Ricci curvature
    orc.compute_ricci_curvature()
    # Return updated graph with curvature property on edges
    return orc.G


def get_orc_vec(Gt, alpha_val=0.5, base_val=math.e, power_val=0, method_val: str = "OTDSinkhornMix"):
    """Get Ollivier-Ricci curvatures as a vector."""
    return list(map(lambda G: get_orc(G, alpha_val, base_val, power_val, method_val=method_val), Gt))


def extract_curvatures(G, curvature: str = "formanCurvature"):
    """Extract curvatures from a graph."""
    return np.array([ddict[curvature] for u, v, ddict in G.edges(data=True)])


def extract_curvatures_vec(Gt, curvature: str = "formanCurvature"):
    """Extract curvatures from a graph as a vector."""
    return list(map(lambda G: extract_curvatures(G, curvature=curvature), Gt))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
