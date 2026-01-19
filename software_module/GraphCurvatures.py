# ============= #
# Preliminaries # 
# ============= # 

import networkx as nx
import numpy as np
import math
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

# ================ #
# Graph Curvatures #
# ================ #

def getFRC(G, method_val="1d"):
    # Initialize Forman-Ricci Curvature class
    frc = FormanRicci(G, method=method_val)
    # Compute the Forman-Ricci curvature
    frc.compute_ricci_curvature()
    # Return updated graph with curvature property on edges
    return frc.G


def getFRCVec(Gt, method_val="1d"):
    return list(map(lambda G: getFRC(G, method_val=method_val), Gt))


def getORC(G, alpha_val=0.5, base_val=math.e, power_val=0, method_val="OTDSinkhornMix"):
    # Initialize Forman-Ricci Curvature class
    orc = OllivierRicci(G, alpha=alpha_val, base=base_val, exp_power=power_val, method=method_val)
    # Compute the Forman-Ricci curvature
    orc.compute_ricci_curvature()
    # Return updated graph with curvature property on edges
    return orc.G


def getORCVec(Gt, alpha_val=0.5, base_val=math.e, power_val=0, method_val="OTDSinkhornMix"):
    return list(map(lambda G: getORC(G, alpha_val, base_val, power_val, method_val=method_val), Gt))


def extractCurvatures(G, curvature="formanCurvature"):
    return np.array([ddict[curvature] for u, v, ddict in G.edges(data=True)])


def extractCurvaturesVec(Gt, curvature="formanCurvature"):
    return list(map(lambda G: extractCurvatures(G, curvature=curvature), Gt))