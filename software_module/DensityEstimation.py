# ============= #
# Preliminaries # 
# ============= # 

import numpy as np
from KDEpy import NaiveKDE, TreeKDE, FFTKDE
from scipy.optimize import minimize, LinearConstraint
# from pywde.simple_estimator import SimpleWaveletDensityEstimator
# from pywde.spwde import SPWDE
# from pywde.square_root_estimator import WaveletDensityEstimator
# from pywde.log_estimator import WaveletDensityEstimator as LogWDE
# from scipy import integrate


def selectKDE(kernel_type: str = "gaussian", bw: str | float | int = "ISJ", norm: int = 2, method: str = "FFT") -> NaiveKDE | TreeKDE | FFTKDE:
    assert (bw in ["scott", "silverman", "ISJ"]) | (isinstance(bw, (int, float))), f"BW {bw} not an approved type!"
    assert isinstance(norm, int)
    match method:
        case "naive":
            return NaiveKDE(kernel=kernel_type, bw=bw, norm=norm)
        case "tree":
            return TreeKDE(kernel=kernel_type, bw=bw, norm=norm)
        case "FFT":
            return FFTKDE(kernel=kernel_type, bw=bw, norm=norm)
        case _:
            raise ValueError(f"KDE method {method} not supported! Must be one of (FFT, naive, tree).")


def fitKDE(data: np.ndarray | list, kernel_type: str = "gaussian", bw: str | float | int = "ISJ", norm: int = 2, method: str = "FFT") -> NaiveKDE | TreeKDE | FFTKDE:
    return selectKDE(kernel_type, bw, norm, method).fit(data)