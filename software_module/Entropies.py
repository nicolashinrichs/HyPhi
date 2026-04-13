# ============= #
# Preliminaries # 
# ============= # 

from typing import List
import numpy.typing as npt
from collections.abc import Callable
try:
    import ray
except Exception:
    ray = None

from scipy.stats import differential_entropy
import infomeasure as im
from DensityEstimation import *
from GraphCurvatures import *

# Compute the entropy distribution of the FRC
# or quantify the diversity or spread of curvature

def getEntropyVasicek(G: nx.classes.graph.Graph, window_length: int | None, curvature: str = "formanCurvature") -> float:
    curvatures = extractCurvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="vasicek", window_length=window_length, nan_policy="omit")


def getEntropyVanEs(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> float:
    curvatures = extractCurvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="van es", nan_policy="omit")


def getEntropyEbrahimi(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> float:
    curvatures = extractCurvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="ebrahimi", nan_policy="omit") 


def getEntropyCorr(G: nx.classes.graph.Graph, curvature: str = "formanCurvature") -> float:
    curvatures = extractCurvatures(G, curvature=curvature)
    return differential_entropy(curvatures, method="correa", nan_policy="omit")


def getEntropyKDEPlugin(G: nx.classes.graph.Graph, curvature: str = "formanCurvature", kernel_type: str = "gaussian", bw: str | float | int = "ISJ", norm: int = 2) -> float:
    """  
    Compute plugin entropy estimate using TreeKDE.  
      
    Parameters:  
    - data: array-like, the sample data  
    - bw: bandwidth parameter for KDE  
    - kernel: kernel function name  
      
    Returns:  
    - entropy: plugin entropy estimate 
    """  
    # Extract the graph curvatures
    curvatures = extractCurvatures(G, curvature=curvature)

    # Fit the TreeKDE estimator  
    # We use the TreeKDE because it is faster than naive
    # But unlike the FFTKDE, we can evaluate at arbitrary points
    f = TreeKDE(kernel=kernel_type, bw=bw, norm=norm).fit(curvatures)  
      
    # Evaluate KDE at the original data points  
    fvals = f.evaluate(curvatures)  
      
    # Compute plugin entropy: -E[log f(X)]  
    # Add small epsilon to avoid log(0)  
    epsilon = 1e-10  
    log_fvals = np.log(fvals + epsilon)   

    return -np.mean(log_fvals)


# def getEntropyWaveletPlugin(G: nx.classes.graph.Graph):


def getEntropyKozachenko(G: nx.classes.graph.Graph, curvature: str = "formanCurvature", num_nn: int = 4) -> float:
    curvatures = extractCurvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="metric", k=num_nn)


def getEntropyRenyi(G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, num_nn: int = 4) -> float:
    curvatures = extractCurvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="renyi", alpha=order, k=num_nn)


def getEntropyTsallis(G: nx.classes.graph.Graph, curvature: str = "formanCurvature", order: float | int = 2, num_nn: int = 4) -> float:
    curvatures = extractCurvatures(G, curvature=curvature)
    return im.entropy(curvatures, approach="tsallis", q=order, k=num_nn)


def vecEntropy(
    Gt: npt.NDArray[nx.classes.graph.Graph] | List[nx.classes.graph.Graph],
    estim: Callable = getEntropyKozachenko,
    use_ray: bool = False,
) -> npt.NDArray[float]:
    """Vectorized entropy over a graph series.

    Defaults to sequential execution to avoid Ray runtime issues on
    constrained/shared systems. Set use_ray=True to opt into Ray.
    """
    if use_ray and ray is not None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        @ray.remote
        def par_estim(g):
            return estim(g)

        try:
            refs = [par_estim.remote(G) for G in Gt]
            return np.array(ray.get(refs))
        finally:
            if ray.is_initialized():
                ray.shutdown()

    return np.array([estim(G) for G in Gt])


def getQuantiles(G: nx.classes.graph.Graph, qs: npt.NDArray[float] | List[float], curvature: str = "formanCurvature") -> npt.NDArray[float]:
    curvatures = extractCurvatures(G, curvature=curvature)
    return np.quantile(curvatures, qs)


def vecQuantiles(Gt: npt.NDArray[nx.classes.graph.Graph] | List[nx.classes.graph.Graph], qs: npt.NDArray[float] | List[float], curvature: str = "formanCurvature") -> npt.NDArray[float]:
    return np.array([getQuantiles(G, qs=qs, curvature=curvature) for G in Gt])
