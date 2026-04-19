import numpy as np
import scipy.linalg as la
import hyphi.spectral.laplace as lap

def main():
    return 0

def diffusion_distance(adj1: np.ndarray, adj2: np.ndarray, time_limit: float, fs: float) -> float:
    """
    Function calculating heat diffusion distance (d_gdd) of two square matrices matching in shape.

    Input
    Adjacency matrix 1, adjacency matrix 2, time_limit, fs
    time limit - how many timepoints (in seconds) should be analysed to find the maximum distance of two heat distributions
    fs - how many timepoints in one second should be analysed

    Output
    maximum diffusion distance (d_gdd) in the given time window (0, time_limit)
    
    """

    frobenius = lambda A: la.norm(A, ord='fro')
    exp = lambda t, eigvals, eigvecs: eigvecs @ la.expm(t * eigvals) @ la.inv(eigvecs)

    time = np.arange(0, time_limit, 1/fs)

    eigvals1, eigvecs1, _ = lap.laplace(adj1)
    eigvals2, eigvecs2, _ = lap.laplace(adj2)


    diffusion_distance = np.array([frobenius(exp(t, eigvals1, eigvecs1) - exp(t, eigvals2, eigvecs2)) for t in time])  # should be a 1D matrix

    return np.max(diffusion_distance)







if __name__ == '__main__':
    main()
