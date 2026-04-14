"""
Simulations module for HyPhi: Kuramoto model and Watts-Strogatz network sweeps.

1. Connectome-informed Kuramoto model with delays
   (structural outline extracted from connectome_kuramoto.ipynb)
2. Watts-Strogatz small-world sweep
   (parameterised version of the sweep in NeuRepsSimulations.py)

Years: 2026
"""

# %% Import
from __future__ import annotations

import networkx as nx
import numpy as np

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ====================================================
# 1. Connectome-Informed Kuramoto with Delays
# ====================================================


def load_connectome(pickle_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load structural connectivity and tract lengths from pickle.

    Parameters
    ----------
    pickle_path : str
        Path to ``connectivity_data.pkl`` containing ``(W, tract, roi_names, ...)``.

    Returns
    -------
    tuple
        ``(W, tract, roi_names)`` — connectivity matrix, tract lengths, region labels.

    """
    import pickle

    with open(pickle_path, "rb") as f:  # noqa: S301
        W, tract, roi_names, _centers_raw, _hemis_raw, _areas_raw = pickle.load(f)  # noqa: S301

    # Symmetrise and zero diagonal
    W = (W + W.T) / 2.0
    tract = (tract + tract.T) / 2.0
    np.fill_diagonal(W, 0)
    np.fill_diagonal(tract, 0)

    # Convert tract lengths from mm to m
    tract /= 1000.0

    roi_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in roi_names]

    return W, tract, roi_names


def create_virtual_partner_connectome(
    W: np.ndarray,
    D: np.ndarray,
    roi_names: list[str],
    C_inter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand a single-brain connectome (NxN) into a dual-brain connectome (2Nx2N).

    Parameters
    ----------
    W : np.ndarray
        NxN connectivity weight matrix.
    D : np.ndarray
        NxN tract lengths matrix.
    roi_names : list[str]
        Region of interest labels.
    C_inter : float
        Inter-brain coupling strength as a percentage of mean intra-brain weight.

    Returns
    -------
    tuple
        ``(W_vir, D_vir)`` — expanded 2Nx2N matrices.

    """
    n = W.shape[0]
    W_vir = np.zeros((2 * n, 2 * n))
    D_vir = np.zeros((2 * n, 2 * n))

    W_vir[:n, :n] = W
    W_vir[n:, n:] = W
    D_vir[:n, :n] = D
    D_vir[n:, n:] = D

    motor_labels = ["lM1", "rM1"]
    visual_labels = ["rV1", "lV1"]

    motor_idx = [roi_names.index(name) for name in motor_labels]
    visual_idx = [roi_names.index(name) for name in visual_labels]

    mean_W = W[W > 0].mean()
    scale = (mean_W / 100.0) * C_inter

    W_inter = np.zeros((n, n))
    W_inter[np.ix_(motor_idx, visual_idx)] = scale
    W_inter = W_inter + W_inter.T

    W_vir[:n, n:] = W_inter
    W_vir[n:, :n] = W_inter

    np.fill_diagonal(W_vir, 0)
    np.fill_diagonal(D_vir, 0)

    return W_vir, D_vir


def setup_delayed_kuramoto(
    W: np.ndarray,
    tract: np.ndarray,
    omega: np.ndarray,
    c_intra: float = 2.0,
    velocity: float = 1.65,
    noise_strength: float = 0.1,
    seed: int = 42,
) -> object:
    """
    Set up the delayed Kuramoto DDE solver.

    Parameters
    ----------
    W : np.ndarray
        Connectivity matrix (2N x 2N for dual-brain).
    tract : np.ndarray
        Tract lengths matrix (2N x 2N).
    omega : np.ndarray
        Natural frequencies of oscillators.
    c_intra : float
        Intra-brain coupling scale.
    velocity : float
        Axonal velocity in m/s.
    noise_strength : float
        Dirac delta noise strength.
    seed : int
        Random seed.

    Returns
    -------
    object
        Configured DDE solver.

    Notes
    -----
    The solver uses ``jitcdde`` with delayed coupling::

        d(theta_i)/dt = omega_i + c * sum_j W_ji * sin(theta_j(t - delay_ij) - theta_i(t))

    where ``delay_ij = tract_ij / velocity``.

    Reference implementation in ``connectome_kuramoto.ipynb``::

        from jitcdde import jitcdde, t, y
        from symengine import sin

        delays = tract / velocity
        solver = jitcdde(system_generator, n=n_osc, delays=delays.flatten())
        solver.set_integration_parameters(rtol=0, atol=1e-5)
        solver.constant_past(initial_phases, time=0.0)
        solver.integrate_blindly(max_delay, 0.1)

    """
    raise NotImplementedError(
        "TODO: Insert original delayed Kuramoto DDE simulation logic. "
        "See connectome_kuramoto.ipynb for reference implementation."
    )


def run_delayed_kuramoto(
    solver: object,
    dt: float,
    t_max: float,
    n_osc: int,
    t_skip: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the delayed Kuramoto model and return phase trajectories.

    Parameters
    ----------
    solver : object
        Configured DDE solver from :func:`setup_delayed_kuramoto`.
    dt : float
        Integration time step.
    t_max : float
        Total simulation time.
    n_osc : int
        Number of oscillators.
    t_skip : float
        Seconds of transients to discard.

    Returns
    -------
    tuple
        ``(times, theta_history, order_parameters)``

    Notes
    -----
    For each time step::

        phases = solver.integrate(time) % (2 * np.pi)
        phases += noise_term
        r = |mean(exp(j * phases))|   # order parameter

    Return trimmed results after ``t_skip``.

    """
    raise NotImplementedError(
        "TODO: Insert original delayed Kuramoto integration loop. "
        "See connectome_kuramoto.ipynb for reference implementation."
    )


# ====================================================
# 2. Watts-Strogatz Small-World Sweep
# ====================================================


def gen_weighted_sw(n: int, k: int, p: float, epsilon: float, seed: int = 42) -> nx.Graph:
    """
    Generate a single weighted small-world graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is connected to *k* nearest neighbours in ring topology.
    p : float
        Rewiring probability.
    epsilon : float
        Spacing parameter for distance-based weights.
    seed : int
        Random seed.

    Returns
    -------
    nx.Graph
        Weighted Watts-Strogatz graph.

    """
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    nx.set_node_attributes(G, values=1.0, name="weight")

    Dmax = (np.floor(n / 2) + 1) * epsilon
    for ii, jj in G.edges:
        d_ij = min(np.abs(ii - jj), n - np.abs(ii - jj))
        G[ii][jj]["weight"] = (Dmax - d_ij).item()

    return G


def run_ws_sweep(
    n: int = 1000,
    k: int = 50,
    epsilon: float = 1.0,
    t_rez: int = 100,
    min_pow: float = -4,
    max_pow: float = 0,
    n_reps: int = 200,
    seed_base: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Watts-Strogatz sweep across rewiring probabilities.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Neighbourhood size.
    epsilon : float
        Spacing parameter.
    t_rez : int
        Number of probability grid points.
    min_pow, max_pow : float
        log10 range for rewiring probability.
    n_reps : int
        Number of replications.
    seed_base : int
        Base seed for replication.

    Returns
    -------
    tuple
        ``(pt, Hreps, Qreps)`` — probability grid, entropy array, quantile array.
        Shapes: ``pt (t_rez,)``, ``Hreps (n_reps, t_rez)``, ``Qreps (n_reps, t_rez, 5)``.

    """
    from hyphi.analyses import compute_frc_vec
    from hyphi.modeling.entropies import vec_entropy, vec_quantiles

    pt = np.logspace(min_pow, max_pow, t_rez)
    qvals = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    Hreps = np.zeros((n_reps, t_rez))
    Qreps = np.zeros((n_reps, t_rez, 5))

    for rep in range(n_reps):
        # Generate time-varying SW networks
        Gt = [gen_weighted_sw(n, k, pt[t], epsilon, seed=seed_base + rep) for t in range(t_rez)]

        # Compute curvatures
        FRCt = compute_frc_vec(Gt)

        # Get entropy
        Ht = vec_entropy(FRCt)
        Hreps[rep, :] = Ht

        # Get quantiles
        Qt = vec_quantiles(FRCt, qs=qvals)
        Qreps[rep, :, :] = Qt

    return pt, Hreps, Qreps


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
