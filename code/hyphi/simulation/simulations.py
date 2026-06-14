"""
Simulations module for HyPhi: Kuramoto model and Watts-Strogatz network sweeps.

1. Connectome-informed Kuramoto model with delays
   (structural outline extracted from connectome_kuramoto.ipynb)
2. Watts-Strogatz small-world sweep
   (parameterized version of the sweep in NeuRepsSimulations.py)

Years: 2026
"""

# %% Import
from __future__ import annotations

import warnings

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

    # Symmetrize and zero diagonal
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
        Connectivity weight matrix (N x N single-brain, 2N x 2N for dual-brain). Must be square,
        symmetric, and finite. It is spectrally normalized (divided by its largest eigenvalue) so
        ``c_intra`` is a scale-free coupling strength independent of the connectome's magnitude.
    tract : np.ndarray
        Tract lengths matrix (same shape as ``W``), in metres. Zero-length tracts give zero delay
        (instantaneous coupling): this is the intended model for inter-brain links, which are
        coupled in weight but carry no conduction delay.
    omega : np.ndarray
        Natural frequencies of the oscillators (length ``W.shape[0]``).
    c_intra : float
        Coupling scale (applied to the spectrally normalized weights).
    velocity : float
        Axonal velocity in m/s (must be positive).
    noise_strength : float
        Diffusion coefficient D of the additive phase noise (the reference's "Dirac delta noise
        strength"); applied in :func:`run_delayed_kuramoto` as a Wiener increment.
    seed : int
        Random seed.

    Returns
    -------
    object
        Configured DDE solver. The seeded rng and the noise strength travel on the solver so the
        run function keeps its signature.

    Raises
    ------
    ValueError
        If ``velocity`` is not positive; if ``W``/``tract``/``omega`` contain non-finite values;
        if ``W`` is not square or ``tract``/``omega`` shapes do not match ``W``; if ``W`` is not
        symmetric; or if ``W`` has no positive eigenvalue (it cannot be spectrally normalized).

    Notes
    -----
    The solver uses ``jitcdde`` with weighted, delayed coupling::

        d(theta_i)/dt = omega_i + c_intra * sum_j W_ji * sin(theta_j(t - delay_ij) - theta_i(t))

    where ``delay_ij = tract_ij / velocity`` and ``W`` is spectrally normalized. A connection is
    included whenever ``W_ji != 0`` (a zero delay is a valid instantaneous term), matching the
    reference ``connectome_kuramoto.ipynb``.

    """
    from jitcdde import jitcdde, t, y  # noqa: PLC0415 (lazy: keep the DDE compile stack out of import hyphi)
    from symengine import sin  # noqa: PLC0415 (lazy: keep the DDE compile stack out of import hyphi)

    w = np.asarray(W, dtype=float)
    tract_arr = np.asarray(tract, dtype=float)
    omega_arr = np.asarray(omega, dtype=float)
    if velocity <= 0:
        msg = f"velocity must be positive (got {velocity}); a non-positive velocity has no physical meaning."
        raise ValueError(msg)
    if not (np.isfinite(w).all() and np.isfinite(tract_arr).all() and np.isfinite(omega_arr).all()):
        msg = "W, tract, and omega must be finite (no inf or NaN); clean the connectome first."
        raise ValueError(msg)
    n_osc = int(w.shape[0])
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        msg = f"W must be a square matrix, got shape {w.shape}."
        raise ValueError(msg)
    if tract_arr.shape != w.shape:
        msg = f"tract must match W shape {w.shape}, got {tract_arr.shape}."
        raise ValueError(msg)
    if omega_arr.shape != (n_osc,):
        msg = f"omega must have length {n_osc} (one per oscillator), got shape {omega_arr.shape}."
        raise ValueError(msg)
    if not np.allclose(w, w.T):
        msg = "W must be symmetric for spectral normalization; symmetrize the connectome first."
        raise ValueError(msg)

    eig_max = float(np.linalg.eigvalsh(w).max())
    if eig_max <= 0:
        msg = "W has no positive eigenvalue, so it cannot be spectrally normalized (is it all-zero?)."
        raise ValueError(msg)
    w = w / eig_max  # scale-free c_intra: dynamics no longer depend on the connectome's magnitude
    delays = tract_arr / velocity  # conduction delay (seconds) per connection; 0 for instantaneous links

    def rhs():
        for i in range(n_osc):
            # Include every weighted connection; a zero delay is a valid instantaneous coupling term
            # (the inter-brain links in the dual-brain model are coupled but carry no conduction delay).
            coupling = sum(
                c_intra * float(w[j, i]) * sin(y(j, t - float(delays[i, j])) - y(i))
                for j in range(n_osc)
                if w[j, i] != 0
            )
            yield omega_arr[i] + coupling

    solver = jitcdde(rhs, n=n_osc, verbose=False)
    rng = np.random.default_rng(seed)
    initial_phases = rng.uniform(0.0, 2.0 * np.pi, n_osc)
    solver.constant_past(initial_phases, time=0.0)
    solver.set_integration_parameters(rtol=0, atol=1e-5)
    max_delay = float(delays.max()) if delays.size else 0.0
    solver.integrate_blindly(max(max_delay, 0.01), 0.01)
    # The noise configuration travels with the solver so run_delayed_kuramoto keeps its signature.
    solver.hyphi_noise_strength = float(noise_strength)  # ty: ignore[unresolved-attribute]
    solver.hyphi_rng = rng  # ty: ignore[unresolved-attribute]
    return solver


def run_delayed_kuramoto(
    solver: object,
    dt: float,
    t_max: float,
    n_osc: int,
    t_skip: float = 2.0,
    n_per_brain: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the delayed Kuramoto model and return phase trajectories.

    Parameters
    ----------
    solver : object
        Configured DDE solver from :func:`setup_delayed_kuramoto`.
    dt : float
        Integration time step (must be positive).
    t_max : float
        Total simulation horizon (must be positive).
    n_osc : int
        Number of oscillators.
    t_skip : float
        Seconds of initial transient to discard (must be shorter than the horizon).
    n_per_brain : int or None
        For a dual-brain run, the number of oscillators per brain (``n_osc / 2``); the order
        parameter is then the per-brain average ``(rA + rB) / 2`` over oscillators ``[0:n_per_brain]``
        and ``[n_per_brain:]``, matching the reference. When ``None`` (single brain), the order
        parameter is the global mean over all oscillators.

    Returns
    -------
    tuple
        ``(times, theta_history, order_parameters)`` after discarding the first ``t_skip`` seconds.
        ``theta_history`` is ``(n_kept_steps, n_osc)``.

    Raises
    ------
    ValueError
        If ``dt`` or ``t_max`` is not positive; if ``n_per_brain`` is given but ``2 * n_per_brain``
        does not equal ``n_osc``; or if ``t_skip`` is not shorter than the simulated horizon (which
        would leave no samples).

    Notes
    -----
    For each step the phases are read from the solver, wrapped to ``[0, 2pi)``, and given an additive
    Wiener-increment read-out noise ``sqrt(2 * D * dt) * N(0, 1)`` (``D`` = ``noise_strength`` from
    setup). The Kuramoto order parameter is ``r = |mean(exp(i * theta))|`` in ``[0, 1]``.

    """
    if dt <= 0:
        msg = f"dt must be positive, got {dt}."
        raise ValueError(msg)
    if t_max <= 0:
        msg = f"t_max must be positive, got {t_max}."
        raise ValueError(msg)
    if n_per_brain is not None and 2 * n_per_brain != n_osc:
        msg = f"n_per_brain ({n_per_brain}) must be half of n_osc ({n_osc}) for a dual-brain split."
        raise ValueError(msg)

    start = float(getattr(solver, "t", 0.0))
    # Sample strictly ahead of the solver's current time so every integration step advances.
    n_steps = max(1, round(t_max / dt))
    times = start + np.arange(1, n_steps + 1) * dt
    horizon = n_steps * dt
    if t_skip >= horizon:
        msg = f"t_skip ({t_skip}) must be shorter than the simulated horizon ({horizon}); nothing would remain."
        raise ValueError(msg)

    noise_strength = float(getattr(solver, "hyphi_noise_strength", 0.0))
    rng = getattr(solver, "hyphi_rng", None)
    if rng is None:
        rng = np.random.default_rng(0)
    # One segment (global r) for a single brain; two equal halves (per-brain r, averaged) for a dyad.
    segments = (slice(None),) if n_per_brain is None else (slice(0, n_per_brain), slice(n_per_brain, None))

    theta_history = np.empty((len(times), n_osc), dtype=float)
    order_parameters = np.empty(len(times), dtype=float)
    with warnings.catch_warnings():
        # Dense sampling (dt finer than the adaptive step) makes jitcdde interpolate within an already
        # accepted step and warn; verified benign (the integration always advances), so silence the flood.
        warnings.filterwarnings(
            "ignore", message="The target time is smaller than the current time", category=UserWarning
        )
        for k, time in enumerate(times):
            phases = np.asarray(solver.integrate(time), dtype=float) % (2.0 * np.pi)  # ty: ignore[unresolved-attribute]
            if noise_strength:
                phases = (phases + np.sqrt(2.0 * noise_strength * dt) * rng.standard_normal(n_osc)) % (2.0 * np.pi)
            theta_history[k] = phases
            order_parameters[k] = float(np.mean([np.abs(np.mean(np.exp(1j * phases[seg]))) for seg in segments]))

    keep = times >= (start + t_skip)
    return times[keep], theta_history[keep], order_parameters[keep]


# ====================================================
# 2. Watts-Strogatz Small-World Sweep
# ====================================================


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
        Neighborhood size.
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
    from hyphi.modeling.entropies import vec_entropy, vec_quantiles
    from hyphi.modeling.graph_curvatures import compute_frc_vec
    from hyphi.simulation.graph_simulations import gen_weighted_sw

    pt = np.logspace(min_pow, max_pow, t_rez)
    qvals = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    Hreps = np.zeros((n_reps, t_rez))
    Qreps = np.zeros((n_reps, t_rez, 5))

    for rep in range(n_reps):
        # Generate time-varying SW networks
        Gt = [gen_weighted_sw(n, k, pt[t], epsilon, seed_val=seed_base + rep) for t in range(t_rez)]

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
