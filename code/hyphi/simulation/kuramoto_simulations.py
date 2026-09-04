"""Kuramoto oscillator simulations: integration, PLV connectivity, and graph construction.

Provides RK4 integration of the all-to-all Kuramoto model, sliding-window
Phase-Locking Value (PLV) matrix computation via JAX/vmap, conversion of PLV
matrices to NetworkX graphs, and a convenience test simulation.
"""

# %% Import
import jax.numpy as jnp
import networkx as nx
import numpy as np
from jax import jit, lax, random, vmap

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def kuramoto_vector_field(thetas, K, omegas):
    """Compute the Kuramoto vector field (all-to-all mean field).

    Parameters
    ----------
    thetas : jnp.ndarray
        Phase angles of the N oscillators, shape (N,).
    K : float
        Global coupling strength.
    omegas : jnp.ndarray
        Natural frequencies of the N oscillators, shape (N,).

    Returns
    -------
    jnp.ndarray
        Time derivatives d(thetas)/dt, shape (N,).
    """
    coss, sins = jnp.cos(thetas), jnp.sin(thetas)
    rx, ry = jnp.mean(coss), jnp.mean(sins)
    return omegas + K * (ry * coss - rx * sins)


def rk4(func, state, dt):
    """Compute a single RK4 integration step.

    Parameters
    ----------
    func : callable
        Vector field function f(state) -> d(state)/dt.
    state : jnp.ndarray
        Current state vector, shape (N,).
    dt : float
        Integration time step.

    Returns
    -------
    jnp.ndarray
        Updated state after one RK4 step, shape (N,).
    """
    k1 = func(state)
    k2 = func(state + k1 * dt / 2)
    k3 = func(state + k2 * dt / 2)
    k4 = func(state + k3 * dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def simulate_kuramoto(func, solver, init_state, dt, n_steps):
    """Integrate and save full phase trajectories.

    Parameters
    ----------
    func : callable
        Vector field function f(state) -> d(state)/dt.
    solver : callable
        One-step ODE solver with signature solver(func, state, dt).
    init_state : jnp.ndarray
        Initial phase angles, shape (N,).
    dt : float
        Integration time step.
    n_steps : int
        Number of integration steps to perform.

    Returns
    -------
    jnp.ndarray
        Phase history array of shape (n_steps + 1, N), where index 0 is
        the initial state.
    """

    # compiled single-step RK4
    @jit
    def update_state(state):
        return solver(func, state, dt)

    N = init_state.size
    theta_hist = jnp.zeros((n_steps + 1, N))
    theta_hist = theta_hist.at[0].set(init_state)

    def body(i, hist):
        new_state = update_state(hist[i - 1])
        return hist.at[i].set(new_state)

    return lax.fori_loop(1, n_steps + 1, body, theta_hist)  # theta hist


def get_plv_pair(phi_i, phi_j):
    """Compute the Phase-Locking Value (PLV) for one oscillator pair over a window.

    Parameters
    ----------
    phi_i : jnp.ndarray
        Phase time series of oscillator i, shape (T_w,).
    phi_j : jnp.ndarray
        Phase time series of oscillator j, shape (T_w,).

    Returns
    -------
    jnp.ndarray
        Scalar PLV in [0, 1].
    """
    return jnp.abs(jnp.mean(jnp.exp(1j * (phi_i - phi_j))))


# Note: get_plv_row is a vmapped helper that builds one full row of the PLV matrix
# by vectorizing get_plv_pair across all j oscillators for a fixed oscillator i.
# Its placement here (module level) is intentional so it is JIT-compiled once and
# reused across calls to get_plv_matrix.
get_plv_row = vmap(get_plv_pair, in_axes=(None, 0), out_axes=0)


def get_plv_matrix(phase_window):
    """Compute the PLV matrix from a windowed phase array.

    Parameters
    ----------
    phase_window : jnp.ndarray
        Phase time series of shape (N, T_w), where N is the number of
        oscillators and T_w is the window length in samples.

    Returns
    -------
    jnp.ndarray
        Symmetric PLV matrix of shape (N, N) with values in [0, 1].
    """
    n = phase_window.shape[0]
    c = jnp.zeros((n, n))
    for i in range(n):  # plv pair of osc. i with every other osc.
        c = c.at[i].set(get_plv_row(phase_window[i], phase_window))
    # ensure symmetry (c_ij = c_ji)
    return (c + c.T) / 2.0


def get_plv_graphs(n_steps, w_size, w_stride, theta_hist):
    """Build a sequence of PLV-weighted NetworkX graphs over sliding windows.

    Parameters
    ----------
    n_steps : int
        Total number of integration steps (excluding the initial state).
    w_size : int
        Window length in samples.
    w_stride : int
        Stride between consecutive windows in samples.
    theta_hist : jnp.ndarray
        Full phase history of shape (n_steps + 1, N).

    Returns
    -------
    list of nx.Graph
        One fully connected weighted graph per window; edge weight equals
        the PLV between the two endpoint oscillators.
    """
    graphs = []
    for start in range(0, n_steps - w_size + 1, w_stride):
        # slice window & transpose to (N, T_w)
        window_phases = theta_hist[start : start + w_size].T
        c = get_plv_matrix(window_phases)  # PLV connectivity
        c_np = np.asarray(c)  # convert to NumPy for networkx
        graph = nx.from_numpy_array(c_np, create_using=nx.Graph)
        graphs.append(graph)
    return graphs


def get_avg_plv(graphs):
    """Compute the mean PLV edge weight at each time window.

    Parameters
    ----------
    graphs : list of nx.Graph
        Sequence of PLV-weighted graphs, one per time window.

    Returns
    -------
    np.ndarray
        1-D array of mean PLV values, one entry per graph.
    """
    avg_plv = [np.mean([d["weight"] for _, _, d in g.edges(data=True)]) for g in graphs]
    return np.array(avg_plv)


def kuramoto_test_sim():
    """Run a Kuramoto test simulation and return phase history and PLV graphs.

    Uses 100 oscillators with Gaussian natural frequencies and uniform random
    initial phases, integrated for 20 seconds at dt=0.01 with coupling K=2.

    Returns
    -------
    time_axis : np.ndarray
        Time stamps (in seconds) for each PLV graph window, shape (n_windows,).
    theta_hist : jnp.ndarray
        Full phase history of shape (n_steps + 1, N).
    plv_graphs : list of nx.Graph
        PLV-weighted graphs for each sliding window.
    """
    # parameters
    n_oscillators = 100
    coupling_strength = 2.0  # K
    dt = 0.01
    t_max = 20.0
    n_steps = int(t_max / dt)

    omegas = random.normal(random.PRNGKey(0), (n_oscillators,))
    init_thetas = random.uniform(random.PRNGKey(1), (n_oscillators,), maxval=2 * jnp.pi)

    # simulate
    theta_hist = simulate_kuramoto(
        lambda th: kuramoto_vector_field(th, coupling_strength, omegas), rk4, init_thetas, dt, n_steps
    )  # shape = (n_steps+1, N)

    # build sliding-window graphs
    win_len = 2.0  # seconds
    win_step = 0.5  # seconds
    w_size = int(win_len / dt)  # samples per window
    w_stride = int(win_step / dt)

    plv_graphs = get_plv_graphs(n_steps, w_size, w_stride, theta_hist)

    time_axis = np.arange(len(plv_graphs)) * win_step

    return time_axis, theta_hist, plv_graphs


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
