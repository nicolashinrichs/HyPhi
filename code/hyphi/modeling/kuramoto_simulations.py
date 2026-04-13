"""TODO: describe what this module does."""

# %% Import
from jax import random, jit, vmap, lax  # TODO: add as dependency
import jax.numpy as jnp
import numpy as np
import networkx as nx

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def kuramoto_vector_field(thetas, K, omegas):
    """Compute the Kuramoto vector field (all-to-all mean field)."""
    coss, sins = jnp.cos(thetas), jnp.sin(thetas)
    rx, ry = jnp.mean(coss), jnp.mean(sins)
    return omegas + K * (ry * coss - rx * sins)


def rk4(func, state, dt):
    """Compute the one-step RK4 integrator."""
    k1 = func(state)
    k2 = func(state + k1 * dt / 2)
    k3 = func(state + k2 * dt / 2)
    k4 = func(state + k3 * dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def simulate_kuramoto(func, solver, init_state, dt, n_steps):
    """
    Integrate and save full phase trajectories.

    :returns theta_hist: array (n_steps+1, N) with every phase snapshot.
    """

    # compiled single-step RK4
    @jit
    def update(state):
        return solver(func, state, dt)

    N = init_state.size
    theta_hist = jnp.zeros((n_steps + 1, N))
    theta_hist = theta_hist.at[0].set(init_state)

    def body(i, hist):
        new_state = update(hist[i - 1])
        return hist.at[i].set(new_state)

    theta_hist = lax.fori_loop(1, n_steps + 1, body, theta_hist)
    return theta_hist


def get_plv_pair(phi_i, phi_j):
    """Get the PLV for one oscillator pair in one window."""
    return jnp.abs(jnp.mean(jnp.exp(1j * (phi_i - phi_j))))


# TODO: should this be here, alone?
# vectorize across j to build full row (PLV for )  # TODO: for ...?!
getPLVRow = vmap(get_plv_pair, in_axes=(None, 0), out_axes=0)


def get_plv_matrix(phase_window):
    """
    Get the PLV matrix.

    :phase_window: (N, T_w) → PLV matrix (N, N)
    :returns: (N, N) PLV matrix (N, N)
    """
    n = phase_window.shape[0]
    c = jnp.zeros((n, n))
    for i in range(n):  # plv pair of osc. i with every other osc.
        c = c.at[i].set(getPLVRow(phase_window[i], phase_window))
    # ensure symmetry (c_ij = c_ji)
    return (c + c.T) / 2.0


def get_plv_graphs(n_steps, w_size, w_stride, theta_hist):
    """Get the PLV graphs."""
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
    """Get the average PLV over time."""
    avg_plv = [np.mean([d["weight"] for _, _, d in g.edges(data=True)]) for g in graphs]
    return np.array(avg_plv)


def kuramoto_test_sim():
    """Compute a Kuramoto test simulation."""
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
