from jax import random, jit, vmap, lax
import jax.numpy as jnp
import numpy as np             
import networkx as nx
import matplotlib.pyplot as plt

# Kuramoto vector field (all-to-all mean field)
def kuramotoVectorField(thetas, K, omegas):
    coss, sins = jnp.cos(thetas), jnp.sin(thetas)
    rx, ry = jnp.mean(coss), jnp.mean(sins)
    return omegas + K * (ry * coss - rx * sins)


# one-step RK4 integrator 
def rk4(func, state, dt):
    k1 = func(state)
    k2 = func(state + k1 * dt / 2)
    k3 = func(state + k2 * dt / 2)
    k4 = func(state + k3 * dt)
    return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6


# integrate & save full phase trajectories 
def simulateKuramoto(func, solver, init_state, dt, n_steps):
    """
    Returns theta_hist: array (n_steps+1, N) with every phase snapshot.
    """
    # compiled single-step RK4
    @jit
    def update(state):
        return solver(func, state, dt)

    N = init_state.size
    theta_hist = jnp.zeros((n_steps + 1, N))
    theta_hist = theta_hist.at[0].set(init_state)

    def body(i, hist):
        new_state   = update(hist[i - 1])
        return hist.at[i].set(new_state)

    theta_hist = lax.fori_loop(1, n_steps + 1, body, theta_hist)
    return theta_hist


# PLV for one oscillator pair in one window
def getPLVPair(phi_i, phi_j):
    return jnp.abs(jnp.mean(jnp.exp(1j*(phi_i - phi_j))))


# vectorise across j to build full row (PLV for )
getPLVRow = vmap(getPLVPair, in_axes=(None, 0), out_axes=0)


def getPLVMatrix(phase_window):
    """phase_window: (N, T_w) → PLV matrix (N, N)"""
    N = phase_window.shape[0]
    C = jnp.zeros((N, N))
    for i in range(N): # plv pair of osc. i with every other osc.
        C = C.at[i].set(getPLVRow(phase_window[i], phase_window))
    # ensure symmetry (c_ij = c_ji)
    C = (C + C.T) / 2.0
    return C


def getPLVGraphs(n_steps, w_size, w_stride, theta_hist):
    graphs = []
    for start in range(0, n_steps - w_size + 1, w_stride):
        # slice window & transpose to (N, T_w)
        window_phases = theta_hist[start:start+w_size].T
        C = getPLVMatrix(window_phases)              # PLV connectivity
        C_np = np.asarray(C)                       # convert to NumPy for networkx
        G = nx.from_numpy_array(C_np, create_using=nx.Graph)
        graphs.append(G)
    return graphs


# average PLV over time 
def getAvgPLV(graphs): 
    avg_plv = [np.mean([d['weight'] for _,_,d in g.edges(data=True)]) for g in graphs]
    return np.array(avg_plv)


def kuramotoTestSim():
    # parameters 
    N        = 100 # no. of oscillators
    K        = 2.0 # coupling strength 
    dt       = 0.01
    t_max    = 20.0
    n_steps  = int(t_max / dt)

    omegas      = random.normal(random.PRNGKey(0), (N,))
    init_thetas = random.uniform(random.PRNGKey(1), (N,), maxval=2*jnp.pi)

    # simulate 
    theta_hist = simulateKuramoto(
        lambda th: kuramotoVectorField(th, K, omegas),
        rk4,
        init_thetas,
        dt,
        n_steps
    )  # shape = (n_steps+1, N)

    # build sliding-window graphs 
    win_len   = 2.0           # seconds
    win_step  = 0.5           # seconds
    w_size    = int(win_len  / dt)   # samples per window
    w_stride  = int(win_step / dt)

    plv_graphs = getPLVGraphs(n_steps, w_size, w_stride, theta_hist)

    time_axis = np.arange(len(plv_graphs)) * win_step

    return time_axis, theta_hist, plv_graphs