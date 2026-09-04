"""Graph simulation utilities for generating small-world network time series."""

# %% Import
import networkx as nx
import numpy as np

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def gen_weighted_sw(n: int, k: int, p: float, epsilon: float, seed_val: int = 42):
    """Generate a weighted Watts-Strogatz small-world graph.

    Node weights are set to unity for curvature computations.  Edge weights
    are derived from the ring-lattice distance so that closer nodes on the
    original ring receive higher weight: ``weight = Dmax - d_ij``, where
    ``Dmax = (floor(n / 2) + 1) * epsilon``.

    Parameters
    ----------
    n : int
        Number of nodes in the ring lattice.
    k : int
        Each node is initially connected to ``k`` nearest neighbours.
    p : float
        Probability of rewiring each edge (Watts-Strogatz parameter).
    epsilon : float
        Spacing between adjacent nodes on the ring lattice; scales edge
        weights.
    seed_val : int, optional
        Random seed passed to :func:`networkx.watts_strogatz_graph`.
        Default is 42.

    Returns
    -------
    G : networkx.Graph
        Weighted undirected graph with ``n`` nodes.  Node attribute
        ``"weight"`` is 1.0 for every node; edge attribute ``"weight"``
        encodes the ring-lattice proximity.
    """
    # Generate a small world network using Watts-Strogatz
    G = nx.watts_strogatz_graph(n, k, p, seed=seed_val)

    # Set the node weights to unity for curvature computations
    nx.set_node_attributes(G, values=1.0, name="weight")

    # Maximum distance between nodes = max(d_ij) + epsilon
    # If nodes are spaced epsilon apart on the ring, then
    # max(d_ij) = floor(n/2) * epsilon even if n is odd
    Dmax = (np.floor(n / 2) + 1) * epsilon

    for ii, jj in G.edges:
        # Distance between nodes is the shortest path around the ring
        d_ij = min(np.abs(ii - jj), n - np.abs(ii - jj))
        # abs() returns an np.float, get the regular float from it
        G[ii][jj]["weight"] = (Dmax - d_ij).item()

    return G


def gen_tv_weighted_sw(
    n: int,
    k: int,
    epsilon: float,
    trez: int,
    minpow: float | int,
    maxpow: float | int,
    seed_val: int = 42,
):
    """Generate a time-varying sequence of weighted small-world graphs.

    Rewiring probability is swept on a log scale from ``10**minpow`` to
    ``10**maxpow`` at ``trez`` points.  Each graph in the series is produced
    by :func:`gen_weighted_sw`.

    Parameters
    ----------
    n : int
        Number of nodes in each graph.
    k : int
        Each node is initially connected to ``k`` nearest neighbours.
    epsilon : float
        Node spacing on the ring lattice; passed through to
        :func:`gen_weighted_sw`.
    trez : int
        Number of time / probability points (resolution of the sweep).
    minpow : float or int
        Exponent of the minimum rewiring probability (``10**minpow``).
    maxpow : float or int
        Exponent of the maximum rewiring probability (``10**maxpow``).
    seed_val : int, optional
        Random seed for graph generation.  Default is 42.

    Returns
    -------
    pt : numpy.ndarray
        Array of rewiring probabilities of shape ``(trez,)``.
    Gt : list of networkx.Graph
        List of weighted graphs, one per probability point.
    """
    # "Time" / probability points for simulation
    pt = np.logspace(minpow, maxpow, trez)

    # Initialize empty list for graphs
    Gt: list[nx.Graph] = []

    # Simulate
    for t in range(trez):
        Gt.append(gen_weighted_sw(n, k, pt[t], epsilon, seed_val=seed_val))

    # Return time series of graphs
    return pt, Gt


def gen_tv_sw(n: int, k: int, trez: int, minpow: float | int, maxpow: float | int, seed_val: int = 42):
    """Generate a time-varying sequence of unweighted small-world graphs.

    Rewiring probability is swept on a log scale from ``10**minpow`` to
    ``10**maxpow`` at ``trez`` points using the Watts-Strogatz model.

    Parameters
    ----------
    n : int
        Number of nodes in each graph.
    k : int
        Each node is initially connected to ``k`` nearest neighbours.
    trez : int
        Number of time / probability points (resolution of the sweep).
    minpow : float or int
        Exponent of the minimum rewiring probability (``10**minpow``).
    maxpow : float or int
        Exponent of the maximum rewiring probability (``10**maxpow``).
    seed_val : int, optional
        Random seed for graph generation.  Default is 42.

    Returns
    -------
    pt : numpy.ndarray
        Array of rewiring probabilities of shape ``(trez,)``.
    Gt : list of networkx.Graph
        List of unweighted graphs, one per probability point.
    """
    # "Time" / probability points for simulation
    pt = np.logspace(minpow, maxpow, trez)

    # Initialize empty list for graphs
    Gt: list[nx.Graph] = []

    # Simulate
    for t in range(trez):
        Gt.append(nx.watts_strogatz_graph(n, k, pt[t], seed=seed_val))

    # Return time series of graphs
    return pt, Gt


def gen_nature_sw(seed_val: int = 42):
    """Generate the standard Nature-methods small-world graph time series.

    Convenience wrapper for :func:`gen_tv_sw` using the parameters from
    the Nature-methods pipeline: n=1000, k=50, trez=100, rewiring
    probability swept from 1e-4 to 1.

    Parameters
    ----------
    seed_val : int, optional
        Random seed for graph generation.  Default is 42.

    Returns
    -------
    pt : numpy.ndarray
        Array of rewiring probabilities of shape ``(100,)``.
    Gt : list of networkx.Graph
        List of 100 unweighted graphs.
    """
    return gen_tv_sw(1000, 50, 100, -4, 0, seed_val=seed_val)


def gen_neureps_wsw(seed_val: int = 42):
    """Generate the standard NeuroReps weighted small-world graph time series.

    Convenience wrapper for :func:`gen_tv_weighted_sw` using the parameters
    from the NeuroReps pipeline: n=1000, k=50, epsilon=1.0, trez=100, rewiring
    probability swept from 1e-4 to 1.

    Parameters
    ----------
    seed_val : int, optional
        Random seed for graph generation.  Default is 42.

    Returns
    -------
    pt : numpy.ndarray
        Array of rewiring probabilities of shape ``(100,)``.
    Gt : list of networkx.Graph
        List of 100 weighted graphs.
    """
    return gen_tv_weighted_sw(1000, 50, 1.0, 100, -4, 0, seed_val=seed_val)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
