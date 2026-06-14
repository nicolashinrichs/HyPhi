"""
Watts-Strogatz small-world graph generators for ground-truth benchmarks.

Builds static and time-varying small-world graphs by sweeping the Watts-Strogatz rewiring
probability, which carries a network from a regular lattice to a random graph through a
small-world regime. These series give HyPhi a controlled setting whose structural transition
is known, so the curvature-entropy detector can be validated against it. ``gen_tv_sw`` and
``gen_tv_weighted_sw`` produce time-varying (unweighted and weighted) series, and the
package also ships fixed-parameter presets of those generators.
"""

# %% Import
import networkx as nx
import numpy as np

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def gen_weighted_sw(n: int, k: int, p: float, ε: float, seed_val: int = 42):
    """
    Generate a distance-weighted Watts-Strogatz small-world graph.

    Builds a Watts-Strogatz graph, sets every node weight to unity, and assigns each
    edge a weight of ``Dmax - d_ij``, where ``d_ij`` is the shortest distance between the
    two endpoints around the ring and ``Dmax = (floor(n / 2) + 1) * ε`` is the maximum
    possible ring distance plus one spacing.

    Parameters
    ----------
    n : int
        Number of nodes in the ring lattice.
    k : int
        Each node is joined to its ``k`` nearest neighbors in the ring.
    p : float
        Watts-Strogatz rewiring probability.
    ε : float
        Spacing between adjacent nodes on the ring, used to scale edge distances.
    seed_val : int, optional
        Seed passed to the Watts-Strogatz generator for reproducibility (default 42).

    Returns
    -------
    networkx.Graph
        The small-world graph with unit node weights and distance-based edge weights.

    """
    # Generate a small world network using Watts-Strogatz
    G = nx.watts_strogatz_graph(n, k, p, seed=seed_val)

    # Set the node weights to unity for curvature computations
    nx.set_node_attributes(G, values=1.0, name="weight")

    # Maximum distance between nodes = max(d_ij) + ε
    # If nodes are spaced ε apart on the ring, then
    # max(d_ij) = ⌊n/2⌋ * ε even if n is odd
    Dmax = (np.floor(n / 2) + 1) * ε

    for ii, jj in G.edges:
        # Distance between nodes is the shortest path around the ring
        d_ij = min(np.abs(ii - jj), n - np.abs(ii - jj))
        # abs() returns an np.float, get the regular float from it
        G[ii][jj]["weight"] = (Dmax - d_ij).item()

    return G


def gen_tv_weighted_sw(
    n: int, k: int, ε: float, trez: int, minpow: float | int, maxpow: float | int, seed_val: int = 42
):
    """
    Generate a time-varying series of distance-weighted small-world graphs.

    Sweeps the Watts-Strogatz rewiring probability over ``trez`` points spaced
    logarithmically between ``10 ** minpow`` and ``10 ** maxpow``, building a
    distance-weighted small-world graph (via :func:`gen_weighted_sw`) at each point.

    Parameters
    ----------
    n : int
        Number of nodes in each graph.
    k : int
        Each node is joined to its ``k`` nearest neighbors in the ring.
    ε : float
        Node spacing on the ring, used to scale edge distances.
    trez : int
        Number of probability (time) points in the series.
    minpow : float | int
        Base-10 exponent of the smallest rewiring probability.
    maxpow : float | int
        Base-10 exponent of the largest rewiring probability.
    seed_val : int, optional
        Seed passed to the underlying generator for reproducibility (default 42).

    Returns
    -------
    tuple[numpy.ndarray, list[networkx.Graph]]
        The array of rewiring probabilities ``pt`` and the list ``Gt`` of weighted
        graphs, one per probability point.

    """
    # "Time" / probability points for simulation
    pt = np.logspace(minpow, maxpow, trez)

    # Initialize empty list for graphs
    Gt: list[nx.Graph] = []

    # Simulate
    for t in range(trez):
        Gt.append(gen_weighted_sw(n, k, pt[t], ε, seed_val=seed_val))

    # Return time series of graphs
    return pt, Gt


def gen_tv_sw(n: int, k: int, trez: int, minpow: float | int, maxpow: float | int, seed_val: int = 42):
    """
    Generate a time-varying series of unweighted small-world graphs.

    Sweeps the Watts-Strogatz rewiring probability over ``trez`` points spaced
    logarithmically between ``10 ** minpow`` and ``10 ** maxpow``, building a plain
    (unweighted) Watts-Strogatz graph at each point.

    Parameters
    ----------
    n : int
        Number of nodes in each graph.
    k : int
        Each node is joined to its ``k`` nearest neighbors in the ring.
    trez : int
        Number of probability (time) points in the series.
    minpow : float | int
        Base-10 exponent of the smallest rewiring probability.
    maxpow : float | int
        Base-10 exponent of the largest rewiring probability.
    seed_val : int, optional
        Seed passed to the Watts-Strogatz generator for reproducibility (default 42).

    Returns
    -------
    tuple[numpy.ndarray, list[networkx.Graph]]
        The array of rewiring probabilities ``pt`` and the list ``Gt`` of unweighted
        graphs, one per probability point.

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
    """
    Generate the fixed-parameter unweighted small-world series preset.

    Convenience wrapper around :func:`gen_tv_sw` with the preset parameters
    ``n=1000``, ``k=50``, ``trez=100``, ``minpow=-4``, ``maxpow=0``.

    Parameters
    ----------
    seed_val : int, optional
        Seed passed to the generator for reproducibility (default 42).

    Returns
    -------
    tuple[numpy.ndarray, list[networkx.Graph]]
        The rewiring-probability array and the list of unweighted graphs.

    """
    return gen_tv_sw(1000, 50, 100, -4, 0, seed_val=seed_val)


def gen_neureps_wsw(seed_val: int = 42):
    """
    Generate the fixed-parameter weighted small-world series preset.

    Convenience wrapper around :func:`gen_tv_weighted_sw` with the preset parameters
    ``n=1000``, ``k=50``, ``ε=1.0``, ``trez=100``, ``minpow=-4``, ``maxpow=0``.

    Parameters
    ----------
    seed_val : int, optional
        Seed passed to the generator for reproducibility (default 42).

    Returns
    -------
    tuple[numpy.ndarray, list[networkx.Graph]]
        The rewiring-probability array and the list of distance-weighted graphs.

    """
    return gen_tv_weighted_sw(1000, 50, 1.0, 100, -4, 0, seed_val=seed_val)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
