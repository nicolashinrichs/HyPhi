# ============= #
# Preliminaries #
# ============= #

import numpy as np
import networkx as nx

# ================ #
# Small World Sims #
# ================ #

def genWeightedSW(n: int, k: int, p: float, ε: float, seed_val: int = 42):
    # Generate a small world network using Watts-Strogatz
    G = nx.watts_strogatz_graph(n, k, p, seed=seed_val)

    # Set the node weights to unity for curvature computations
    nx.set_node_attributes(G, values=1.0, name="weight")

    # Maximum distance between nodes = max(d_ij) + ε
    # If nodes are spaced ε apart on the ring, then
    # max(d_ij) = ⌊n/2⌋ * ε even if n is odd
    Dmax = (np.floor(n/2) + 1) * ε

    for ii, jj in G.edges:
        # Distance between nodes is the shortest path around the ring
        d_ij = min(np.abs(ii - jj), n - np.abs(ii - jj))
        # abs() returns an np.float, get the regular float from it
        G[ii][jj]["weight"] = (Dmax - d_ij).item() 
        
    return G


def genTVWeightedSW(n: int, k: int, ε: float, trez: int, minpow: float | int, maxpow: float | int, seed_val: int = 42):
    # "Time" / probability points for simulation
    pt = np.logspace(minpow, maxpow, trez)

    # Initialize empty list for graphs
    Gt = []
    
    # Simulate 
    for t in range(trez):
        Gt.append(genWeightedSW(n, k, pt[t], ε, seed_val=seed_val))

    # Return time series of graphs
    return pt, Gt


def genTVSW(n: int, k: int, trez: int, minpow: float | int, maxpow: float | int, seed_val: int = 42):
    # "Time" / probability points for simulation
    pt = np.logspace(minpow, maxpow, trez)

    # Initialize empty list for graphs
    Gt = []
    
    # Simulate 
    for t in range(trez):
        Gt.append(nx.watts_strogatz_graph(n, k, pt[t], seed=seed_val))

    # Return time series of graphs
    return pt, Gt


def genNatureSW(seed_val: int = 42):
    return genTVSW(1000, 50, 100, -4, 0, seed_val=seed_val)


def genNeuRepsWSW(seed_val: int = 42):
    return genTVWeightedSW(1000, 50, 1., 100, -4, 0, seed_val=seed_val)