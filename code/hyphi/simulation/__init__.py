"""
Public API for the simulation subpackage: Kuramoto dynamics, small-world graphs, ground truth.

These are the ground-truth generators HyPhi uses to validate the curvature-entropy detector
against a known synchronization transition. ``gen_weighted_sw`` is re-exported from
``graph_simulations`` (the module that also carries the time-varying generators); the connectome
delayed-Kuramoto functions are public API stubs that the ground-truth baseline work fills in.
"""

# %% Imports
from .graph_simulations import (
    gen_nature_sw,
    gen_neureps_wsw,
    gen_tv_sw,
    gen_tv_weighted_sw,
    gen_weighted_sw,
)
from .kuramoto_simulations import (
    get_avg_plv,
    get_plv_graphs,
    get_plv_matrix,
    kuramoto_test_sim,
    simulate_kuramoto,
)
from .simulations import (
    create_virtual_partner_connectome,
    load_connectome,
    run_delayed_kuramoto,
    run_ws_sweep,
    setup_delayed_kuramoto,
)

__all__ = [
    "create_virtual_partner_connectome",
    "gen_nature_sw",
    "gen_neureps_wsw",
    "gen_tv_sw",
    "gen_tv_weighted_sw",
    # Small-world graph generators
    "gen_weighted_sw",
    "get_avg_plv",
    "get_plv_graphs",
    "get_plv_matrix",
    "kuramoto_test_sim",
    # Connectome and sweeps (ground truth)
    "load_connectome",
    "run_delayed_kuramoto",
    "run_ws_sweep",
    "setup_delayed_kuramoto",
    # Kuramoto oscillator dynamics
    "simulate_kuramoto",
]
