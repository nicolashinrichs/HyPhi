"""
Tests for hyphi.simulation.simulations.

Covers:
  - load_connectome: round-trip with a tiny synthetic pickle.
  - create_virtual_partner_connectome: output shape and block structure.
  - setup_delayed_kuramoto: asserts NotImplementedError (stub).
  - run_delayed_kuramoto: asserts NotImplementedError (stub).
  - gen_weighted_sw: node/edge counts and distance-based edge weights.
  - run_ws_sweep: return shapes and array dtypes at toy scale.
"""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest
from hyphi.simulation.simulations import (
    create_virtual_partner_connectome,
    gen_weighted_sw,
    load_connectome,
    run_delayed_kuramoto,
    run_ws_sweep,
    setup_delayed_kuramoto,
)

# ---------------------------------------------------------------------------
# Named constants (PLR2004: no bare magic numbers in comparisons)
# ---------------------------------------------------------------------------
_N_SMALL = 6  # small brain region count for connectome tests
_N_NODES_SW = 10  # nodes in small-world graph smoke test
_K_SW = 4  # k-nearest-neighbor ring parameter
_EPSILON_SW = 1.0  # distance spacing for weights
_SEED = 42

# run_ws_sweep toy parameters
_SWEEP_N = 8
_SWEEP_K = 4
_SWEEP_T_REZ = 3
_SWEEP_N_REPS = 2
_SWEEP_MIN_POW = -2.0
_SWEEP_MAX_POW = 0.0
_N_QUANTILES = 5  # fixed by qvals inside run_ws_sweep
_WEIGHT_TOL = 1e-9  # tolerance for floating-point weight comparisons
_LOG_TOL = 1e-12  # tolerance for logspace boundary comparisons

# create_virtual_partner_connectome
_C_INTER_ZERO = 0.0
_C_INTER_NONZERO = 50.0

# gen_weighted_sw edge-weight math (p=0, ring, n=10, k=4, epsilon=1.0)
_DMAX_N10 = 6.0  # (floor(10/2)+1)*1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connectome_pickle(tmp_path, n: int) -> str:
    """
    Write a minimal synthetic connectome pickle and return its path.

    Format expected by load_connectome:
      (W, tract, roi_names, centers_raw, hemis_raw, areas_raw)
    """
    rng = np.random.default_rng(_SEED)
    W = rng.uniform(0.0, 1.0, (n, n))
    tract = rng.uniform(1.0, 100.0, (n, n))  # mm, will be divided by 1000
    roi_names = ["lM1", "rM1", "rV1", "lV1", "lFP", "rFP"][:n]
    centers = np.zeros((n, 3))
    hemis = [b"L"] * n
    areas = np.ones(n)

    path = tmp_path / "connectivity_data.pkl"
    with path.open("wb") as fh:
        pickle.dump((W, tract, roi_names, centers, hemis, areas), fh)
    return str(path)


# ---------------------------------------------------------------------------
# load_connectome
# ---------------------------------------------------------------------------


class TestLoadConnectome:
    """Round-trip tests for load_connectome with a synthetic pickle."""

    def test_returns_correct_shapes(self, tmp_path):
        """W, tract must be (N, N) and roi_names must have N entries."""
        path = _make_connectome_pickle(tmp_path, _N_SMALL)
        W, tract, names = load_connectome(path)
        assert W.shape == (_N_SMALL, _N_SMALL)
        assert tract.shape == (_N_SMALL, _N_SMALL)
        assert len(names) == _N_SMALL

    def test_diagonal_zeroed(self, tmp_path):
        """load_connectome must zero the diagonal of both matrices."""
        path = _make_connectome_pickle(tmp_path, _N_SMALL)
        W, tract, _ = load_connectome(path)
        assert np.all(np.diag(W) == 0.0)
        assert np.all(np.diag(tract) == 0.0)

    def test_matrix_symmetric(self, tmp_path):
        """Both returned matrices must be symmetric (symmetrised internally)."""
        path = _make_connectome_pickle(tmp_path, _N_SMALL)
        W, tract, _ = load_connectome(path)
        np.testing.assert_array_almost_equal(W, W.T)
        np.testing.assert_array_almost_equal(tract, tract.T)

    def test_tract_converted_mm_to_m(self, tmp_path):
        """Tract values must be in metres (< 1 m for input range 1-100 mm)."""
        path = _make_connectome_pickle(tmp_path, _N_SMALL)
        _, tract, _ = load_connectome(path)
        # Original synthetic tract was in mm (1-100); after /1000 all values < 1 m
        assert tract.max() < 1.0

    def test_roi_names_are_strings(self, tmp_path):
        """Byte-encoded roi_names must be decoded to plain str."""
        path = _make_connectome_pickle(tmp_path, _N_SMALL)
        _, _, names = load_connectome(path)
        assert all(isinstance(name, str) for name in names)


# ---------------------------------------------------------------------------
# create_virtual_partner_connectome
# ---------------------------------------------------------------------------


class TestCreateVirtualPartnerConnectome:
    """Output shape and block-structure tests."""

    @pytest.fixture
    def small_brain(self):
        """Return (W, D, roi_names) for a 6-region toy brain."""
        n = _N_SMALL
        rng = np.random.default_rng(_SEED)
        W = rng.uniform(0.1, 1.0, (n, n))
        np.fill_diagonal(W, 0.0)
        D = rng.uniform(0.0, 0.1, (n, n))
        np.fill_diagonal(D, 0.0)
        roi_names = ["lM1", "rM1", "rV1", "lV1", "lFP", "rFP"]
        return W, D, roi_names

    def test_output_shapes(self, small_brain):
        """Expanded matrices must be (2N, 2N)."""
        W, D, names = small_brain
        n = W.shape[0]
        W_vir, D_vir = create_virtual_partner_connectome(W, D, names, _C_INTER_ZERO)
        assert W_vir.shape == (2 * n, 2 * n)
        assert D_vir.shape == (2 * n, 2 * n)

    def test_diagonal_zeroed(self, small_brain):
        """Diagonal of the expanded matrices must be zero."""
        W, D, names = small_brain
        W_vir, D_vir = create_virtual_partner_connectome(W, D, names, _C_INTER_ZERO)
        assert np.all(np.diag(W_vir) == 0.0)
        assert np.all(np.diag(D_vir) == 0.0)

    def test_intra_blocks_match_original(self, small_brain):
        """Top-left and bottom-right blocks must reproduce the input W and D."""
        W, D, names = small_brain
        n = W.shape[0]
        W_vir, D_vir = create_virtual_partner_connectome(W, D, names, _C_INTER_ZERO)
        np.testing.assert_array_equal(W_vir[:n, :n], W)
        np.testing.assert_array_equal(W_vir[n:, n:], W)
        np.testing.assert_array_equal(D_vir[:n, :n], D)
        np.testing.assert_array_equal(D_vir[n:, n:], D)

    def test_cross_blocks_zero_when_c_inter_zero(self, small_brain):
        """With C_inter=0 the off-diagonal brain blocks must be all zeros."""
        W, D, names = small_brain
        n = W.shape[0]
        W_vir, _ = create_virtual_partner_connectome(W, D, names, _C_INTER_ZERO)
        np.testing.assert_array_equal(W_vir[:n, n:], np.zeros((n, n)))
        np.testing.assert_array_equal(W_vir[n:, :n], np.zeros((n, n)))

    def test_cross_blocks_nonzero_when_c_inter_positive(self, small_brain):
        """With C_inter > 0 at least one entry in the cross blocks must be nonzero."""
        W, D, names = small_brain
        n = W.shape[0]
        W_vir, _ = create_virtual_partner_connectome(W, D, names, _C_INTER_NONZERO)
        assert W_vir[:n, n:].max() > 0.0


class TestDelayedKuramotoStubs:
    """Both Kuramoto functions are unimplemented stubs."""

    def test_setup_delayed_kuramoto_raises(self):
        """setup_delayed_kuramoto must raise NotImplementedError."""
        rng = np.random.default_rng(_SEED)
        W = rng.uniform(0.0, 1.0, (_N_SMALL, _N_SMALL))
        tract = rng.uniform(0.0, 0.1, (_N_SMALL, _N_SMALL))
        omega = rng.uniform(0.0, 1.0, _N_SMALL)
        with pytest.raises(NotImplementedError):
            setup_delayed_kuramoto(W, tract, omega)

    def test_run_delayed_kuramoto_raises(self):
        """run_delayed_kuramoto must raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            run_delayed_kuramoto(
                solver=None,
                dt=0.01,
                t_max=1.0,
                n_osc=_N_SMALL,
            )


# ---------------------------------------------------------------------------
# gen_weighted_sw
# ---------------------------------------------------------------------------


class TestGenWeightedSW:
    """Node/edge counts and edge-weight formula for gen_weighted_sw."""

    def test_node_count(self):
        """Graph must have exactly n nodes."""
        G = gen_weighted_sw(_N_NODES_SW, _K_SW, 0.0, _EPSILON_SW, seed=_SEED)
        assert G.number_of_nodes() == _N_NODES_SW

    def test_edge_count_no_rewiring(self):
        """With p=0, WS graph has n*k/2 edges (ring lattice with even k)."""
        expected_edges = (_N_NODES_SW * _K_SW) // 2
        G = gen_weighted_sw(_N_NODES_SW, _K_SW, 0.0, _EPSILON_SW, seed=_SEED)
        assert G.number_of_edges() == expected_edges

    def test_all_edges_have_weight(self):
        """Every edge must carry a 'weight' attribute after generation."""
        G = gen_weighted_sw(_N_NODES_SW, _K_SW, 0.3, _EPSILON_SW, seed=_SEED)
        weights = nx.get_edge_attributes(G, "weight")
        assert len(weights) == G.number_of_edges()

    def test_edge_weights_positive(self):
        """All edge weights must be strictly positive (Dmax > ring distance)."""
        G = gen_weighted_sw(_N_NODES_SW, _K_SW, 0.0, _EPSILON_SW, seed=_SEED)
        weights = nx.get_edge_attributes(G, "weight")
        assert all(w > 0.0 for w in weights.values())

    def test_edge_weight_formula_no_rewiring(self):
        """With p=0 and known params, verify Dmax - d_ij for two ring distances."""
        # n=10, k=4, epsilon=1 -> Dmax = (floor(10/2)+1)*1 = 6
        G = gen_weighted_sw(_N_NODES_SW, _K_SW, 0.0, _EPSILON_SW, seed=_SEED)
        weights = nx.get_edge_attributes(G, "weight")
        for (i, j), w in weights.items():
            d_ij = min(abs(i - j), _N_NODES_SW - abs(i - j))
            expected = _DMAX_N10 - d_ij
            assert abs(w - expected) < _WEIGHT_TOL, f"edge ({i},{j}): got {w}, expected {expected}"

    def test_reproducible_with_seed(self):
        """Same seed must produce identical graphs."""
        g1 = gen_weighted_sw(_N_NODES_SW, _K_SW, 0.5, _EPSILON_SW, seed=_SEED)
        g2 = gen_weighted_sw(_N_NODES_SW, _K_SW, 0.5, _EPSILON_SW, seed=_SEED)
        assert set(g1.edges()) == set(g2.edges())


# ---------------------------------------------------------------------------
# run_ws_sweep
# ---------------------------------------------------------------------------


class TestRunWsSweep:
    """Return structure of run_ws_sweep at toy scale."""

    @pytest.fixture(scope="class")
    def sweep_result(self):
        """Run a tiny sweep once and cache the result for all tests in this class."""
        return run_ws_sweep(
            n=_SWEEP_N,
            k=_SWEEP_K,
            epsilon=_EPSILON_SW,
            t_rez=_SWEEP_T_REZ,
            min_pow=_SWEEP_MIN_POW,
            max_pow=_SWEEP_MAX_POW,
            n_reps=_SWEEP_N_REPS,
            seed_base=0,
        )

    def test_returns_three_arrays(self, sweep_result):
        """run_ws_sweep must return a 3-tuple."""
        assert len(sweep_result) == 3  # noqa: PLR2004 -- structural, not magic

    def test_pt_shape(self, sweep_result):
        """Probability grid (pt) must have shape (t_rez,)."""
        pt, _, _ = sweep_result
        assert pt.shape == (_SWEEP_T_REZ,)

    def test_hreps_shape(self, sweep_result):
        """Hreps must have shape (n_reps, t_rez)."""
        _, Hreps, _ = sweep_result
        assert Hreps.shape == (_SWEEP_N_REPS, _SWEEP_T_REZ)

    def test_qreps_shape(self, sweep_result):
        """Qreps must have shape (n_reps, t_rez, 5) — five fixed quantile levels."""
        _, _, Qreps = sweep_result
        assert Qreps.shape == (_SWEEP_N_REPS, _SWEEP_T_REZ, _N_QUANTILES)

    def test_pt_is_logspaced(self, sweep_result):
        """Probability grid must span from 10**min_pow to 10**max_pow on a log scale."""
        pt, _, _ = sweep_result
        assert abs(pt[0] - 10**_SWEEP_MIN_POW) < _LOG_TOL
        assert abs(pt[-1] - 10**_SWEEP_MAX_POW) < _LOG_TOL

    def test_pt_dtype_float(self, sweep_result):
        """Probability grid must be a floating-point array."""
        pt, _, _ = sweep_result
        assert np.issubdtype(pt.dtype, np.floating)

    def test_hreps_finite(self, sweep_result):
        """Hreps must contain only finite values (no NaN from failed entropy)."""
        _, Hreps, _ = sweep_result
        assert np.all(np.isfinite(Hreps))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
