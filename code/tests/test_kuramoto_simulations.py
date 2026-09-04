"""
Known-Answer and Smoke Tests for Kuramoto Oscillator Simulations.

Covers:
  - kuramoto_vector_field: shape and zero-coupling special case.
  - rk4: one step on a constant-velocity field advances by exactly const*dt.
  - simulate_kuramoto: output shape is (n_steps + 1, n_osc).
  - get_plv_pair: PLV of two identical phase series is 1.0.
  - get_plv_matrix: diagonal is 1.0; off-diagonal is 1.0 for identical rows.
  - get_plv_graphs: returns the expected number of NetworkX graphs.
  - get_avg_plv: length matches the graph list; values in [0, 1].
  - kuramoto_test_sim: NOT called here (100 osc x 2000 steps - too slow).
"""

# %% Import
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from hyphi.simulation.kuramoto_simulations import (
    get_avg_plv,
    get_plv_graphs,
    get_plv_matrix,
    get_plv_pair,
    kuramoto_vector_field,
    rk4,
    simulate_kuramoto,
)
from jax import random

# %% Named constants (avoids PLR2004 magic-value lint errors) >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

_N_OSC = 4  # number of oscillators for toy simulations
_N_STEPS = 8  # number of integration steps
_DT = 0.1  # time step
_T_WIN = 5  # window length in samples
_T_STRIDE = 2  # window stride in samples
_PLV_ATOL = 1e-5  # absolute tolerance for PLV comparisons
_SHAPE_ROWS = _N_STEPS + 1  # expected rows in theta_hist
_SEED_OMEGAS = 42  # JAX PRNGKey seed for natural frequencies
_SEED_INIT = 7  # JAX PRNGKey seed for initial phases
_COUPLING_ZERO = 0.0
_COUPLING_UNIT = 1.0
_T_SERIES_LEN = 20  # length of phase time-series used in PLV unit tests

# %% Helpers >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _make_init_state(n: int = _N_OSC, seed: int = _SEED_INIT) -> jnp.ndarray:
    """Return a reproducible random initial phase vector of length n."""
    return random.uniform(random.PRNGKey(seed), (n,), maxval=2 * jnp.pi)


def _make_omegas(n: int = _N_OSC, seed: int = _SEED_OMEGAS) -> jnp.ndarray:
    """Return reproducible random natural frequencies of length n."""
    return random.normal(random.PRNGKey(seed), (n,))


# %% Tests >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestKuramotoVectorField:
    """Shape and analytical-limit tests for the Kuramoto vector field."""

    def test_output_shape(self):
        """The derivative array must have the same shape as the input phases."""
        thetas = _make_init_state()
        omegas = _make_omegas()
        dtheta = kuramoto_vector_field(thetas, _COUPLING_UNIT, omegas)
        assert dtheta.shape == thetas.shape

    def test_zero_coupling_returns_natural_frequencies(self):
        """With K=0 the mean-field term vanishes; d(theta)/dt == omegas exactly."""
        thetas = _make_init_state()
        omegas = _make_omegas()
        dtheta = kuramoto_vector_field(thetas, _COUPLING_ZERO, omegas)
        np.testing.assert_allclose(np.asarray(dtheta), np.asarray(omegas), atol=_PLV_ATOL)


class TestRk4:
    """One-step accuracy tests for the RK4 integrator."""

    def test_constant_field_advances_by_const_times_dt(self):
        """RK4 on d(x)/dt = c must advance the state by exactly c * dt.

        For a constant vector field f(x) = c all four RK4 slopes equal c,
        so the update collapses to x_new = x + c * dt -- exact regardless of dt.
        """
        c = jnp.array([1.0, -2.0, 0.5, 3.0])
        x0 = jnp.zeros(_N_OSC)
        x1 = rk4(lambda _: c, x0, _DT)
        expected = x0 + c * _DT
        np.testing.assert_allclose(np.asarray(x1), np.asarray(expected), atol=_PLV_ATOL)

    def test_zero_field_leaves_state_unchanged(self):
        """RK4 on d(x)/dt = 0 must leave the state unchanged."""
        x0 = _make_init_state()
        x1 = rk4(jnp.zeros_like, x0, _DT)
        np.testing.assert_allclose(np.asarray(x1), np.asarray(x0), atol=_PLV_ATOL)


class TestSimulateKuramoto:
    """Output-shape and initial-condition tests for simulate_kuramoto."""

    def _run_tiny(self):
        """Run a tiny Kuramoto simulation and return theta_hist."""
        omegas = _make_omegas()
        init = _make_init_state()
        func = lambda th: kuramoto_vector_field(th, _COUPLING_UNIT, omegas)  # noqa: E731
        return simulate_kuramoto(func, rk4, init, _DT, _N_STEPS)

    def test_output_shape(self):
        """theta_hist must have shape (n_steps + 1, n_osc)."""
        hist = self._run_tiny()
        assert hist.shape == (_SHAPE_ROWS, _N_OSC)

    def test_initial_state_preserved(self):
        """Row 0 of theta_hist must equal the supplied initial state exactly."""
        omegas = _make_omegas()
        init = _make_init_state()
        func = lambda th: kuramoto_vector_field(th, _COUPLING_UNIT, omegas)  # noqa: E731
        hist = simulate_kuramoto(func, rk4, init, _DT, _N_STEPS)
        np.testing.assert_allclose(np.asarray(hist[0]), np.asarray(init), atol=_PLV_ATOL)

    def test_frozen_oscillators_stay_constant(self):
        """With K=0 and omegas=0, phases must remain constant throughout."""
        omegas = jnp.zeros(_N_OSC)
        init = _make_init_state()
        func = lambda th: kuramoto_vector_field(th, _COUPLING_ZERO, omegas)  # noqa: E731
        hist = simulate_kuramoto(func, rk4, init, _DT, _N_STEPS)
        # every row should equal the initial state
        for row_idx in range(_SHAPE_ROWS):
            np.testing.assert_allclose(np.asarray(hist[row_idx]), np.asarray(init), atol=_PLV_ATOL)


class TestGetPlvPair:
    """Known-answer tests for the pair-wise PLV computation."""

    def test_identical_signals_plv_is_one(self):
        """PLV of a signal with itself must be exactly 1.0."""
        phi = jnp.linspace(0, 2 * jnp.pi, _T_SERIES_LEN)
        plv = get_plv_pair(phi, phi)
        assert float(plv) == pytest.approx(1.0, abs=_PLV_ATOL)

    def test_opposite_phase_plv_is_one(self):
        """PLV of phi and phi + pi must still be 1.0 (constant phase difference)."""
        phi = jnp.linspace(0, 2 * jnp.pi, _T_SERIES_LEN)
        plv = get_plv_pair(phi, phi + jnp.pi)
        assert float(plv) == pytest.approx(1.0, abs=_PLV_ATOL)

    def test_plv_in_unit_interval(self):
        """PLV must lie in [0, 1] for arbitrary random inputs."""
        rng = np.random.default_rng(0)
        phi_i = jnp.array(rng.uniform(0, 2 * np.pi, _T_SERIES_LEN))
        phi_j = jnp.array(rng.uniform(0, 2 * np.pi, _T_SERIES_LEN))
        plv = float(get_plv_pair(phi_i, phi_j))
        assert 0.0 <= plv <= 1.0 + _PLV_ATOL


class TestGetPlvMatrix:
    """Matrix-level PLV tests: shape, symmetry, diagonal, and identical-row case."""

    def _identical_phase_window(self, n: int = _N_OSC, t: int = _T_SERIES_LEN):
        """Return an (n, t) phase window where every row is the same signal."""
        phi = jnp.linspace(0, 2 * jnp.pi, t)
        return jnp.stack([phi] * n)

    def test_output_shape(self):
        """PLV matrix must be square (N, N)."""
        window = self._identical_phase_window()
        c = get_plv_matrix(window)
        assert c.shape == (_N_OSC, _N_OSC)

    def test_diagonal_is_one(self):
        """Every oscillator has PLV 1.0 with itself."""
        window = self._identical_phase_window()
        c = np.asarray(get_plv_matrix(window))
        np.testing.assert_allclose(np.diag(c), 1.0, atol=_PLV_ATOL)

    def test_all_ones_for_identical_rows(self):
        """When all oscillators share the same phase, the PLV matrix is all-ones."""
        window = self._identical_phase_window()
        c = np.asarray(get_plv_matrix(window))
        np.testing.assert_allclose(c, 1.0, atol=_PLV_ATOL)

    def test_symmetry(self):
        """PLV matrix must be symmetric (c_ij == c_ji)."""
        rng = np.random.default_rng(1)
        window = jnp.array(rng.uniform(0, 2 * np.pi, (_N_OSC, _T_SERIES_LEN)))
        c = np.asarray(get_plv_matrix(window))
        np.testing.assert_allclose(c, c.T, atol=_PLV_ATOL)


class TestGetPlvGraphs:
    """Tests for the sliding-window graph builder."""

    def _build_theta_hist(self) -> jnp.ndarray:
        """Simulate a tiny trajectory used across graph-builder tests."""
        omegas = _make_omegas()
        init = _make_init_state()
        func = lambda th: kuramoto_vector_field(th, _COUPLING_UNIT, omegas)  # noqa: E731
        return simulate_kuramoto(func, rk4, init, _DT, _N_STEPS)

    def test_returns_list_of_nx_graphs(self):
        """Each element of the output must be a networkx Graph."""
        hist = self._build_theta_hist()
        graphs = get_plv_graphs(_N_STEPS, _T_WIN, _T_STRIDE, hist)
        assert isinstance(graphs, list)
        for g in graphs:
            assert isinstance(g, nx.Graph)

    def test_correct_number_of_windows(self):
        """Number of graphs equals the number of valid sliding windows.

        The range is ``range(0, n_steps - w_size + 1, w_stride)``.
        """
        hist = self._build_theta_hist()
        graphs = get_plv_graphs(_N_STEPS, _T_WIN, _T_STRIDE, hist)
        expected = len(range(0, _N_STEPS - _T_WIN + 1, _T_STRIDE))
        assert len(graphs) == expected

    def test_graph_nodes_match_oscillator_count(self):
        """Each graph must have exactly N_OSC nodes."""
        hist = self._build_theta_hist()
        graphs = get_plv_graphs(_N_STEPS, _T_WIN, _T_STRIDE, hist)
        for g in graphs:
            assert g.number_of_nodes() == _N_OSC

    def test_edge_weights_in_unit_interval(self):
        """All PLV edge weights must lie in [0, 1]."""
        hist = self._build_theta_hist()
        graphs = get_plv_graphs(_N_STEPS, _T_WIN, _T_STRIDE, hist)
        for g in graphs:
            for _, _, d in g.edges(data=True):
                w = d["weight"]
                assert 0.0 - _PLV_ATOL <= w <= 1.0 + _PLV_ATOL


class TestGetAvgPlv:
    """Tests for the per-window mean-PLV summary."""

    def _build_graphs(self) -> list:
        """Return a short list of PLV graphs from a tiny simulation."""
        omegas = _make_omegas()
        init = _make_init_state()
        func = lambda th: kuramoto_vector_field(th, _COUPLING_UNIT, omegas)  # noqa: E731
        hist = simulate_kuramoto(func, rk4, init, _DT, _N_STEPS)
        return get_plv_graphs(_N_STEPS, _T_WIN, _T_STRIDE, hist)

    def test_output_length_matches_graph_list(self):
        """get_avg_plv must return one value per graph."""
        graphs = self._build_graphs()
        avg = get_avg_plv(graphs)
        assert len(avg) == len(graphs)

    def test_values_in_unit_interval(self):
        """Every mean PLV must lie in [0, 1]."""
        graphs = self._build_graphs()
        avg = get_avg_plv(graphs)
        assert np.all(avg >= 0.0 - _PLV_ATOL)
        assert np.all(avg <= 1.0 + _PLV_ATOL)

    def test_identical_oscillators_avg_plv_is_one(self):
        """When all oscillators are phase-locked, mean PLV must be 1.0 per window."""
        # Build a trajectory where all oscillators share the same phase at every step.
        # Use omegas=0 and K=0 so phases stay constant at init_state[0] for all osc.
        omegas = jnp.zeros(_N_OSC)
        # All start at the same phase value (0.0)
        init = jnp.zeros(_N_OSC)
        func = lambda th: kuramoto_vector_field(th, _COUPLING_ZERO, omegas)  # noqa: E731
        hist = simulate_kuramoto(func, rk4, init, _DT, _N_STEPS)
        graphs = get_plv_graphs(_N_STEPS, _T_WIN, _T_STRIDE, hist)
        avg = get_avg_plv(graphs)
        np.testing.assert_allclose(avg, 1.0, atol=_PLV_ATOL)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
