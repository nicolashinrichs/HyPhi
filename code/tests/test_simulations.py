"""
Tests for the connectome-delayed Kuramoto ground truth.

Covers the end-to-end smoke path plus regressions for the reconciled physics: zero-delay (inter-brain)
coupling must actually couple, the order parameter is per-brain for a dyad, and the additive noise is
applied and reproducible. The delayed Kuramoto is built on jitcdde + symengine + sympy (all declared
dependencies); the jitcdde/symengine importorskips are belt-and-suspenders for a partial install.
"""

# %% Import
import numpy as np
import pytest

# Importing the functions is safe without the jitcdde/symengine stack: that import is lazy (inside the
# functions), so it loads only when a test actually runs the solver, which the fixture below guards.
from hyphi.simulation.simulations import run_delayed_kuramoto, setup_delayed_kuramoto

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _small_connectome(n=4, seed=0):
    """Build a tiny synthetic connectome: symmetric weights, tract lengths (m), and frequencies."""
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, n)))
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    tract = np.abs(rng.standard_normal((n, n))) * 0.05 + 0.02  # roughly 2 to 7 cm
    tract = (tract + tract.T) / 2.0
    np.fill_diagonal(tract, 0.0)
    omega = 2.0 * np.pi * (8.0 + rng.standard_normal(n))  # around 8 Hz
    return w, tract, omega


def _dual_brain(omega_a, omega_b, intra_w=1.0, intra_tract=0.01):
    """Two 2-oscillator brains: intra-brain edges (delayed), no inter-brain link unless added by caller."""
    w = np.zeros((4, 4))
    tract = np.zeros((4, 4))
    w[0, 1] = w[1, 0] = intra_w
    w[2, 3] = w[3, 2] = intra_w
    tract[0, 1] = tract[1, 0] = intra_tract
    tract[2, 3] = tract[3, 2] = intra_tract
    omega = 2.0 * np.pi * np.array([omega_a, omega_a, omega_b, omega_b])
    return w, tract, omega


@pytest.fixture(autouse=True)
def _require_solver_deps():
    """Skip cleanly when the jitcdde/symengine/sympy compile stack is absent."""
    pytest.importorskip("jitcdde")
    pytest.importorskip("symengine")
    pytest.importorskip("sympy")


class TestDelayedKuramoto:
    """run_delayed_kuramoto produces finite phases and a per-step order parameter on a small connectome."""

    def test_smoke(self):
        """A short single-brain run returns aligned (times, phases, order parameters) in range."""
        n = 4
        w, tract, omega = _small_connectome(n=n)
        solver = setup_delayed_kuramoto(w, tract, omega, noise_strength=0.0, seed=1)
        times, theta, order = run_delayed_kuramoto(solver, dt=0.01, t_max=0.5, n_osc=n, t_skip=0.1)

        assert times.shape[0] == theta.shape[0] == order.shape[0]
        assert theta.shape[1] == n
        assert np.all(np.isfinite(theta))
        assert np.all(theta >= 0.0)
        assert np.all(theta < 2.0 * np.pi + 1e-6)
        assert np.all(order >= 0.0)
        assert np.all(order <= 1.0 + 1e-9)

    def test_zero_delay_coupling_synchronizes(self):
        """A zero-delay link must still couple: strong instantaneous coupling phase-locks a detuned pair."""
        w = np.array([[0.0, 1.0], [1.0, 0.0]])
        tract = np.zeros((2, 2))  # zero delay -> instantaneous coupling
        omega = 2.0 * np.pi * np.array([10.0, 10.5])  # detuned: they would beat if uncoupled
        solver = setup_delayed_kuramoto(w, tract, omega, c_intra=8.0, noise_strength=0.0, seed=3)
        _, _, order = run_delayed_kuramoto(solver, dt=0.01, t_max=3.0, n_osc=2, t_skip=1.5)
        tail = order[-20:]
        locked_r_min, steady_std_max = 0.9, 0.05
        assert tail.mean() > locked_r_min  # phase-locked
        assert tail.std() < steady_std_max  # steady (not the beating of an uncoupled pair)

    def test_per_brain_order_parameter(self):
        """Per-brain (rA+rB)/2 exceeds the global r for two synchronized brains at different frequencies."""
        w, tract, omega = _dual_brain(omega_a=10.0, omega_b=16.0)
        common = {"c_intra": 8.0, "noise_strength": 0.0, "seed": 5}
        solver_per = setup_delayed_kuramoto(w, tract, omega, **common)
        _, theta_per, r_per = run_delayed_kuramoto(solver_per, dt=0.01, t_max=2.0, n_osc=4, t_skip=1.0, n_per_brain=2)
        solver_glob = setup_delayed_kuramoto(w, tract, omega, **common)
        _, theta_glob, r_glob = run_delayed_kuramoto(solver_glob, dt=0.01, t_max=2.0, n_osc=4, t_skip=1.0)

        np.testing.assert_allclose(theta_per, theta_glob)  # same dynamics, only the metric differs
        sync_min, global_margin = 0.9, 0.1
        assert r_per.mean() > sync_min  # each brain is internally synchronized
        assert r_per.mean() > r_glob.mean() + global_margin  # global conflates the two out-of-phase brains

    def test_noise_is_applied_and_reproducible(self):
        """Nonzero noise changes the trajectory versus noiseless, and is reproducible for a fixed seed."""
        w = np.array([[0.0, 1.0], [1.0, 0.0]])
        tract = np.array([[0.0, 0.05], [0.05, 0.0]])
        omega = 2.0 * np.pi * np.array([10.0, 10.5])
        run_kw = {"dt": 0.01, "t_max": 1.0, "n_osc": 2, "t_skip": 0.2}

        solver_clean = setup_delayed_kuramoto(w, tract, omega, noise_strength=0.0, seed=7)
        _, theta_clean, _ = run_delayed_kuramoto(solver_clean, **run_kw)
        solver_noisy_a = setup_delayed_kuramoto(w, tract, omega, noise_strength=0.5, seed=7)
        _, theta_noisy_a, _ = run_delayed_kuramoto(solver_noisy_a, **run_kw)
        solver_noisy_b = setup_delayed_kuramoto(w, tract, omega, noise_strength=0.5, seed=7)
        _, theta_noisy_b, _ = run_delayed_kuramoto(solver_noisy_b, **run_kw)

        np.testing.assert_allclose(theta_noisy_a, theta_noisy_b)  # same seed -> reproducible
        assert not np.allclose(theta_noisy_a, theta_clean)  # noise actually perturbs the read-out


class TestDelayedKuramotoValidation:
    """setup/run reject malformed input loudly rather than producing silent garbage."""

    def test_rejects_nonpositive_velocity(self):
        """A non-positive velocity (which would null or invert all delays) is rejected."""
        w, tract, omega = _small_connectome(n=3)
        with pytest.raises(ValueError, match="velocity must be positive"):
            setup_delayed_kuramoto(w, tract, omega, velocity=0.0)

    def test_rejects_nonfinite_input(self):
        """inf/NaN in the connectome is rejected (it would silently propagate to all-NaN phases)."""
        w, tract, omega = _small_connectome(n=3)
        w[0, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            setup_delayed_kuramoto(w, tract, omega)

    def test_rejects_asymmetric_weights(self):
        """A non-symmetric W is rejected (spectral normalization assumes symmetry)."""
        w, tract, omega = _small_connectome(n=3)
        w[0, 1] += 5.0  # break symmetry
        with pytest.raises(ValueError, match="symmetric"):
            setup_delayed_kuramoto(w, tract, omega)

    def test_rejects_mismatched_split(self):
        """n_per_brain that does not halve n_osc is rejected."""
        w, tract, omega = _small_connectome(n=4)
        solver = setup_delayed_kuramoto(w, tract, omega, noise_strength=0.0, seed=1)
        with pytest.raises(ValueError, match="half of n_osc"):
            run_delayed_kuramoto(solver, dt=0.01, t_max=0.5, n_osc=4, t_skip=0.1, n_per_brain=3)

    def test_rejects_tskip_past_horizon(self):
        """A t_skip that exceeds the simulated horizon (which would return empty arrays) is rejected."""
        w, tract, omega = _small_connectome(n=3)
        solver = setup_delayed_kuramoto(w, tract, omega, noise_strength=0.0, seed=1)
        with pytest.raises(ValueError, match="horizon"):
            run_delayed_kuramoto(solver, dt=0.01, t_max=0.5, n_osc=3, t_skip=2.0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
