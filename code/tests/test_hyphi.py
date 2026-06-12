"""Smoke tests for the density-estimation and the simulation-to-entropy pipeline."""

# %% Import
import numpy as np
import pytest
from hyphi.modeling.density_estimation import fit_kde
from hyphi.modeling.entropies import entropy_kde_plugin, vec_entropy
from hyphi.modeling.graph_curvatures import compute_frc_vec
from hyphi.simulation.graph_simulations import gen_tv_sw
from scipy.stats import norm

# %% Test Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

MODE_TOLERANCE = 0.5  # how far the estimated mode of N(0, 1) may sit from 0


def test_kde_recovers_standard_normal():
    """KDE of N(0, 1) samples is a valid density: non-negative, integrates to 1, mode near 0."""
    data = norm(loc=0, scale=1).rvs(2000, random_state=0)

    # Sheather-Jones (ISJ) bandwidth, and bw=1 which is optimal here since the true std is 1.
    x_isj, y_isj = fit_kde(data, bw="ISJ", method="tree")()
    x_truth, y_truth = fit_kde(data, bw=1, method="tree")()

    for x, y in ((x_isj, y_isj), (x_truth, y_truth)):
        assert len(x) == len(y)
        assert np.all(y >= 0.0)
        # A density integrates to 1 over its support (numpy is pinned < 2.0, so np.trapz).
        assert np.trapz(y, x) == pytest.approx(1.0, abs=0.05)  # noqa: NPY201
        # N(0, 1) has its mode at 0.
        assert abs(x[np.argmax(y)]) < MODE_TOLERANCE


def test_small_world_curvature_entropy_pipeline():
    """A time-varying small-world simulation flows to a finite curvature-entropy series."""
    # Mirror the Nature small-world setup (gen_nature_sw) but on a small graph that runs in
    # seconds, with rewiring probabilities in [0.1, 1.0] so the curvature distribution stays
    # non-degenerate (a near-regular lattice gives a single curvature value and is the job of
    # the degenerate-input safeguard, not of this smoke test).
    trez = 4
    _rewiring, graphs = gen_tv_sw(n=120, k=6, trez=trez, minpow=-1, maxpow=0)
    assert len(graphs) == trez

    frc_graphs = compute_frc_vec(graphs)
    assert len(frc_graphs) == trez

    entropies = vec_entropy(frc_graphs, entropy_kde_plugin)
    assert len(entropies) == trez
    assert np.all(np.isfinite(entropies))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
