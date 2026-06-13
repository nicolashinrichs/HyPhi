"""End-to-end smoke tests: density estimation and the FRC-entropy pipeline."""

# %% Import
import numpy as np
import pytest
from hyphi.modeling.density_estimation import fit_kde
from hyphi.modeling.entropies import entropy_kde_plugin, vec_entropy
from hyphi.modeling.graph_curvatures import compute_frc_vec, extract_curvatures
from hyphi.simulation.graph_simulations import gen_tv_sw
from scipy.stats import norm

# %% Test functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@pytest.fixture(scope="module")
def gaussian_sample():
    """1000 draws from a standard normal (its optimal KDE bandwidth is ~1)."""
    return norm(loc=0, scale=1).rvs(1000, random_state=0)


def test_fit_kde_isj_recovers_normal_density(gaussian_sample):
    """ISJ bandwidth selection recovers the true density, beating an oversmoothed bw."""
    # The optimal bandwidth for n=1000 standard-normal draws is ~0.27, not the
    # data's standard deviation, so bw=1 deliberately oversmooths. Grids differ
    # per bandwidth, so each estimate is compared to the true pdf on its own grid.
    x_isj, y_isj = fit_kde(gaussian_sample, bw="ISJ", method="tree")()
    x_wide, y_wide = fit_kde(gaussian_sample, bw=1, method="tree")()

    assert np.all(np.isfinite(y_isj))
    assert np.all(y_isj >= 0)
    assert np.trapz(y_isj, x_isj) == pytest.approx(1.0, abs=0.01)  # noqa: NPY201 (numpy is pinned <2.0)
    assert np.trapz(y_wide, x_wide) == pytest.approx(1.0, abs=0.01)  # noqa: NPY201 (numpy is pinned <2.0)
    err_isj = np.max(np.abs(y_isj - norm.pdf(x_isj)))
    err_wide = np.max(np.abs(y_wide - norm.pdf(x_wide)))
    assert err_isj < 0.1  # noqa: PLR2004 (tolerance)
    assert err_isj < err_wide


def test_sw_frc_entropy_pipeline():
    """The small-world graphs -> FRC -> entropy chain runs end to end at toy scale."""
    # Same chain as the Nature methods pipeline; gen_nature_sw() runs it at
    # full size (1000 nodes, 100 graphs) in minutes. Rewiring probabilities in
    # [0.1, 1]: near-lattice graphs (p ~ 1e-4) have almost constant curvature,
    # where the ISJ bandwidth solver is known to fail (the degenerate-entropy
    # safeguard is tracked in issue #28).
    pt_nat, graphs = gen_tv_sw(100, 10, 10, -1, 0)
    assert len(graphs) == len(pt_nat)

    frc_graphs = compute_frc_vec(graphs)
    assert len(frc_graphs) == len(graphs)

    entropies = vec_entropy(frc_graphs, entropy_kde_plugin)
    entropies = np.asarray(entropies, dtype=float)
    assert entropies.shape == (len(graphs),)
    assert np.all(np.isfinite(entropies))


def test_ordered_lattice_entropy_reads_minimum():
    """Issue #28: at p = 1e-4 the graphs are pure ring lattices (one FRC value). The dither turns
    that constant distribution into a unit-width uniform, so the entropy reads ~0 (minimum, the
    ordered end) instead of raising on the KDE bandwidth selection or returning -inf."""
    _, graphs = gen_tv_sw(100, 10, 3, -4, -4)
    curvature_graphs = compute_frc_vec(graphs)
    assert all(np.unique(extract_curvatures(g)).size == 1 for g in curvature_graphs)
    entropies = np.asarray(vec_entropy(curvature_graphs, entropy_kde_plugin), dtype=float)
    assert np.all(np.isfinite(entropies))
    assert np.all(np.abs(entropies) < 0.5)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
