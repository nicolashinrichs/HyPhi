"""
Tests for the hyperbolic embeddings module.

Checks the Poincare-ball geometry utilities and that a hyperbolic embedding of simulated
multichannel data lands inside the open ball with the right shape.
"""

# %% Import
import numpy as np
import pytest
from hyphi.embeddings import (
    hyperbolic_embedding,
    poincare_distance,
    poincare_distance_matrix,
    poincare_exp0,
)

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _simulated_signal(n_channels=8, n_times=400, seed=0):
    """Build simulated multichannel data: a few correlated groups, as a stand-in for Kuramoto output."""
    rng = np.random.default_rng(seed)
    groups = [rng.standard_normal(n_times) for _ in range(3)]
    channels = []
    for c in range(n_channels):
        base = groups[c % 3]
        channels.append(base + 0.5 * rng.standard_normal(n_times))
    return np.stack(channels)


class TestPoincareGeometry:
    """The Poincare-ball utilities behave as a hyperbolic metric should."""

    def test_exp0_maps_into_open_ball(self):
        """The exponential map sends any Euclidean vector strictly inside the unit ball."""
        rng = np.random.default_rng(0)
        euclidean = rng.standard_normal((50, 2)) * 10.0  # large vectors
        ball = poincare_exp0(euclidean)
        radii = np.linalg.norm(ball, axis=-1)
        assert np.all(radii < 1.0)

    def test_distance_is_a_metric(self):
        """Poincare distance is non-negative, zero on the diagonal, and symmetric."""
        rng = np.random.default_rng(1)
        points = poincare_exp0(rng.standard_normal((6, 2)))
        matrix = poincare_distance_matrix(points)
        assert np.all(matrix >= 0.0)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-9)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)

    def test_distance_grows_toward_the_boundary(self):
        """Two points near the boundary are hyperbolically far apart."""
        center = np.array([0.0, 0.0])
        near_edge = np.array([0.9, 0.0])
        far_edge = np.array([-0.9, 0.0])
        assert poincare_distance(near_edge, far_edge) > poincare_distance(center, near_edge)

    def test_distance_rejects_out_of_ball(self):
        """A point outside the unit ball is rejected, not clamped to a finite-but-wrong distance."""
        with pytest.raises(ValueError, match="strictly inside"):
            poincare_distance(np.array([2.0, 0.0]), np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="strictly inside"):
            poincare_distance(np.array([1.2, 0.0]), np.array([1.3, 0.0]))


class TestHyperbolicEmbedding:
    """A hyperbolic embedding of simulated data has the right shape and lands in the ball."""

    def test_embedding_shape_and_in_ball(self):
        """The embedding has one point per channel, in the open Poincare ball."""
        signal = _simulated_signal(n_channels=8)
        embedding = hyperbolic_embedding(signal, n_components=2)
        assert embedding.shape == (8, 2)
        assert np.all(np.linalg.norm(embedding, axis=-1) < 1.0)

    def test_reproducible(self):
        """A fixed random_state gives the same embedding."""
        signal = _simulated_signal()
        a = hyperbolic_embedding(signal, random_state=3)
        b = hyperbolic_embedding(signal, random_state=3)
        np.testing.assert_allclose(a, b)

    def test_unknown_metric_raises(self):
        """An unknown dissimilarity metric raises a clear error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            hyperbolic_embedding(_simulated_signal(), metric="not_a_metric")

    def test_rejects_dead_channel(self):
        """A constant (dead/flat) channel is rejected loudly, not passed as a NaN correlation to MDS."""
        signal = _simulated_signal(n_channels=5)
        signal[2, :] = 1.0  # a flat electrode
        with pytest.raises(ValueError, match="constant"):
            hyperbolic_embedding(signal, metric="correlation")

    def test_rejects_nonfinite_data(self):
        """inf/NaN in the signal is rejected up front (it would otherwise reach MDS as a cryptic error)."""
        signal = _simulated_signal(n_channels=5)
        signal[0, 10] = np.inf
        with pytest.raises(ValueError, match="inf or NaN"):
            hyperbolic_embedding(signal)

    def test_rejects_too_few_channels(self):
        """A single channel cannot form a dissimilarity / MDS embedding and is rejected."""
        with pytest.raises(ValueError, match="at least 2 channels"):
            hyperbolic_embedding(_simulated_signal(n_channels=1))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
