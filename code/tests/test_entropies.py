"""
Known-Answer Tests for Entropy Estimation.

Tests entropy estimators against analytically known values
using synthetic distributions.
"""

# %% Import
import networkx as nx
import numpy as np
import pytest
from hyphi.analyses import compute_entropy
from hyphi.modeling.entropies import (
    _CONTENTLESS_ENTROPY,
    entropy_kde_plugin,
    entropy_vasicek,
    get_estimator,
    get_quantiles,
    vec_entropy,
    vec_quantiles,
)

ALL_ESTIMATOR_NAMES = [
    "vasicek",
    "van_es",
    "ebrahimi",
    "correa",
    "kde_plugin",
    "kozachenko",
    "renyi",
    "tsallis",
]


def _graph_with_curvatures(curvatures):
    """Build a path graph and set each edge's formanCurvature from the given values."""
    n_edges = len(curvatures)
    G = nx.path_graph(n_edges + 1)
    for idx, (u, v) in enumerate(G.edges()):
        G[u][v]["formanCurvature"] = curvatures[idx]
    return G


# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestEntropyEstimation:
    """Tests for differential entropy estimators."""

    def _make_graph_with_known_curvatures(self, curvatures):
        """Helper: build a graph and manually set curvature values.

        Creates a path graph and overrides edge curvatures with provided values.
        """
        import networkx as nx

        n_edges = len(curvatures)
        G = nx.path_graph(n_edges + 1)

        for idx, (u, v) in enumerate(G.edges()):
            G[u][v]["formanCurvature"] = curvatures[idx]

        return G

    def test_vasicek_uniform_distribution(self):
        """Uniform[0,1] → theoretical entropy = 0.
        Vasicek estimate should be close to 0.
        """
        np.random.seed(42)
        curvatures = np.random.uniform(0, 1, 500)
        G = self._make_graph_with_known_curvatures(curvatures)
        H = entropy_vasicek(G)
        # Uniform[0,1] has differential entropy = ln(1) = 0
        assert abs(H - 0.0) < 0.3  # generous tolerance for finite sample

    def test_vasicek_gaussian_distribution(self):
        """N(0,1) → theoretical entropy ≈ 1.4189 (½ ln(2πe)).
        Check Vasicek estimate is in the right ballpark.
        """
        np.random.seed(42)
        curvatures = np.random.randn(1000)
        G = self._make_graph_with_known_curvatures(curvatures)
        H = entropy_vasicek(G)
        expected = 0.5 * np.log(2 * np.pi * np.e)  # ≈ 1.4189
        assert abs(H - expected) < 0.3

    def test_kde_plugin_gaussian(self):
        """KDE plugin on N(0,1) should approximate ½ ln(2πe)."""
        np.random.seed(42)
        curvatures = np.random.randn(1000)
        G = self._make_graph_with_known_curvatures(curvatures)
        H = entropy_kde_plugin(G)
        expected = 0.5 * np.log(2 * np.pi * np.e)
        assert abs(H - expected) < 0.3


class TestQuantiles:
    """Tests for quantile extraction."""

    def _make_graph_with_known_curvatures(self, curvatures):
        import networkx as nx

        n_edges = len(curvatures)
        G = nx.path_graph(n_edges + 1)
        for idx, (u, v) in enumerate(G.edges()):
            G[u][v]["formanCurvature"] = curvatures[idx]
        return G

    def test_quantiles_symmetric_distribution(self):
        """For a symmetric distribution centred at 0, median should be ≈ 0."""
        np.random.seed(42)
        curvatures = np.random.randn(1000)
        G = self._make_graph_with_known_curvatures(curvatures)
        qs = get_quantiles(G, qs=[0.25, 0.5, 0.75])
        assert abs(qs[1]) < 0.15  # median near 0

    def test_vec_quantiles(self):
        """vec_quantiles should work on multiple graphs."""
        np.random.seed(42)
        G1 = self._make_graph_with_known_curvatures(np.random.randn(200))
        G2 = self._make_graph_with_known_curvatures(np.random.randn(200))
        result = vec_quantiles([G1, G2], qs=[0.5])
        assert result.shape == (2, 1)


class TestVecEntropy:
    """Tests for vectorised entropy."""

    def _make_graph_with_known_curvatures(self, curvatures):
        import networkx as nx

        n_edges = len(curvatures)
        G = nx.path_graph(n_edges + 1)
        for idx, (u, v) in enumerate(G.edges()):
            G[u][v]["formanCurvature"] = curvatures[idx]
        return G

    def test_vec_entropy_returns_array(self):
        """vec_entropy should return an ndarray of length n_graphs."""
        np.random.seed(42)
        graphs = [self._make_graph_with_known_curvatures(np.random.randn(100)) for _ in range(3)]
        result = vec_entropy(graphs, estimator=entropy_vasicek)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


class TestEstimatorRegistry:
    """The registry exposes every estimator by name and dispatches through compute_entropy."""

    @pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
    def test_each_estimator_runs_and_is_finite(self, name):
        """Every registered estimator runs on a Gaussian curvature distribution and is finite."""
        rng = np.random.default_rng(0)
        G = _graph_with_curvatures(rng.standard_normal(400))
        H = get_estimator(name)(G)
        assert np.isfinite(H)

    def test_compute_entropy_reaches_every_estimator(self):
        """compute_entropy dispatches to all eight estimators plus the kde alias."""
        rng = np.random.default_rng(1)
        G = _graph_with_curvatures(rng.standard_normal(400))
        for name in [*ALL_ESTIMATOR_NAMES, "kde"]:
            out = compute_entropy([G], method=name)
            assert out.shape == (1,)
            assert np.isfinite(out[0])

    def test_unknown_method_raises(self):
        """An unknown estimator name raises a clear ValueError, both directly and via the facade."""
        rng = np.random.default_rng(2)
        G = _graph_with_curvatures(rng.standard_normal(50))
        with pytest.raises(ValueError, match="Unknown entropy method"):
            get_estimator("not_a_method")
        with pytest.raises(ValueError, match="Unknown entropy method"):
            compute_entropy([G], method="not_a_method")


class TestDegenerateInput:
    """Degenerate and tied input is finite (no -inf/NaN/raise): contentless -> 0.0, constant -> ~0."""

    @pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
    def test_single_value_is_contentless(self, name):
        """A single-edge graph (one curvature value) has no distribution: returns the 0.0 sentinel."""
        G = _graph_with_curvatures([1.0])
        assert get_estimator(name)(G) == _CONTENTLESS_ENTROPY

    @pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
    def test_empty_is_contentless(self, name):
        """An edgeless graph (no curvatures) has no distribution: returns the 0.0 sentinel."""
        G = nx.Graph()
        assert get_estimator(name)(G) == _CONTENTLESS_ENTROPY

    @pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
    @pytest.mark.parametrize("n_samples", [2, 3, 4])
    def test_too_few_samples_are_contentless(self, name, n_samples):
        """Fewer than k + 1 = 5 samples is too few for the kNN estimators (which would raise/return
        inf); the guard returns the 0.0 sentinel for every estimator rather than raising."""
        G = _graph_with_curvatures([1.0 + i for i in range(n_samples)])
        assert get_estimator(name)(G) == _CONTENTLESS_ENTROPY

    @pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
    def test_constant_curvatures_read_as_minimum(self, name):
        """A constant (zero-variance) distribution dithers to a unit uniform, so entropy reads ~0
        (the low end), not the high extreme and not -inf."""
        G = _graph_with_curvatures([2.0] * 30)
        value = float(get_estimator(name)(G))
        assert np.isfinite(value)
        assert abs(value) < 1.2

    @pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
    def test_few_distinct_integer_curvatures_are_finite(self, name):
        """Regression for the #28 near-constant gap: a few-distinct, heavily-tied integer curvature
        distribution (what near-lattice Forman-Ricci graphs produce across the small-world
        transition) must be finite for every estimator, not -inf/NaN/raise."""
        rng = np.random.default_rng(0)
        curvatures = rng.choice([-9.0, -8.0, -7.0], size=60).tolist()
        G = _graph_with_curvatures(curvatures)
        value = float(get_estimator(name)(G))
        assert np.isfinite(value)

    @pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
    def test_constant_below_disordered(self, name):
        """The dithered constant (ordered) reads no higher than a genuinely disordered distribution,
        so the entropy-over-time transition peak is preserved (ordered = minimum)."""
        rng = np.random.default_rng(1)
        ordered = _graph_with_curvatures([3.0] * 60)
        disordered = _graph_with_curvatures(rng.integers(-12, 5, size=60).astype(float).tolist())
        assert float(get_estimator(name)(ordered)) <= float(get_estimator(name)(disordered)) + 1e-9


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
