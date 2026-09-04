"""
Known-Answer and Smoke Tests for graph_simulations module.

Covers: gen_weighted_sw, gen_tv_weighted_sw, gen_tv_sw,
        gen_nature_sw, gen_neureps_wsw.

Numeric derivations
-------------------
gen_weighted_sw(n, k, p, eps, seed):
  - Base ring lattice with p=0 has exactly n*k/2 edges.
  - Dmax = (floor(n/2) + 1) * eps
  - Edge weight = Dmax - d_ij  where d_ij in 1..floor(n/2)
  - For n=_N, k=_K, p=0, eps=1.0:
      Dmax = 11.0; edges link nodes at distance 1 or 2
      -> unique weights {9.0, 10.0}; all weights > 0
  - All node weights are set to 1.0.

gen_tv_sw / gen_tv_weighted_sw:
  - Return (pt, Gt) with len(pt) == len(Gt) == trez.
  - pt = np.logspace(minpow, maxpow, trez)
  - pt[0] == 10**minpow,  pt[-1] == 10**maxpow.

gen_nature_sw / gen_neureps_wsw:
  - Convenience wrappers; exercise structure not full 1000-node size.
"""

# %% Import
import networkx as nx
import numpy as np
import pytest
from hyphi.simulation.graph_simulations import (
    gen_nature_sw,
    gen_neureps_wsw,
    gen_tv_sw,
    gen_tv_weighted_sw,
    gen_weighted_sw,
)

# %% Named constants (ruff PLR2004: no bare magic numbers in comparisons) >>>

_N = 20  # number of nodes used in toy tests
_K = 4  # nearest-neighbour count on ring lattice
_EPS = 1.0  # node spacing for weighted tests
_TREZ = 5  # number of time/probability points
_MINPOW = -4  # 10**-4 = 1e-4
_MAXPOW = 0  # 10** 0 = 1.0
_SEED = 42

# Derived from math (see module docstring above)
_RING_EDGES = _N * _K // 2  # 40 edges for n=20, k=4 with p=0
_DMAX = (np.floor(_N / 2) + 1) * _EPS  # 11.0
_W_D1 = _DMAX - 1  # weight for ring-distance 1 = 10.0
_W_D2 = _DMAX - 2  # weight for ring-distance 2 =  9.0
_NATURE_TREZ = 100  # expected length from gen_nature_sw


# %% Test classes >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


class TestGenWeightedSW:
    """Known-answer tests for gen_weighted_sw."""

    def test_node_count(self):
        """Graph must contain exactly n nodes."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        assert G.number_of_nodes() == _N

    def test_edge_count_no_rewiring(self):
        """With p=0, ring lattice has exactly n*k/2 edges."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        assert G.number_of_edges() == _RING_EDGES

    def test_node_weights_unity(self):
        """Every node must carry weight attribute equal to 1.0."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        for v in G.nodes:
            assert G.nodes[v]["weight"] == pytest.approx(1.0)

    def test_all_edges_have_weight(self):
        """Every edge must carry a 'weight' attribute."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        for u, v in G.edges:
            assert "weight" in G[u][v], f"Edge ({u},{v}) missing weight"

    def test_edge_weights_positive(self):
        """All edge weights must be strictly positive (Dmax - d_ij > 0)."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        for u, v in G.edges:
            assert G[u][v]["weight"] > 0

    def test_edge_weights_bounded(self):
        """Edge weights cannot exceed Dmax - 1 (minimum ring distance is 1)."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        for u, v in G.edges:
            assert G[u][v]["weight"] <= _W_D1 + 1e-12

    def test_edge_weights_known_values_ring(self):
        """With p=0 on n=20,k=4, ring distances are 1 or 2 -> weights 10 or 9."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        unique_w = {round(G[u][v]["weight"], 10) for u, v in G.edges}
        assert unique_w == {_W_D1, _W_D2}

    def test_seed_reproducibility(self):
        """Same seed must produce identical edge sets."""
        g1 = gen_weighted_sw(_N, _K, 0.5, _EPS, seed_val=_SEED)
        g2 = gen_weighted_sw(_N, _K, 0.5, _EPS, seed_val=_SEED)
        assert set(g1.edges) == set(g2.edges)

    def test_different_seeds_may_differ(self):
        """Different seeds at high rewiring probability should produce distinct graphs."""
        g1 = gen_weighted_sw(_N, _K, 0.9, _EPS, seed_val=0)
        g2 = gen_weighted_sw(_N, _K, 0.9, _EPS, seed_val=99)
        # Very high probability: rewired graphs almost certainly differ
        assert set(g1.edges) != set(g2.edges)

    def test_returns_undirected_graph(self):
        """Result must be an undirected networkx.Graph."""
        G = gen_weighted_sw(_N, _K, 0.0, _EPS, seed_val=_SEED)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)


class TestGenTvSW:
    """Known-answer tests for gen_tv_sw (unweighted time-varying series)."""

    def test_pt_length(self):
        """Probability array must have length trez."""
        pt, _ = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        assert len(pt) == _TREZ

    def test_graphs_length(self):
        """Graph list must have length trez."""
        _, gt = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        assert len(gt) == _TREZ

    def test_pt_first_value(self):
        """First probability value must equal 10**minpow."""
        pt, _ = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        assert pt[0] == pytest.approx(10.0**_MINPOW)

    def test_pt_last_value(self):
        """Last probability value must equal 10**maxpow."""
        pt, _ = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        assert pt[-1] == pytest.approx(10.0**_MAXPOW)

    def test_pt_logspace(self):
        """Probability array must match np.logspace(minpow, maxpow, trez) exactly."""
        pt, _ = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        expected = np.logspace(_MINPOW, _MAXPOW, _TREZ)
        np.testing.assert_array_almost_equal(pt, expected)

    def test_each_graph_has_n_nodes(self):
        """Every graph in the series must have exactly n nodes."""
        _, gt = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        for t, g in enumerate(gt):
            assert g.number_of_nodes() == _N, f"Graph at t={t} has wrong node count"

    def test_graphs_are_undirected(self):
        """Every graph must be an undirected networkx.Graph."""
        _, gt = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        for g in gt:
            assert isinstance(g, nx.Graph)

    def test_no_duplicate_pt(self):
        """Probability array values should be strictly increasing (logspace)."""
        pt, _ = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        diffs = np.diff(pt)
        assert np.all(diffs > 0)


class TestGenTvWeightedSW:
    """Known-answer tests for gen_tv_weighted_sw."""

    def test_pt_length(self):
        """Probability array must have length trez."""
        pt, _ = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        assert len(pt) == _TREZ

    def test_graphs_length(self):
        """Graph list must have length trez."""
        _, gt = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        assert len(gt) == _TREZ

    def test_pt_logspace(self):
        """Probability array must match np.logspace(minpow, maxpow, trez)."""
        pt, _ = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        expected = np.logspace(_MINPOW, _MAXPOW, _TREZ)
        np.testing.assert_array_almost_equal(pt, expected)

    def test_each_graph_has_n_nodes(self):
        """Every graph must have exactly n nodes."""
        _, gt = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        for t, g in enumerate(gt):
            assert g.number_of_nodes() == _N, f"Graph at t={t} wrong node count"

    def test_all_edges_have_weight(self):
        """Every edge in every graph must carry a 'weight' attribute."""
        _, gt = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        for t, g in enumerate(gt):
            for u, v in g.edges:
                assert "weight" in g[u][v], f"t={t} edge ({u},{v}) missing weight"

    def test_all_nodes_have_weight(self):
        """Every node in every graph must carry a 'weight' attribute equal to 1.0."""
        _, gt = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        for t, g in enumerate(gt):
            for v in g.nodes:
                assert g.nodes[v]["weight"] == pytest.approx(1.0), f"t={t} node {v} bad weight"

    def test_edge_weights_positive(self):
        """All edge weights must be strictly positive across the time series."""
        _, gt = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        for g in gt:
            for u, v in g.edges:
                assert g[u][v]["weight"] > 0

    def test_matches_gen_tv_sw_shape(self):
        """Weighted and unweighted series must have same pt and graph-count structure."""
        pt_w, gt_w = gen_tv_weighted_sw(_N, _K, _EPS, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        pt_u, gt_u = gen_tv_sw(_N, _K, _TREZ, _MINPOW, _MAXPOW, seed_val=_SEED)
        np.testing.assert_array_almost_equal(pt_w, pt_u)
        assert len(gt_w) == len(gt_u)


class TestGenNatureSW:
    """Smoke tests for gen_nature_sw (the 1000-node Nature-methods series)."""

    def test_returns_two_items(self):
        """gen_nature_sw must return a (pt, Gt) tuple."""
        # Note: only inspects metadata (pt, len) without iterating over 1000-node
        # graphs, so this stays fast.
        result = gen_nature_sw(seed_val=_SEED)
        assert len(result) == 2  # noqa: PLR2004 -- length-2 tuple, not a comparison to a domain constant

    def test_pt_length(self):
        """Probability array must have 100 entries (trez=100 in the Nature pipeline)."""
        pt, _ = gen_nature_sw(seed_val=_SEED)
        assert len(pt) == _NATURE_TREZ

    def test_graphs_length(self):
        """Graph list must contain 100 graphs."""
        _, gt = gen_nature_sw(seed_val=_SEED)
        assert len(gt) == _NATURE_TREZ

    def test_pt_bounds(self):
        """First entry ~ 1e-4 and last entry ~ 1.0 (logspace -4 to 0)."""
        pt, _ = gen_nature_sw(seed_val=_SEED)
        assert pt[0] == pytest.approx(1e-4)
        assert pt[-1] == pytest.approx(1.0)


class TestGenNeurepsWSW:
    """Smoke tests for gen_neureps_wsw (the 1000-node NeuroReps weighted series)."""

    def test_returns_two_items(self):
        """gen_neureps_wsw must return a (pt, Gt) tuple."""
        result = gen_neureps_wsw(seed_val=_SEED)
        assert len(result) == 2  # noqa: PLR2004 -- tuple length, not domain value

    def test_pt_length(self):
        """Probability array must have 100 entries."""
        pt, _ = gen_neureps_wsw(seed_val=_SEED)
        assert len(pt) == _NATURE_TREZ

    def test_graphs_length(self):
        """Graph list must contain 100 graphs."""
        _, gt = gen_neureps_wsw(seed_val=_SEED)
        assert len(gt) == _NATURE_TREZ

    def test_pt_bounds(self):
        """First entry ~ 1e-4 and last entry ~ 1.0."""
        pt, _ = gen_neureps_wsw(seed_val=_SEED)
        assert pt[0] == pytest.approx(1e-4)
        assert pt[-1] == pytest.approx(1.0)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
