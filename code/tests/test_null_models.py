"""
Tests for hyphi.null_models: signal-, dyad-, and condition-level surrogates.

Covers:
  * phase_randomize preserves per-channel power spectrum (and DC/Nyquist bins)
  * circular_time_shift preserves the multiset of values per channel
  * dyad_subject_swap is a true derangement — no dyad keeps its original B
  * dyad_label_shuffle preserves the label multiset
  * condition_label_shuffle_within_dyad preserves per-dyad condition counts,
    and (in trial mode) all rows of a given trial share one label
  * generate_surrogate_stack dispatches the known methods and rejects others

Years: 2026
"""

# %% Import
import logging
from collections import Counter

import networkx as nx
import numpy as np
import pytest
from hyphi.null_models import (
    _random_derangement,
    circular_time_shift,
    condition_label_shuffle_within_dyad,
    configuration_model_null,
    degree_preserving_rewire,
    dyad_label_shuffle,
    dyad_subject_swap,
    generate_surrogate_stack,
    phase_randomize,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Tests for phase_randomize >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestPhaseRandomize:
    """Amplitude-preserving phase randomization."""

    def test_preserves_power_spectrum(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=(3, 256))
        surrogate = phase_randomize(sig, rng=rng)
        amp_orig = np.abs(np.fft.rfft(sig, axis=1))
        amp_surr = np.abs(np.fft.rfft(surrogate, axis=1))
        np.testing.assert_allclose(amp_orig, amp_surr, rtol=1e-6, atol=1e-6)

    def test_handles_1d_input(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=256)
        out = phase_randomize(sig, rng=rng)
        assert out.shape == sig.shape

    def test_different_seed_yields_different_surrogate(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=(1, 128))
        a = phase_randomize(sig, rng=np.random.default_rng(1))
        b = phase_randomize(sig, rng=np.random.default_rng(2))
        # They should not be the raw input, nor equal to each other
        assert not np.allclose(a, sig)
        assert not np.allclose(a, b)


# %% Tests for circular_time_shift >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestCircularTimeShift:
    """Channel-wise circular rotation."""

    def test_preserves_value_multiset(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=(4, 64))
        out = circular_time_shift(sig, min_shift=1, rng=rng)
        for ch in range(sig.shape[0]):
            np.testing.assert_allclose(np.sort(out[ch]), np.sort(sig[ch]))

    def test_handles_1d_input(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=64)
        out = circular_time_shift(sig, min_shift=1, rng=rng)
        assert out.shape == sig.shape
        np.testing.assert_allclose(np.sort(out), np.sort(sig))

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            circular_time_shift(np.zeros((1, 4)), min_shift=10)


# %% Tests for dyad_subject_swap >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestDyadSubjectSwap:
    """B-subject derangement across real dyads."""

    def test_b_subject_is_deranged(self):
        rng = np.random.default_rng(0)
        n_dyads = 6
        data = np.arange(n_dyads * 2 * 10, dtype=float).reshape(n_dyads, 2, 10)
        out = dyad_subject_swap(data, rng=rng)
        # A-subjects unchanged
        np.testing.assert_allclose(out[:, 0], data[:, 0])
        # No dyad keeps its original B-subject
        for d in range(n_dyads):
            assert not np.allclose(out[d, 1], data[d, 1])
        # Multiset of B-subjects preserved
        assert sorted(out[:, 1, 0].tolist()) == sorted(data[:, 1, 0].tolist())

    def test_single_dyad_is_noop_with_warning(self, caplog):
        data = np.zeros((1, 2, 5))
        out = dyad_subject_swap(data, rng=np.random.default_rng(0))
        np.testing.assert_allclose(out, data)

    def test_four_dim_input_supported(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(4, 2, 3, 5))
        out = dyad_subject_swap(data, rng=rng)
        assert out.shape == data.shape


# %% Tests for dyad_label_shuffle >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestDyadLabelShuffle:
    """Label permutation sanity check."""

    def test_preserves_label_counts(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(4), 10)
        out = dyad_label_shuffle(labels, rng=rng)
        assert sorted(out.tolist()) == sorted(labels.tolist())

    def test_shape_preserved(self):
        labels = np.array([1, 2, 3, 4, 5])
        out = dyad_label_shuffle(labels, rng=np.random.default_rng(0))
        assert out.shape == labels.shape


# %% Tests for condition_label_shuffle_within_dyad >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestConditionLabelShuffleWithinDyad:
    """Per-dyad condition shuffle, optionally preserving trial blocks."""

    def test_per_dyad_counts_preserved(self):
        rng = np.random.default_rng(0)
        dyad = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        cond = np.array(["A", "A", "B", "B", "A", "B", "A", "B"])
        out = condition_label_shuffle_within_dyad(cond, dyad, rng=rng)
        for d in np.unique(dyad):
            assert sorted(out[dyad == d].tolist()) == sorted(cond[dyad == d].tolist())

    def test_trial_mode_keeps_trial_block_homogeneous(self):
        rng = np.random.default_rng(0)
        dyad = np.array([0, 0, 0, 0, 0, 0])
        trial = np.array([0, 0, 1, 1, 2, 2])  # 2 rows per trial
        cond = np.array(["A", "A", "B", "B", "A", "A"])
        out = condition_label_shuffle_within_dyad(cond, dyad, trial_labels=trial, rng=rng)
        for t in np.unique(trial):
            assert len(set(out[trial == t])) == 1  # all rows of a trial share a label

    def test_trial_mode_preserves_trial_level_counts(self):
        rng = np.random.default_rng(0)
        dyad = np.zeros(8, dtype=int)
        trial = np.repeat(np.arange(4), 2)
        cond = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])
        out = condition_label_shuffle_within_dyad(cond, dyad, trial_labels=trial, rng=rng)
        # Two A-trials and two B-trials before and after
        trial_labels_before = [cond[trial == t][0] for t in np.unique(trial)]
        trial_labels_after = [out[trial == t][0] for t in np.unique(trial)]
        assert sorted(trial_labels_before) == sorted(trial_labels_after)


# %% Tests for generate_surrogate_stack >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestGenerateSurrogateStack:
    """Stack-of-surrogates convenience dispatcher."""

    def test_phase_randomize_stack_shape(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=(2, 128))
        stack = generate_surrogate_stack(sig, method="phase_randomize", n_surrogates=5, rng=rng)
        assert stack.shape == (5,) + sig.shape

    def test_time_shift_stack_shape(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=(2, 128))
        stack = generate_surrogate_stack(sig, method="circular_time_shift", n_surrogates=3, rng=rng)
        assert stack.shape == (3,) + sig.shape

    def test_dyad_swap_stack_shape(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(4, 2, 10))
        stack = generate_surrogate_stack(data, method="dyad_subject_swap", n_surrogates=4, rng=rng)
        assert stack.shape == (4,) + data.shape

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method must be one of"):
            generate_surrogate_stack(np.zeros((2, 8)), method="bogus", n_surrogates=1)


# %% Tests for _random_derangement >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestRandomDerangement:
    """The derangement helper has no fixed points, handles tiny n, and warns on fallback."""

    def test_is_a_derangement(self):
        """A derangement of [0, n) leaves no index in place."""
        rng = np.random.default_rng(0)
        perm = _random_derangement(8, rng)
        assert not np.any(perm == np.arange(8))
        assert sorted(perm) == list(range(8))  # still a permutation

    def test_small_n_returns_identity(self):
        """For n < 2 there is no derangement, so the helper returns the identity."""
        rng = np.random.default_rng(0)
        assert list(_random_derangement(0, rng)) == []
        assert list(_random_derangement(1, rng)) == [0]

    def test_fallback_warns_and_still_deranges(self, caplog):
        """Exhausting rejection sampling (max_tries=0) warns and falls back to a rotation."""
        rng = np.random.default_rng(0)
        with caplog.at_level("WARNING"):
            perm = _random_derangement(4, rng, max_tries=0)
        assert "falling back to a deterministic rotation" in caplog.text
        # The rotation is still a valid derangement (no fixed points).
        assert not np.any(perm == np.arange(4))


# %% Tests for degenerate phase_randomize input >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestPhaseRandomizeDegenerate:
    """phase_randomize stays finite and shape-preserving on degenerate channels."""

    def test_constant_channel_preserves_power_spectrum(self):
        """A constant channel has all power in DC; the surrogate preserves that and is finite."""
        signal = np.full((1, 64), 5.0)
        out = phase_randomize(signal, rng=np.random.default_rng(0))
        assert out.shape == signal.shape
        assert np.all(np.isfinite(out))
        # Power spectrum (|rfft|) is the phase-randomization invariant, even for a constant.
        np.testing.assert_allclose(np.abs(np.fft.rfft(out[0])), np.abs(np.fft.rfft(signal[0])), atol=1e-8)

    def test_constant_channel_alongside_varying_channel(self):
        """A degenerate channel does not corrupt a normal channel processed in the same call."""
        rng = np.random.default_rng(1)
        varying = rng.standard_normal(128)
        signal = np.stack([np.full(128, 2.0), varying])
        out = phase_randomize(signal, rng=np.random.default_rng(2))
        assert out.shape == signal.shape
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(np.abs(np.fft.rfft(out[1])), np.abs(np.fft.rfft(varying)), atol=1e-8)


# %% Null-model hardening >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestNullModelHardening:
    """Spectral DC sign, derangement uniformity, never-identity, stack independence, trial homogeneity."""

    def test_phase_randomize_preserves_negative_mean(self):
        """The DC component keeps its sign: a negative-mean channel does not flip to positive."""
        constant = phase_randomize(np.full((1, 64), -5.0), rng=np.random.default_rng(0))
        assert constant[0].mean() == pytest.approx(-5.0, abs=1e-8)
        rng = np.random.default_rng(1)
        varying = rng.standard_normal(128) - 3.0
        surrogate = phase_randomize(varying, rng=np.random.default_rng(2))
        assert surrogate.mean() == pytest.approx(varying.mean(), abs=1e-8)

    def test_derangement_is_approximately_uniform(self):
        """Rejection-sampled derangements are ~uniform over the 9 derangements of [0, 4)."""
        rng = np.random.default_rng(0)
        counts = Counter(tuple(_random_derangement(4, rng).tolist()) for _ in range(9000))
        assert len(counts) == 9  # all 9 derangements appear
        assert min(counts.values()) > 700  # ~1000 expected each; generous band, deterministic seed
        assert max(counts.values()) < 1300

    def test_circular_time_shift_never_identity(self):
        """A circular shift (min_shift >= 1) is never the original signal."""
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((3, 32))
        for _ in range(50):
            assert not np.array_equal(circular_time_shift(sig, min_shift=1, rng=rng), sig)

    def test_surrogate_stack_draws_are_independent(self):
        """A surrogate stack holds distinct draws, not one surrogate repeated."""
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((2, 64))
        stack = generate_surrogate_stack(sig, method="phase_randomize", n_surrogates=5, rng=rng)
        for i in range(len(stack)):
            for j in range(i + 1, len(stack)):
                assert not np.allclose(stack[i], stack[j])

    def test_condition_shuffle_rejects_heterogeneous_trial(self):
        """A (dyad, trial) block carrying multiple conditions is rejected (would lose labels)."""
        dyad = np.array([0, 0, 0, 0])
        trial = np.array([0, 0, 1, 1])
        cond = np.array(["A", "B", "A", "B"])  # trial 0 carries both A and B -> invalid
        with pytest.raises(ValueError, match="multiple conditions"):
            condition_label_shuffle_within_dyad(cond, dyad, trial_labels=trial, rng=np.random.default_rng(0))


# %% Tests for intra-brain (single-graph) nulls >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestDegreePreservingRewire:
    """The degree-preserving rewire keeps every node's degree and is reproducible."""

    def test_preserves_degree_sequence(self):
        """Double-edge swaps keep the degree of every node fixed."""
        graph = nx.watts_strogatz_graph(20, 4, 0.3, seed=0)
        rewired = degree_preserving_rewire(graph, rng=np.random.default_rng(0))
        before = sorted(d for _, d in graph.degree())
        after = sorted(d for _, d in rewired.degree())
        assert before == after

    def test_does_not_mutate_input(self):
        """The input graph is left untouched."""
        graph = nx.watts_strogatz_graph(20, 4, 0.3, seed=0)
        edges_before = set(graph.edges())
        degree_preserving_rewire(graph, rng=np.random.default_rng(1))
        assert set(graph.edges()) == edges_before

    def test_reproducible(self):
        """Same seed gives the same rewired graph."""
        graph = nx.watts_strogatz_graph(20, 4, 0.3, seed=0)
        a = degree_preserving_rewire(graph, rng=np.random.default_rng(7))
        b = degree_preserving_rewire(graph, rng=np.random.default_rng(7))
        assert set(a.edges()) == set(b.edges())

    def test_tiny_graph_is_noop(self):
        """A graph too small to swap is returned unchanged (a distinct copy, not the input object)."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        out = degree_preserving_rewire(graph, rng=np.random.default_rng(0))
        assert set(out.edges()) == {(0, 1)}
        assert out is not graph

    def test_complete_graph_warns_not_crashes(self, caplog):
        """A complete graph has a unique degree realisation, so it must warn and return a copy, not crash."""
        graph = nx.complete_graph(6)
        with caplog.at_level(logging.WARNING):
            out = degree_preserving_rewire(graph, rng=np.random.default_rng(0))
        assert out is not graph
        assert set(out.edges()) == set(graph.edges())  # unrewired copy
        assert "no power" in caplog.text

    def test_degenerate_graph_warns(self, caplog):
        """A graph past the size guard with no valid swap returns the input and warns (no silent degenerate null)."""
        graph = nx.path_graph(4)
        graph.add_nodes_from([4, 5, 6])  # isolated nodes -> 7 nodes, 3 edges, no valid swap
        with caplog.at_level(logging.WARNING):
            out = degree_preserving_rewire(graph, rng=np.random.default_rng(0))
        assert set(out.edges()) == set(graph.edges())
        assert "no power" in caplog.text

    def test_rejects_directed_graph(self):
        """A directed graph is out of contract and rejected with a clear error."""
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
        with pytest.raises(TypeError, match="undirected simple graph"):
            degree_preserving_rewire(graph, rng=np.random.default_rng(0))

    def test_rejects_multigraph(self):
        """A multigraph is out of contract and rejected with a clear error."""
        graph = nx.MultiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
        with pytest.raises(TypeError, match="undirected simple graph"):
            degree_preserving_rewire(graph, rng=np.random.default_rng(0))

    def test_dense_feasible_graph_not_falsely_flagged_no_power(self, caplog):
        """A dense but rewireable graph is a valid randomised null; it must not be mislabelled 'no power'."""
        graph = nx.erdos_renyi_graph(40, 0.8, seed=0)
        with caplog.at_level(logging.WARNING):
            out = degree_preserving_rewire(graph, rng=np.random.default_rng(0))
        assert sorted(d for _, d in out.degree()) == sorted(d for _, d in graph.degree())
        assert set(out.edges()) != set(graph.edges())  # genuinely randomised
        assert "no power" not in caplog.text


class TestConfigurationModelNull:
    """The configuration-model null matches the node set and is degree-bounded and reproducible."""

    def test_same_node_count_and_degree_bounded(self):
        """The null has the same nodes and no more total degree than the original."""
        graph = nx.watts_strogatz_graph(30, 6, 0.2, seed=0)
        null = configuration_model_null(graph, rng=np.random.default_rng(0))
        assert null.number_of_nodes() == graph.number_of_nodes()
        # Self-loops and parallel edges are removed, so total degree cannot exceed the target.
        assert sum(d for _, d in null.degree()) <= sum(d for _, d in graph.degree())
        assert nx.number_of_selfloops(null) == 0

    def test_reproducible(self):
        """Same seed gives the same configuration-model graph."""
        graph = nx.watts_strogatz_graph(30, 6, 0.2, seed=0)
        a = configuration_model_null(graph, rng=np.random.default_rng(3))
        b = configuration_model_null(graph, rng=np.random.default_rng(3))
        assert set(a.edges()) == set(b.edges())

    def test_preserves_node_labels(self):
        """The null keeps the input node identities (not relabeled to 0..n-1)."""
        graph = nx.relabel_nodes(nx.watts_strogatz_graph(8, 4, 0.3, seed=0), {i: f"ch{i}" for i in range(8)})
        null = configuration_model_null(graph, rng=np.random.default_rng(0))
        assert set(null.nodes()) == set(graph.nodes())


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
