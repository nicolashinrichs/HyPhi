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
import numpy as np
import pytest

from hyphi.null_models import (
    circular_time_shift,
    condition_label_shuffle_within_dyad,
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


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
