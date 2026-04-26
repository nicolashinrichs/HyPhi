"""
Tests for hyphi.stats: hierarchical permutation, mixed-effects, effect sizes, power.

Covers:
  * long-form reshape from nested dyad → condition → (n_freq, n_trials, n_windows)
  * Cohen's d scalar and per-window time series with known-answer Gaussians
  * mixed-effects smoke-fit on a 3-dyad × 2-condition synthetic dataset
  * hierarchical permutation controls: p ≈ 0.5 under H0, p small under H1
  * hierarchical energy distance wrapper
  * required_sample_size against a known TTestIndPower solution

Years: 2026
"""

# %% Import
import numpy as np
import pandas as pd
import pytest

from hyphi.stats import (
    cohens_d,
    cohens_d_timeseries,
    energy_distance_hierarchical,
    entropy_to_long_df,
    hierarchical_permutation_test,
    mixed_effects_test,
    required_sample_size,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Helpers >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _synthetic_entropy_dict(
    n_dyads: int = 3,
    conditions=("A", "B"),
    n_freq: int = 2,
    n_trials: int = 4,
    n_windows: int = 3,
    effect: float = 0.0,
    seed: int = 0,
) -> dict:
    """Nested entropy dict with optional mean shift between conditions."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for d in range(n_dyads):
        out[d] = {}
        for c_idx, c in enumerate(conditions):
            shift = effect * c_idx
            out[d][c] = rng.normal(loc=shift, scale=1.0, size=(n_freq, n_trials, n_windows))
    return out


# %% Tests for entropy_to_long_df >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestEntropyToLongDf:
    """Reshape from nested dict to long-form dataframe."""

    def test_shape_and_columns(self):
        data = _synthetic_entropy_dict(n_dyads=2, n_freq=2, n_trials=3, n_windows=4)
        df = entropy_to_long_df(data, freq_bands=["delta", "theta"])
        expected_rows = 2 * 2 * 2 * 3 * 4  # dyads × cond × freq × trials × windows
        assert len(df) == expected_rows
        assert set(df.columns) == {
            "dyad",
            "condition",
            "freq",
            "trial",
            "trial_id",
            "window",
            "entropy",
        }

    def test_trial_id_globally_unique_within_dyad(self):
        data = _synthetic_entropy_dict(n_dyads=2, n_trials=3)
        df = entropy_to_long_df(data)
        # trial_id encodes dyad, so trial 0 of dyad 0 != trial 0 of dyad 1
        ids = df[["dyad", "trial_id"]].drop_duplicates()
        assert (ids.groupby("dyad")["trial_id"].nunique() == 3).all()

    def test_freq_bands_fallback_to_integers(self):
        data = _synthetic_entropy_dict(n_freq=3)
        df = entropy_to_long_df(data)
        assert set(df["freq"].unique()) == {0, 1, 2}

    def test_wrong_shape_raises(self):
        bad = {0: {"A": np.zeros((2, 3))}}  # 2-D, not 3-D
        with pytest.raises(ValueError, match="Expected"):
            entropy_to_long_df(bad)


# %% Tests for Cohen's d >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestCohensD:
    """Cohen's d scalar."""

    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        assert abs(cohens_d(a, b)) < 0.2

    def test_mean_shift_recovered(self):
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 1.0, 2000)
        b = rng.normal(0.0, 1.0, 2000)
        d = cohens_d(a, b)
        assert 0.8 < d < 1.2  # true d == 1.0

    def test_sign_follows_mean_order(self):
        a = np.array([2.0, 3.0, 4.0, 5.0])
        b = np.array([0.0, 1.0, 2.0, 3.0])
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0

    def test_zero_variance_returns_zero(self):
        a = np.ones(10)
        b = np.ones(10)
        assert cohens_d(a, b) == 0.0

    def test_short_sample_returns_zero(self):
        assert cohens_d(np.array([1.0]), np.array([2.0])) == 0.0

    def test_paired_requires_equal_length(self):
        with pytest.raises(ValueError):
            cohens_d(np.zeros(3), np.zeros(4), paired=True)


class TestCohensDTimeseries:
    """Cohen's d per non-sample axis."""

    def test_shape_collapses_sample_axis(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(20, 5))  # 20 obs, 5 windows
        b = rng.normal(size=(20, 5))
        d = cohens_d_timeseries(a, b, axis=0)
        assert d.shape == (5,)

    def test_detects_per_window_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=(500, 3))
        b = rng.normal(0.0, 1.0, size=(500, 3))
        a[:, 1] += 1.0  # shift only window 1
        d = cohens_d_timeseries(a, b, axis=0)
        assert abs(d[0]) < 0.2
        assert 0.8 < d[1] < 1.2
        assert abs(d[2]) < 0.2

    def test_mismatched_non_sample_axes_raise(self):
        with pytest.raises(ValueError):
            cohens_d_timeseries(np.zeros((10, 4)), np.zeros((10, 5)), axis=0)


# %% Tests for hierarchical permutation >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestHierarchicalPermutation:
    """Within-dyad, trial-level condition-label permutation."""

    def test_null_yields_midrange_p(self):
        data = _synthetic_entropy_dict(n_dyads=3, effect=0.0, seed=1)
        df = entropy_to_long_df(data)
        res = hierarchical_permutation_test(
            data=df, value_col="entropy", condition_col="condition", n_perms=200, seed=1
        )
        # Under H0 the p-value should be roughly uniform; accept a generous band.
        assert 0.05 < res["p_value"] < 0.95
        assert res["n_dyads"] == 3
        assert res["n_perms"] == 200
        assert res["null_distribution"].shape == (200,)

    def test_strong_effect_rejects(self):
        data = _synthetic_entropy_dict(n_dyads=4, n_trials=6, effect=3.0, seed=2)
        df = entropy_to_long_df(data)
        res = hierarchical_permutation_test(
            data=df, value_col="entropy", condition_col="condition", n_perms=500, seed=2
        )
        assert res["p_value"] < 0.05
        assert res["observed_stat"] > 0

    def test_missing_column_raises(self):
        df = pd.DataFrame({"dyad": [0, 0], "trial_id": ["0__0", "0__1"], "entropy": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            hierarchical_permutation_test(
                data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0
            )

    def test_unknown_tail_raises(self):
        data = _synthetic_entropy_dict(n_dyads=2, n_trials=3)
        df = entropy_to_long_df(data)
        with pytest.raises(ValueError, match="Unknown tail"):
            hierarchical_permutation_test(
                data=df,
                value_col="entropy",
                condition_col="condition",
                n_perms=5,
                seed=0,
                tail="left",
            )


# %% Tests for hierarchical energy distance >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestEnergyDistanceHierarchical:
    """Hierarchical energy-distance wrapper."""

    def test_large_effect_rejects(self):
        data = _synthetic_entropy_dict(n_dyads=3, n_trials=5, effect=2.5, seed=3)
        df = entropy_to_long_df(data)
        res = energy_distance_hierarchical(
            data=df, value_col="entropy", condition_col="condition", n_perms=300, seed=3
        )
        assert res["p_value"] < 0.1
        assert res["observed_stat"] >= 0.0


# %% Tests for mixed-effects >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestMixedEffects:
    """Random-intercept model on a small synthetic sample."""

    def test_fits_and_exposes_params(self):
        data = _synthetic_entropy_dict(n_dyads=4, n_trials=5, effect=1.0, seed=4)
        df = entropy_to_long_df(data)
        df_band = df[df["freq"] == 0]
        result = mixed_effects_test(
            data=df_band,
            formula="entropy ~ C(condition)",
            groups="dyad",
            re_formula="1",
        )
        # Intercept and C(condition)[T.B] should both be present
        params = dict(result.params)
        assert "Intercept" in params
        assert any(k.startswith("C(condition)") for k in params)
        assert hasattr(result, "llf")


# %% Tests for required_sample_size >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class TestRequiredSampleSize:
    """Power-analysis wrapper around TTestIndPower."""

    def test_medium_effect_known_answer(self):
        # Classical result: d = 0.5, alpha=0.05, power=0.8, two-sided → ≈ 64 per group
        out = required_sample_size(effect_size=0.5)
        assert 60 < out["n_per_group"] < 70
        assert out["n_total"] == pytest.approx(out["n_per_group"] * 2)
        assert out["alpha"] == 0.05
        assert out["power"] == 0.8

    def test_zero_effect_returns_inf(self):
        out = required_sample_size(effect_size=0.0)
        assert np.isinf(out["n_per_group"])
        assert np.isinf(out["n_total"])

    def test_larger_effect_needs_fewer(self):
        small = required_sample_size(effect_size=0.2)
        big = required_sample_size(effect_size=1.0)
        assert big["n_per_group"] < small["n_per_group"]


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
