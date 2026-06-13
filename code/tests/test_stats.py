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
from scipy.stats import kstest

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
        # trial_id encodes dyad and condition: 2 conditions x 3 trials = 6
        # distinct trials per dyad, and trial 0 of dyad 0 != trial 0 of dyad 1.
        ids = df[["dyad", "trial_id"]].drop_duplicates()
        assert (ids.groupby("dyad")["trial_id"].nunique() == 6).all()
        # Every (dyad, trial_id) block carries exactly one condition.
        blocks = df.drop_duplicates(["dyad", "trial_id", "condition"]).groupby(["dyad", "trial_id"]).size()
        assert (blocks == 1).all()

    def test_freq_bands_fallback_to_integers(self):
        data = _synthetic_entropy_dict(n_freq=3)
        df = entropy_to_long_df(data)
        assert set(df["freq"].unique()) == {0, 1, 2}

    def test_wrong_shape_raises(self):
        bad = {0: {"A": np.zeros((2, 3))}}  # 2-D, not 3-D
        with pytest.raises(ValueError, match="Expected"):
            entropy_to_long_df(bad)

    def test_freq_bands_length_mismatch_raises(self):
        # 3 frequency rows but only 2 band names would silently index out of range.
        data = _synthetic_entropy_dict(n_freq=3)
        with pytest.raises(ValueError, match="freq_bands has 2 names"):
            entropy_to_long_df(data, freq_bands=["delta", "theta"])


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
            hierarchical_permutation_test(data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0)

    def test_null_p_calibrated_across_seeds(self):
        # Regression: a degenerate null distribution once made every seed
        # return the minimum possible p (a 100% false-positive rate).
        ps = []
        for seed in range(40):
            df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=3, effect=0.0, seed=seed))
            res = hierarchical_permutation_test(
                data=df, value_col="entropy", condition_col="condition", n_perms=200, seed=seed
            )
            ps.append(res["p_value"])
        ps = np.asarray(ps)
        # Under a true null, p is ~uniform on (0, 1]: the mean sits mid-range and
        # only a small fraction fall below 0.05. The degenerate bug made EVERY
        # seed return the minimum p. The mean/tail bands catch that severe failure;
        # the KS test against Uniform(0, 1) is the sharper guard, since
        # anti-conservativeness (the bug class fixed here) shifts the whole p-value
        # distribution toward 0 and collapses the KS p-value. Everything here is
        # seeded, so these thresholds are deterministic, not flaky (numpy is pinned
        # < 2.0, so default_rng's stream is stable); the bands keep a wide margin
        # over the observed values (mean 0.53, tail 0.0, KS p 0.27).
        assert 0.35 < ps.mean() < 0.65
        assert np.mean(ps < 0.05) <= 0.10
        assert kstest(ps, "uniform").pvalue > 0.02

    def test_inf_value_raises(self):
        """Refuse non-finite (inf) values: isna() misses inf, but it still poisons the statistic."""
        df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=3, effect=0.0, seed=1))
        df.loc[5, "entropy"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            hierarchical_permutation_test(data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0)

    def test_three_conditions_run(self):
        """K > 2 conditions: the pairwise squared-mean-difference statistic generalizes beyond two."""
        data = _synthetic_entropy_dict(n_dyads=4, conditions=("A", "B", "C"), n_trials=4, effect=0.0, seed=3)
        df = entropy_to_long_df(data)
        res = hierarchical_permutation_test(
            data=df, value_col="entropy", condition_col="condition", n_perms=200, seed=3
        )
        assert 0.0 < res["p_value"] <= 1.0
        assert res["n_dyads"] == 4

    def test_pvalue_floored_by_add_one(self):
        """The add-one estimator never returns 0: a huge effect floors p at 1/(n_perms + 1), not 0."""
        df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=4, n_trials=6, effect=50.0, seed=2))
        n_perms = 500
        res = hierarchical_permutation_test(
            data=df, value_col="entropy", condition_col="condition", n_perms=n_perms, seed=2
        )
        assert res["p_value"] == pytest.approx(1.0 / (n_perms + 1))

    def test_n_trials_per_dyad_counts_across_conditions(self):
        """n_trials_per_dyad sums permutable trial blocks across conditions (2 conditions x 4 trials = 8)."""
        df = entropy_to_long_df(
            _synthetic_entropy_dict(n_dyads=3, conditions=("A", "B"), n_trials=4, effect=0.0, seed=1)
        )
        res = hierarchical_permutation_test(
            data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0
        )
        assert res["n_trials_per_dyad"] == {0: 8, 1: 8, 2: 8}

    def test_row_order_invariance(self):
        # Regression: groupby index labels were once used as positional
        # indices, so the result depended on the dataframe's row order.
        df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=3, effect=0.0, seed=1))
        kwargs = {"value_col": "entropy", "condition_col": "condition", "n_perms": 200, "seed": 7}
        res_a = hierarchical_permutation_test(data=df, **kwargs)
        res_b = hierarchical_permutation_test(data=df.sample(frac=1.0, random_state=0), **kwargs)
        assert res_a["p_value"] == res_b["p_value"]

    def test_nan_condition_raises(self):
        """Refuse data with missing condition labels."""
        # Regression: a NaN condition label slipped past the uniqueness guard
        # (nunique ignores NaN), made the observed statistic NaN, and the test
        # silently returned the minimum possible p-value for any data.
        df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=3, effect=0.0, seed=1))
        df.loc[5, "condition"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            hierarchical_permutation_test(data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0)

    def test_nan_value_raises(self):
        """Refuse data with missing values."""
        # pandas mean() silently skips NaN values (a complete-case analysis the
        # caller never asked for); the function must refuse instead.
        df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=3, effect=0.0, seed=1))
        df.loc[5, "entropy"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            hierarchical_permutation_test(data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0)

    def test_nan_dyad_or_trial_raises(self):
        """Refuse missing dyad/trial ids (groupby silently drops NaN keys, biasing the null)."""
        # Regression: the validator guarded condition/value but not dyad/trial.
        # A NaN there passed validation, then groupby dropped the row from the
        # permutation index, freezing its label and biasing the null (measured:
        # trial_id=NaN -> p=0.99, a row silently excluded).
        for col in ("dyad", "trial_id"):
            df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=2, n_trials=3, effect=0.0, seed=1))
            df.loc[df.index[0], col] = np.nan
            with pytest.raises(ValueError, match="missing values"):
                hierarchical_permutation_test(
                    data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0
                )

    def test_single_condition_per_dyad_raises(self):
        """Refuse a between-subjects design: within-dyad permutation cannot move any label."""
        # Regression: each dyad in only one condition silently returned p=1.0 for
        # ANY effect (measured: A=0 vs B=100 across 6 dyads -> p=1.0).
        rows = [
            {
                "dyad": d,
                "condition": "A" if d < 3 else "B",
                "trial_id": f"{d}__x__{t}",
                "entropy": 0.0 if d < 3 else 100.0,
            }
            for d in range(6)
            for t in range(4)
        ]
        with pytest.raises(ValueError, match="within-dyad permutation cannot alter"):
            hierarchical_permutation_test(
                data=pd.DataFrame(rows), value_col="entropy", condition_col="condition", n_perms=50, seed=0
            )

    def test_tiny_permutation_space_warns(self):
        """Warn when one trial per condition makes significance structurally unreachable."""
        rows = [
            {"dyad": d, "condition": c, "trial_id": f"{d}__{c}__0", "entropy": v}
            for d in range(3)
            for c, v in [("A", 0.0), ("B", 1.0)]
        ]
        with pytest.warns(UserWarning, match="structurally impossible"):
            hierarchical_permutation_test(
                data=pd.DataFrame(rows), value_col="entropy", condition_col="condition", n_perms=50, seed=0
            )

    def test_non_positive_n_perms_raises(self):
        """Refuse n_perms < 1 (zero permutations once returned a meaningless p=1.0)."""
        df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=2, n_trials=3, effect=0.0, seed=1))
        for n in (0, -5):
            with pytest.raises(ValueError, match="n_perms must be"):
                hierarchical_permutation_test(
                    data=df, value_col="entropy", condition_col="condition", n_perms=n, seed=0
                )

    def test_empty_data_raises(self):
        """Refuse an empty frame (once reported p=1.0 on zero data)."""
        df = pd.DataFrame({"entropy": [], "condition": [], "dyad": [], "trial_id": []})
        with pytest.raises(ValueError, match="Empty data"):
            hierarchical_permutation_test(data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0)

    def test_mixed_type_trial_ids_run(self):
        """Accept hand-built data with mixed-type trial ids."""
        # Hand-built data may mix int and str trial ids; sorting by string form
        # must not crash, and the result must stay valid.
        df = pd.DataFrame(
            {
                "dyad": [0] * 8,
                "condition": ["A", "A", "B", "B"] * 2,
                "trial_id": [0, 1, "t2", "t3"] * 2,
                "entropy": np.arange(8, dtype=float),
            }
        )
        res = hierarchical_permutation_test(
            data=df, value_col="entropy", condition_col="condition", n_perms=20, seed=0
        )
        assert 0.0 < res["p_value"] <= 1.0

    def test_colliding_trial_ids_raise(self):
        # A trial id that restarts per condition merges blocks across
        # conditions; the function must refuse rather than degenerate.
        df = pd.DataFrame(
            {
                "dyad": [0] * 8,
                "condition": ["A"] * 4 + ["B"] * 4,
                "trial_id": [0, 0, 1, 1, 0, 0, 1, 1],
                "entropy": np.arange(8, dtype=float),
            }
        )
        with pytest.raises(ValueError, match="more than one condition"):
            hierarchical_permutation_test(data=df, value_col="entropy", condition_col="condition", n_perms=10, seed=0)

    def test_two_sided_tail_uses_absolute_values(self):
        """Cover the two-sided branch: a signed statistic counts |null| >= |observed|."""
        df = entropy_to_long_df(_synthetic_entropy_dict(n_dyads=4, n_trials=4, effect=0.0, seed=3))

        def signed_mean_diff(d, value_col, condition_col):
            conds = sorted(d[condition_col].unique())
            a = d.loc[d[condition_col] == conds[0], value_col].mean()
            b = d.loc[d[condition_col] == conds[1], value_col].mean()
            return float(a - b)  # can be negative, unlike the squared default

        res = hierarchical_permutation_test(
            data=df,
            value_col="entropy",
            condition_col="condition",
            n_perms=200,
            seed=3,
            test_stat_fn=signed_mean_diff,
            tail="two-sided",
        )
        assert 0.0 < res["p_value"] <= 1.0
        assert res["tail"] == "two-sided"

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


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
