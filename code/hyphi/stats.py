"""
Statistical testing for HyPhi: hierarchical permutation, mixed-effects, effect sizes, power.

This module replaces the original pooled-permutation approach with tests that
respect the dyad → trial → window nesting of hyperscanning data.  Every
entry point here is framed as a **proof-of-concept** given the small empirical
sample (N = 2 dyads in the reference dataset); see :func:`required_sample_size`
for the N implied by an observed effect at alpha=0.05, power=0.80.

Changes (2026-04):
  * Hierarchical permutation tests that permute condition labels *within* each
    dyad at the trial level, keeping window blocks intact.
  * Mixed-effects fitter wrapper (statsmodels) with dyad-level random intercept
    and optional trial-level nested variance component.
  * Cohen's d (scalar and per-window time series) for interpretable effect sizes.
  * Power-analysis helper reporting the minimum N dyads needed to detect an
    observed effect size at conventional thresholds.
  * Long-form data reshaper for arrays of shape ``(n_freq, n_trials, n_windows)``.

The old pooled tests in ``experiments/scripts/hyper_ccor_ragg_frc.py`` remain
intact for backward compatibility; this module is additive.

Years: 2026
"""

from __future__ import annotations

# %% Import
import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import energy_distance
from statsmodels.stats.power import TTestIndPower

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
logger = logging.getLogger(__name__)

__all__ = [
    "cohens_d",
    "cohens_d_timeseries",
    "energy_distance_hierarchical",
    "entropy_to_long_df",
    "hierarchical_permutation_test",
    "mixed_effects_test",
    "required_sample_size",
]


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ---------------------
# Data reshaping
# ---------------------


def entropy_to_long_df(
    entropy_by_dyad_condition: Mapping[Any, Mapping[Any, np.ndarray]],
    freq_bands: list[str] | None = None,
    value_col: str = "entropy",
) -> pd.DataFrame:
    """
    Reshape nested per-dyad, per-condition entropy arrays to a long dataframe.

    Parameters
    ----------
    entropy_by_dyad_condition : mapping
        ``{dyad_id: {condition: np.ndarray of shape (n_freq, n_trials, n_windows)}}``.
    freq_bands : list[str], optional
        Names for the frequency axis.  If ``None``, integer indices are used.
    value_col : str
        Name of the resulting value column.

    Returns
    -------
    pd.DataFrame
        Long-form dataframe with columns
        ``[dyad, condition, freq, trial, trial_id, window, value_col]``.

    Notes
    -----
    The ``trial_id`` column is ``f"{dyad}__{trial}"``, a concatenation of dyad
    and trial index so trials within a dyad keep a globally unique identifier.
    This is the block used by the hierarchical permutation scheme.

    """
    rows = []
    for dyad, by_cond in entropy_by_dyad_condition.items():
        for cond, arr in by_cond.items():
            if arr.ndim != 3:
                msg = (
                    f"Expected (n_freq, n_trials, n_windows) for dyad={dyad} "
                    f"cond={cond}, got shape {arr.shape}."
                )
                raise ValueError(msg)
            n_freq, n_trials, n_windows = arr.shape
            for f in range(n_freq):
                band = freq_bands[f] if freq_bands is not None else f
                for t in range(n_trials):
                    for w in range(n_windows):
                        rows.append(
                            {
                                "dyad": dyad,
                                "condition": cond,
                                "freq": band,
                                "trial": t,
                                "trial_id": f"{dyad}__{t}",
                                "window": w,
                                value_col: float(arr[f, t, w]),
                            }
                        )
    return pd.DataFrame(rows)


# ---------------------
# Effect sizes
# ---------------------


def cohens_d(group_a: np.ndarray, group_b: np.ndarray, paired: bool = False) -> float:
    """
    Compute Cohen's d effect size between two samples.

    Parameters
    ----------
    group_a, group_b : np.ndarray
        1-D arrays of observations.  Positive d means ``group_a > group_b``.
    paired : bool
        If True, compute the paired-sample effect size ``mean(a - b) / sd(a - b)``.

    Returns
    -------
    float
        Cohen's d.  Returns ``0.0`` if the denominator is numerically zero or
        if either sample has fewer than two observations (for the pooled case).

    """
    a = np.asarray(group_a, dtype=float).ravel()
    b = np.asarray(group_b, dtype=float).ravel()

    if paired:
        if len(a) != len(b):
            raise ValueError("Paired Cohen's d requires equal-length arrays.")
        diff = a - b
        sd = float(np.std(diff, ddof=1))
        if sd < 1e-15:
            return 0.0
        return float(np.mean(diff) / sd)

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_sd = float(np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)))
    if pooled_sd < 1e-15:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def cohens_d_timeseries(arr_a: np.ndarray, arr_b: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Cohen's d computed independently at each index along the non-``axis`` axes.

    Use this to obtain a per-window (or per-time-bin) effect-size time series
    between two conditions.

    Parameters
    ----------
    arr_a, arr_b : np.ndarray
        Arrays whose axis ``axis`` is the observation (sample) axis.  All other
        axes are treated as independent comparison points (e.g., windows or
        frequency bins) and must match between the two arrays.
    axis : int
        Sample-stacking axis.

    Returns
    -------
    np.ndarray
        Cohen's d with the ``axis`` dimension collapsed.

    """
    a = np.moveaxis(np.asarray(arr_a, dtype=float), axis, 0)
    b = np.moveaxis(np.asarray(arr_b, dtype=float), axis, 0)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"Non-sample axes must match; got {a.shape[1:]} vs {b.shape[1:]}")

    na, nb = a.shape[0], b.shape[0]
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    var_a = a.var(axis=0, ddof=1) if na > 1 else np.zeros_like(mean_a)
    var_b = b.var(axis=0, ddof=1) if nb > 1 else np.zeros_like(mean_b)
    pooled_sd = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled_sd > 1e-15, (mean_a - mean_b) / pooled_sd, 0.0)
    return d


# ---------------------
# Mixed-effects
# ---------------------


def mixed_effects_test(
    data: pd.DataFrame,
    formula: str,
    groups: str,
    re_formula: str | None = None,
    vc_formula: dict | None = None,
    reml: bool = True,
):
    """
    Fit a linear mixed-effects model with dyad-level random intercept.

    Respects dyad → trial → window nesting: pass ``groups="dyad"`` for the
    top-level random intercept, and optionally
    ``vc_formula={"trial": "0 + C(trial_id)"}`` for a trial-level variance
    component nested within each dyad.

    Parameters
    ----------
    data : pd.DataFrame
        Long-form data.
    formula : str
        Patsy fixed-effects formula (e.g. ``"entropy ~ C(condition)"``).
    groups : str
        Column holding the top-level grouping variable (e.g. ``"dyad"``).
    re_formula : str, optional
        Random-effects formula for the group level.  Defaults to ``"1"``
        (random intercept only).
    vc_formula : dict, optional
        Variance-component formula for nested random effects,
        e.g. ``{"trial": "0 + C(trial_id)"}``.
    reml : bool
        Whether to fit by REML (default) or ML.

    Returns
    -------
    statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted-model results object.

    Examples
    --------
    >>> # data: long-form with columns entropy, condition, dyad, trial_id
    >>> result = mixed_effects_test(
    ...     data,
    ...     formula="entropy ~ C(condition)",
    ...     groups="dyad",
    ...     vc_formula={"trial": "0 + C(trial_id)"},
    ... )
    >>> print(result.summary())

    """
    if re_formula is None:
        re_formula = "1"

    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data[groups],
        re_formula=re_formula,
        vc_formula=vc_formula,
    )
    return model.fit(reml=reml)


# ---------------------
# Hierarchical permutation tests
# ---------------------


def _default_test_stat(df: pd.DataFrame, value_col: str, condition_col: str) -> float:
    """
    Multi-group test statistic: sum of squared pairwise mean differences.

    Equivalent to ``|mean_a - mean_b|²`` for K=2 groups, generalises smoothly
    to K>2, and is permutation-valid under the sharp null of label
    exchangeability.  Always non-negative (use ``tail="right"``).
    """
    conditions = np.unique(df[condition_col].values)
    means = {c: df.loc[df[condition_col] == c, value_col].mean() for c in conditions}
    total = 0.0
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            total += (means[conditions[i]] - means[conditions[j]]) ** 2
    return float(total)


def hierarchical_permutation_test(
    data: pd.DataFrame,
    value_col: str,
    condition_col: str,
    dyad_col: str = "dyad",
    trial_col: str = "trial_id",
    n_perms: int = 2000,
    test_stat_fn: Callable[[pd.DataFrame, str, str], float] | None = None,
    seed: int = 42,
    tail: str = "right",
) -> dict[str, Any]:
    """
    Hierarchical permutation test respecting dyad → trial → window nesting.

    Under the null, condition labels are assumed exchangeable **within** each
    dyad at the trial level.  All rows belonging to the same ``trial_id`` keep
    the same condition label during permutation, preserving the
    window-within-trial block.  Dyad identities are never crossed, which
    avoids the pseudo-replication the reviewer flagged.

    Parameters
    ----------
    data : pd.DataFrame
        Long-form data containing ``value_col``, ``condition_col``,
        ``dyad_col`` and ``trial_col``.
    value_col : str
        Response variable column.
    condition_col : str
        Condition label permuted at the trial level.
    dyad_col : str
        Top-level grouping column; permutation stays within each dyad.
    trial_col : str
        Trial identifier; within a dyad, all rows sharing this value are a
        block that keeps a single condition label.
    n_perms : int
        Number of permutations.
    test_stat_fn : callable, optional
        ``(df, value_col, condition_col) -> float``.  Defaults to
        :func:`_default_test_stat` (sum of squared pairwise mean differences).
    seed : int
        RNG seed for reproducibility.
    tail : {"right", "two-sided"}
        ``"right"`` for non-negative test statistics such as the default
        (this is the correct default); ``"two-sided"`` for statistics that
        can take either sign.

    Returns
    -------
    dict
        Keys: ``observed_stat``, ``null_distribution``, ``p_value``,
        ``n_perms``, ``n_dyads``, ``n_trials_per_dyad``, ``tail``.

    """
    required = {value_col, condition_col, dyad_col, trial_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if test_stat_fn is None:
        test_stat_fn = _default_test_stat

    rng = np.random.default_rng(seed)
    observed = float(test_stat_fn(data, value_col, condition_col))

    # Per-dyad: trial ids and the single condition each carries.
    dyad_trial_info: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
    n_trials_per_dyad: dict[Any, int] = {}
    for d, d_df in data.groupby(dyad_col):
        trial_ids = d_df[trial_col].drop_duplicates().to_numpy()
        trial_conds = (
            d_df.drop_duplicates(subset=[trial_col]).set_index(trial_col).loc[trial_ids, condition_col].to_numpy()
        )
        dyad_trial_info[d] = (trial_ids, trial_conds)
        n_trials_per_dyad[d] = int(len(trial_ids))

    trial_to_row_idx: dict[tuple[Any, Any], np.ndarray] = {}
    for (d, t), rows in data.groupby([dyad_col, trial_col]).groups.items():
        trial_to_row_idx[(d, t)] = np.asarray(rows)

    null_dist = np.empty(n_perms, dtype=float)
    perm_condition = data[condition_col].to_numpy().copy()
    for i in range(n_perms):
        for d, (trial_ids, trial_conds) in dyad_trial_info.items():
            shuffled = rng.permutation(trial_conds)
            for t, new_cond in zip(trial_ids, shuffled):
                perm_condition[trial_to_row_idx[(d, t)]] = new_cond
        data_perm = data.assign(**{condition_col: perm_condition})
        null_dist[i] = float(test_stat_fn(data_perm, value_col, condition_col))

    if tail == "right":
        p_value = float((np.sum(null_dist >= observed) + 1) / (n_perms + 1))
    elif tail == "two-sided":
        p_value = float((np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (n_perms + 1))
    else:
        raise ValueError(f"Unknown tail: {tail!r}")

    return {
        "observed_stat": observed,
        "null_distribution": null_dist,
        "p_value": p_value,
        "n_perms": int(n_perms),
        "n_dyads": int(len(dyad_trial_info)),
        "n_trials_per_dyad": n_trials_per_dyad,
        "tail": tail,
    }


def energy_distance_hierarchical(
    data: pd.DataFrame,
    value_col: str,
    condition_col: str,
    dyad_col: str = "dyad",
    trial_col: str = "trial_id",
    n_perms: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Hierarchical energy-distance test.

    Statistic: per-dyad mean pairwise energy distance between conditions,
    averaged across dyads.  Null distribution constructed by permuting
    condition labels within each dyad at the trial level (see
    :func:`hierarchical_permutation_test`).

    Parameters
    ----------
    data : pd.DataFrame
        Long-form data.
    value_col, condition_col, dyad_col, trial_col : str
        See :func:`hierarchical_permutation_test`.
    n_perms : int
        Number of permutations.
    seed : int
        RNG seed.

    Returns
    -------
    dict
        Same structure as :func:`hierarchical_permutation_test`.

    """

    def stat_fn(df: pd.DataFrame, val: str, cond: str) -> float:
        dyad_stats = []
        for _, d_df in df.groupby(dyad_col):
            conds = np.unique(d_df[cond].values)
            if len(conds) < 2:
                continue
            pairs = []
            for i in range(len(conds)):
                for j in range(i + 1, len(conds)):
                    a = d_df.loc[d_df[cond] == conds[i], val].to_numpy()
                    b = d_df.loc[d_df[cond] == conds[j], val].to_numpy()
                    if len(a) == 0 or len(b) == 0:
                        continue
                    pairs.append(energy_distance(a, b))
            if pairs:
                dyad_stats.append(float(np.mean(pairs)))
        return float(np.mean(dyad_stats)) if dyad_stats else 0.0

    return hierarchical_permutation_test(
        data=data,
        value_col=value_col,
        condition_col=condition_col,
        dyad_col=dyad_col,
        trial_col=trial_col,
        n_perms=n_perms,
        test_stat_fn=stat_fn,
        seed=seed,
        tail="right",
    )


# ---------------------
# Power analysis
# ---------------------


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Minimum dyads per condition to detect an effect of the given size.

    Uses :class:`statsmodels.stats.power.TTestIndPower`, which assumes a
    between-groups two-sample t-test.  For a paired / within-dyad design
    the actual requirement is smaller by roughly sqrt(2) at zero paired
    correlation and more with higher correlation; use this as an upper
    bound for hyperscanning designs where every dyad sees every condition.

    Parameters
    ----------
    effect_size : float
        Target Cohen's d (absolute value).
    alpha : float
        Type-I error rate.
    power : float
        Target statistical power (1 - β).
    ratio : float
        Sample-size ratio of group 2 to group 1 (default 1.0 = balanced).
    alternative : {"two-sided", "larger", "smaller"}
        Direction of the alternative hypothesis.

    Returns
    -------
    dict
        Keys: ``effect_size``, ``alpha``, ``power``, ``n_per_group``,
        ``n_total``, ``alternative``.  ``n_per_group`` is ``inf`` for d == 0.

    """
    if effect_size == 0:
        return {
            "effect_size": 0.0,
            "alpha": float(alpha),
            "power": float(power),
            "n_per_group": float("inf"),
            "n_total": float("inf"),
            "alternative": alternative,
        }
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    return {
        "effect_size": float(effect_size),
        "alpha": float(alpha),
        "power": float(power),
        "n_per_group": float(n),
        "n_total": float(n * (1 + ratio)),
        "alternative": alternative,
    }


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
