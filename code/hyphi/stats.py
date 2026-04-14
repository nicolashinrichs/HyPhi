"""
Statistical testing for HyPhi: Mixed-effects models, permutation tests, and effect sizes.

Replaces pooled permutation tests with proper hierarchical / mixed-effects
approaches that account for the nested variance structure:
    dyad → trial → window
Years: 2026
"""

# %% Import
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def mixed_effects_test(data, formula, groups, re_formula=None):
    """
    Run a linear mixed-effects model using statsmodels.

    This properly nests variance at the dyad/trial/window levels,
    addressing the pseudo-replication concern from reviewer feedback.

    Parameters
    ----------
    data : pd.DataFrame
        Long-form dataframe with columns for the response variable,
        fixed-effect predictors, and grouping variables.
    formula : str
        Patsy/R-style formula for the fixed effects, e.g.:
        "entropy ~ condition"
    groups : str
        Column name for the grouping variable (e.g., "dyad").
    re_formula : str, optional
        Random effects formula, e.g., "1" for random intercepts.
        Defaults to "1".

    Returns
    -------
    statsmodels MixedLMResults
        Fitted model results.

    Example
    -------
    >>> import pandas as pd
    >>> # data has columns: entropy, condition, dyad, trial, window
    >>> result = mixed_effects_test(data, "entropy ~ condition", groups="dyad")
    >>> print(result.summary())

    """
    if re_formula is None:
        re_formula = "1"

    model = smf.mixedlm(formula, data, groups=data[groups], re_formula=re_formula)
    result = model.fit(reml=True)
    return result


def hierarchical_permutation_test(
    data, value_col, condition_col, group_cols, n_perms=2000, test_stat_fn=None, seed=42
):
    """
    Hierarchical permutation test respecting nested structure.

    Permutes condition labels within each group level, preserving the
    dependency structure (e.g., permute within dyads, not across them).

    Parameters
    ----------
    data : pd.DataFrame
        Long-form data.
    value_col : str
        Column name of the response variable.
    condition_col : str
        Column name of the condition label to permute.
    group_cols : list[str]
        Nesting hierarchy, e.g., ["dyad", "trial"].
        Labels are permuted at the highest (first) level.
    n_perms : int
        Number of permutations.
    test_stat_fn : callable, optional
        Function(data, value_col, condition_col) → float.
        Defaults to difference of means.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {
            'observed_stat': float,
            'null_distribution': np.ndarray of shape (n_perms,),
            'p_value': float
        }

    """
    rng = np.random.default_rng(seed)

    if test_stat_fn is None:

        def test_stat_fn(df, val_col, cond_col):
            conditions = df[cond_col].unique()
            if len(conditions) != 2:
                raise ValueError("Exactly two conditions required for default test stat.")
            a = df[df[cond_col] == conditions[0]][val_col].values
            b = df[df[cond_col] == conditions[1]][val_col].values
            return np.mean(a) - np.mean(b)

    # Observed test statistic
    observed = test_stat_fn(data, value_col, condition_col)

    # Null distribution
    null_dist = np.zeros(n_perms)
    top_group = group_cols[0]
    unique_groups = data[top_group].unique()

    for i in range(n_perms):
        perm_data = data.copy()
        # Permute condition labels within each top-level group
        for grp in unique_groups:
            mask = perm_data[top_group] == grp
            labels = perm_data.loc[mask, condition_col].values.copy()
            rng.shuffle(labels)
            perm_data.loc[mask, condition_col] = labels

        null_dist[i] = test_stat_fn(perm_data, value_col, condition_col)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))

    return {
        "observed_stat": observed,
        "null_distribution": null_dist,
        "p_value": p_value,
    }


def cohens_d(group_a, group_b):
    """
    Compute Cohen's d effect size between two groups.

    Parameters
    ----------
    group_a, group_b : np.ndarray
        1-D arrays of observations.

    Returns
    -------
    float
        Cohen's d (positive means group_a > group_b).

    """
    na, nb = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    # Pooled standard deviation
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd < 1e-15:
        return 0.0

    return float((mean_a - mean_b) / pooled_sd)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
