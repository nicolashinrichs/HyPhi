"""
Statistics module for HyPhi: Mixed-effects models, permutation tests, and effect sizes.

Years: 2026
"""

# %% Import
from turtle import st

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def mixed_effects_model(df: pd.DataFrame, formula: str, groups: str, vc_formula: dict | None = None):
    """
    Fit a Mixed Linear Model resolving pseudoreplication issues.

    Example:
        formula = "entropy ~ condition"
        groups = "dyad"
        vc_formula = {"trial": "0 + C(trial)"}

    """
    model = smf.mixedlm(formula, df, groups=groups, vc_formula=vc_formula)
    return model.fit()  # result


def hierarchical_permutation_test(
    data, condition_col: str = "condition", group_col: str = "dyad", n_perms: int = 1000
):
    """
    Run permutation loop.

    Skeleton for a hierarchical permutation test that shuffles conditions
    at the highest level (e.g., dyad) rather than pooling trials or windows.
    """
    # TODO: Implement full permutation loop maintaining nested structure
    pass


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_sd


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
