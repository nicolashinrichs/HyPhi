"""
Hierarchical proof-of-concept statistical runner for HyPhi.

Runs **alongside** ``hyper_ccor_ragg_frc.py``'s pooled energy-distance tests
— the old pipeline is untouched — and writes a parallel JSON report:

    hierarchical_stats_{FRC|AFRC}_n_perm_{N}_config_{ID}.json

What this script does:

1. **Preflight dyad check** — inspects what has actually been loaded (not just
   what the config declares), categorises the statistical regime (single-dyad
   demo / proof-of-concept / small-sample / exploratory), and decides which
   analyses are statistically meaningful *before* the permutation loops run.
   Any stage that can't run given the current N is skipped with a reason.
2. **Hierarchical permutation**: permutes condition labels *within dyad, at
   the trial level*, keeping window blocks intact — no more pooling across
   dyads/trials/windows. (``hyphi.stats.hierarchical_permutation_test``)
3. **Mixed-effects model**: fits ``entropy ~ C(condition)`` with dyad-level
   random intercept and a trial-level nested variance component. Requires
   ≥2 dyads. (``hyphi.stats.mixed_effects_test``)
4. **Hierarchical energy distance**: per-dyad pairwise energy distance
   averaged across dyads, with the same hierarchical permutation for null.
5. **Benchmark head-to-head**: PLV/wPLI-like block summaries and
   graph-theoretic metrics (modularity, global efficiency, assortativity,
   clustering) extracted from the same windowed CCORR graphs, then a
   cross-validated classifier with ``StratifiedGroupKFold`` on dyad (≥2
   dyads) or ``StratifiedKFold`` (N=1) compares curvature-entropy features
   vs. benchmarks vs. their concatenation. (``hyphi.benchmarks``)
6. **Null models**: dyad-label shuffle sanity check (requires ≥2 dyads);
   within-dyad condition-label surrogate cross-check. (``hyphi.null_models``)
7. **Effect sizes**: Cohen's d scalar and per-window time series for each
   condition pair; minimum N dyads required for 80% power at alpha=0.05.
   (``hyphi.stats.cohens_d``, ``cohens_d_timeseries``, ``required_sample_size``)
8. **Adaptive framing**: the banner and JSON framing text are generated
   from the actual loaded N; the word "proof-of-concept" sticks until the
   sample size comfortably exceeds the required-N implied by the observed
   effect sizes.

Usage
-----
    python hierarchical_stats_runner.py <config.toml> <FRC|AFRC> [--strict]

    --strict   abort if no inferential analyses are runnable given the
               loaded N; otherwise the runner still produces descriptive
               effect sizes and power estimates.

Years: 2026
"""

from __future__ import annotations

# %% Import
import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hyphi.benchmarks import classify_curvature_vs_benchmarks, extract_window_features
from hyphi.configs import paths
from hyphi.io import load_config
from hyphi.null_models import dyad_label_shuffle
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
logger = logging.getLogger("hyphi.hierarchical_stats_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s :: %(message)s")

# Regime thresholds (inclusive lower bounds).  Tunable; every stage gates
# on explicit capabilities, not on the regime label.
_REGIME_SINGLE = 1
_REGIME_POC = 2
_REGIME_SMALL = 5
_REGIME_EXPLORATORY = 10


# %% Preflight data class  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@dataclass
class Capabilities:
    """Outcome of the preflight check: what this N can and cannot support."""

    n_dyads_loaded: int
    n_dyads_configured: int
    n_dyads_complete: int  # dyads with every condition present
    n_conditions_loaded: int
    n_conditions_configured: int
    regime: str
    can_hierarchical_perm: bool
    can_mixed_effects: bool
    can_group_cv: bool
    can_dyad_null_sanity: bool
    can_effect_sizes: bool
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# ---------------------
# Path constructors (matched to hyper_ccorr_{frc,aug_frc}.py output naming)
# ---------------------


def _prefix(curvature: str) -> str:
    return "CCORR_FRC" if curvature == "FRC" else "CCORR_aug_FRC"


def entropy_path(config: dict, dyad: Any, trial_type: str, curvature: str) -> Path:
    """Per-dyad × per-trial-type entropy array path (shape: n_freq × n_trials × n_windows)."""
    base = Path(config["result_loc"]).resolve()
    return base / (
        f"{_prefix(curvature)}_entropy_dyad_{dyad}_trial_type_{trial_type}_config_{config['config_id']}.npy"
    )


def curvature_matrix_path(config: dict, dyad: Any, trial_type: str, curvature: str) -> Path:
    """Per-dyad × per-trial-type curvature matrix path (shape: n_freq × n_trials × n_windows × 2n × 2n)."""
    base = Path(config["result_loc"]).resolve()
    return base / (
        f"{_prefix(curvature)}_matrix_dyad_{dyad}_trial_type_{trial_type}_config_{config['config_id']}.npy"
    )


# ---------------------
# Loaders
# ---------------------


def load_entropy_arrays(config: dict, curvature: str) -> dict[Any, dict[str, np.ndarray]]:
    """Return ``{dyad: {trial_type: array(n_freq, n_trials, n_windows)}}``."""
    out: dict[Any, dict[str, np.ndarray]] = {}
    for d in config["dyads"]:
        per_dyad: dict[str, np.ndarray] = {}
        for tt in config["trial_types"]:
            p = entropy_path(config, d, tt, curvature)
            if not p.exists():
                logger.warning("Missing entropy file: %s", p)
                continue
            per_dyad[tt] = np.load(p)
        if per_dyad:
            out[d] = per_dyad
    return out


def load_curvature_arrays(config: dict, curvature: str) -> dict[Any, dict[str, np.ndarray]]:
    """Return ``{dyad: {trial_type: array(n_freq, n_trials, n_windows, 2n, 2n)}}``."""
    out: dict[Any, dict[str, np.ndarray]] = {}
    for d in config["dyads"]:
        per_dyad: dict[str, np.ndarray] = {}
        for tt in config["trial_types"]:
            p = curvature_matrix_path(config, d, tt, curvature)
            if not p.exists():
                logger.warning("Missing curvature file: %s", p)
                continue
            per_dyad[tt] = np.load(p)
        if per_dyad:
            out[d] = per_dyad
    return out


# ---------------------
# Preflight
# ---------------------


def _classify_regime(n_dyads: int) -> str:
    if n_dyads <= 0:
        return "empty"
    if n_dyads < _REGIME_POC:
        return "single_dyad_demo"
    if n_dyads < _REGIME_SMALL:
        return "proof_of_concept"
    if n_dyads < _REGIME_EXPLORATORY:
        return "small_sample"
    return "exploratory"


def preflight_check(entropies: dict, config: dict) -> Capabilities:
    """
    Crucial gate — inspect what actually loaded and decide which analyses to run.

    Called before any permutation loop. Distinguishes:
      * n_dyads_loaded — dyads with at least one condition on disk.
      * n_dyads_complete — dyads that have *every* configured condition.

    Capability flags are derived from these counts:
      * can_hierarchical_perm — ≥1 dyad, ≥2 conditions.
      * can_mixed_effects    — ≥2 dyads (random-intercept variance estimable).
      * can_group_cv         — ≥2 dyads (StratifiedGroupKFold folds ≥2).
      * can_dyad_null_sanity — ≥2 dyads (shuffling 1 label is a no-op).
      * can_effect_sizes     — ≥1 dyad, ≥2 conditions with ≥2 observations each.

    """
    conditions_configured = list(config["trial_types"])
    n_dyads_loaded = len(entropies)
    n_dyads_configured = len(config["dyads"])

    # How many conditions actually loaded anywhere
    loaded_conditions = set()
    for by_cond in entropies.values():
        loaded_conditions.update(by_cond.keys())
    n_conds_loaded = len(loaded_conditions)

    # Dyads that have the full set of configured conditions
    n_dyads_complete = sum(
        1 for by_cond in entropies.values() if set(by_cond.keys()) == set(conditions_configured)
    )

    regime = _classify_regime(n_dyads_loaded)

    caps = Capabilities(
        n_dyads_loaded=n_dyads_loaded,
        n_dyads_configured=n_dyads_configured,
        n_dyads_complete=n_dyads_complete,
        n_conditions_loaded=n_conds_loaded,
        n_conditions_configured=len(conditions_configured),
        regime=regime,
        can_hierarchical_perm=(n_dyads_loaded >= 1 and n_conds_loaded >= 2),
        can_mixed_effects=(n_dyads_loaded >= 2 and n_conds_loaded >= 2),
        can_group_cv=(n_dyads_loaded >= 2),
        can_dyad_null_sanity=(n_dyads_loaded >= 2),
        can_effect_sizes=(n_dyads_loaded >= 1 and n_conds_loaded >= 2),
    )

    # Diagnostic notes
    if n_dyads_loaded < n_dyads_configured:
        caps.notes.append(
            f"Only {n_dyads_loaded} of {n_dyads_configured} configured dyads loaded "
            "(missing .npy files). Continuing with whatever loaded."
        )
    if n_dyads_complete < n_dyads_loaded:
        caps.notes.append(
            f"{n_dyads_loaded - n_dyads_complete} dyad(s) are missing one or more conditions; "
            "they will participate only where their subset of conditions is sufficient."
        )
    if n_conds_loaded < 2:
        caps.notes.append(
            f"Only {n_conds_loaded} condition(s) loaded; between-condition inference disabled."
        )
    if not caps.can_mixed_effects and caps.can_hierarchical_perm:
        caps.notes.append(
            "Mixed-effects skipped: need ≥2 dyads to estimate a dyad-level variance component."
        )
    if not caps.can_group_cv and caps.can_hierarchical_perm:
        caps.notes.append(
            "Classifier falls back to StratifiedKFold (non-group-aware): only one dyad present."
        )

    return caps


def regime_framing(caps: Capabilities) -> str:
    """Human-readable framing paragraph derived from the capabilities."""
    n = caps.n_dyads_loaded
    if caps.regime == "empty":
        return "No dyads loaded — every inferential analysis is disabled."
    if caps.regime == "single_dyad_demo":
        return (
            f"Single-dyad demonstration (N={n}). Between-dyad inference is not defined; "
            "results below are within-dyad descriptives and the method demo only. "
            "Do not quote p-values as confirmatory."
        )
    if caps.regime == "proof_of_concept":
        return (
            f"Proof-of-concept run (N={n}). Severely underpowered for condition-level "
            "inference. Read the output as method demonstration, not as confirmatory "
            "tests. See `effect_sizes_and_power` for the minimum N dyads required to "
            "reach 80% power at alpha=0.05 for each observed Cohen's d."
        )
    if caps.regime == "small_sample":
        return (
            f"Small-sample preliminary (N={n}). Inference is feasible but modestly powered; "
            "focus on effect sizes and 95% CIs rather than p-value thresholds. "
            "Confirm any borderline effect in a pre-registered larger sample."
        )
    return (
        f"Exploratory run (N={n}). Standard hierarchical inference proceeds; "
        "confirmatory claims still depend on pre-registration and on the observed "
        "effect size relative to the required-N in `effect_sizes_and_power`."
    )


def preflight_banner(caps: Capabilities, curvature: str, config: dict) -> str:
    lines = [
        "",
        "========================= PREFLIGHT =========================",
        f"Curvature:                   {curvature}",
        f"Config ID:                   {config['config_id']}",
        f"Dyads configured:            {caps.n_dyads_configured}",
        f"Dyads loaded:                {caps.n_dyads_loaded}",
        f"Dyads complete (all conds):  {caps.n_dyads_complete}",
        f"Conditions configured:       {caps.n_conditions_configured}",
        f"Conditions loaded:           {caps.n_conditions_loaded}",
        f"Statistical regime:          {caps.regime}",
        "",
        "Capability gates:",
        f"  * hierarchical permutation : {caps.can_hierarchical_perm}",
        f"  * mixed-effects model      : {caps.can_mixed_effects}",
        f"  * group-aware classifier CV: {caps.can_group_cv}",
        f"  * dyad-label null sanity   : {caps.can_dyad_null_sanity}",
        f"  * effect sizes + power     : {caps.can_effect_sizes}",
    ]
    if caps.notes:
        lines.append("")
        lines.append("Notes:")
        for note in caps.notes:
            lines.append(f"  - {note}")
    lines.append("")
    lines.append(regime_framing(caps))
    lines.append("=============================================================")
    return "\n".join(lines)


# ---------------------
# Analysis stages
# ---------------------


def run_hierarchical_tests(
    entropies: dict, freq_bands: list[str], n_perms: int, seed: int
) -> tuple[dict, pd.DataFrame]:
    """Run hierarchical permutation + energy-distance tests per frequency band."""
    long_df = entropy_to_long_df(entropies, freq_bands=freq_bands, value_col="entropy")
    results: dict[str, Any] = {}
    for band in long_df["freq"].unique():
        d_band = long_df[long_df["freq"] == band].copy()
        logger.info(
            "Hierarchical tests, band=%s: n_rows=%d, n_dyads=%d",
            band,
            len(d_band),
            d_band["dyad"].nunique(),
        )
        perm_res = hierarchical_permutation_test(
            data=d_band,
            value_col="entropy",
            condition_col="condition",
            dyad_col="dyad",
            trial_col="trial_id",
            n_perms=n_perms,
            seed=seed,
        )
        edist_res = energy_distance_hierarchical(
            data=d_band,
            value_col="entropy",
            condition_col="condition",
            dyad_col="dyad",
            trial_col="trial_id",
            n_perms=n_perms,
            seed=seed,
        )
        results[str(band)] = {
            "hierarchical_permutation": {
                "observed_stat": perm_res["observed_stat"],
                "p_value": perm_res["p_value"],
                "n_perms": perm_res["n_perms"],
                "n_dyads": perm_res["n_dyads"],
            },
            "hierarchical_energy_distance": {
                "observed_stat": edist_res["observed_stat"],
                "p_value": edist_res["p_value"],
                "n_perms": edist_res["n_perms"],
            },
        }
    return results, long_df


def run_mixed_effects(long_df: pd.DataFrame, freq_bands: list[str]) -> dict:
    """Fit a MixedLM per frequency band: random intercept on dyad, trial-level variance component."""
    results: dict[str, Any] = {}
    for band in long_df["freq"].unique():
        d_band = long_df[long_df["freq"] == band].copy()
        n_dyads = int(d_band["dyad"].nunique())
        if n_dyads < 2:
            results[str(band)] = {"fitted": False, "reason": "n_dyads<2"}
            continue
        try:
            fit = mixed_effects_test(
                data=d_band,
                formula="entropy ~ C(condition)",
                groups="dyad",
                re_formula="1",
                vc_formula={"trial": "0 + C(trial_id)"},
            )
            results[str(band)] = {
                "fitted": True,
                "n_dyads": n_dyads,
                "n_observations": int(len(d_band)),
                "fixed_effects": {k: float(v) for k, v in fit.params.items()},
                "p_values": {k: float(v) for k, v in fit.pvalues.items()},
                "log_likelihood": float(fit.llf),
                "aic": float(fit.aic) if hasattr(fit, "aic") else None,
            }
        except Exception as e:  # statsmodels can fail on rank-deficient designs
            logger.warning("MixedLM failed for band %s: %s", band, e)
            results[str(band)] = {"fitted": False, "reason": str(e)}
    return results


def run_effect_sizes(entropies: dict, conditions: list[str], freq_bands: list[str]) -> dict:
    """Cohen's d scalar + per-window time series + required-N power analysis."""
    stacks: dict[str, dict[str, np.ndarray]] = {c: {} for c in conditions}
    for c in conditions:
        for f, band in enumerate(freq_bands):
            arrs = []
            for _d, by_cond in entropies.items():
                if c not in by_cond:
                    continue
                arrs.append(by_cond[c][f])  # (n_trials, n_windows)
            if arrs:
                stacks[c][band] = np.concatenate(arrs, axis=0)  # (total_trials, n_windows)

    results: dict[str, Any] = {}
    for i, ca in enumerate(conditions):
        for cb in conditions[i + 1:]:
            for band in freq_bands:
                if band not in stacks.get(ca, {}) or band not in stacks.get(cb, {}):
                    continue
                a = stacks[ca][band]
                b = stacks[cb][band]
                if min(a.size, b.size) < 2:
                    continue
                d_scalar = cohens_d(a.ravel(), b.ravel())
                d_ts = cohens_d_timeseries(a, b, axis=0)
                power = required_sample_size(d_scalar)
                results[f"{band}__{ca}_vs_{cb}"] = {
                    "cohens_d": float(d_scalar),
                    "cohens_d_per_window": d_ts.tolist(),
                    "required_n_per_group": power["n_per_group"],
                    "power_alpha": power["alpha"],
                    "power_target": power["power"],
                }
    return results


def run_classifier_comparison(
    entropies: dict,
    curvatures: dict,
    config: dict,
    n_ch_per_subject: int,
    use_group_cv: bool,
) -> dict:
    """Curvature-entropy vs. benchmark features, with dyad-aware CV when possible."""
    conditions = config["trial_types"]
    cond_idx = {c: i for i, c in enumerate(conditions)}

    X_curv_list, X_bench_list, y_list, groups_list = [], [], [], []
    for d, by_cond in entropies.items():
        for c, H in by_cond.items():
            if d not in curvatures or c not in curvatures[d]:
                logger.warning("Curvature missing for dyad=%s cond=%s; skipping.", d, c)
                continue
            C = curvatures[d][c]  # (n_freq, n_trials, n_windows, 2n, 2n)
            n_freq, n_trials, n_windows = H.shape
            bench_feats, _names = extract_window_features(C, n_ch_per_subject=n_ch_per_subject)
            X_curv_list.append(H.transpose(1, 2, 0).reshape(n_trials * n_windows, n_freq))
            X_bench_list.append(bench_feats.transpose(1, 2, 0, 3).reshape(n_trials * n_windows, -1))
            y_list.extend([cond_idx[c]] * (n_trials * n_windows))
            groups_list.extend([d] * (n_trials * n_windows))

    if not X_curv_list:
        logger.warning("No samples assembled — classifier comparison skipped.")
        return {"skipped": True, "reason": "no_data"}

    X_curv = np.concatenate(X_curv_list, axis=0)
    X_bench = np.concatenate(X_bench_list, axis=0)
    y = np.asarray(y_list)
    groups_arr = np.asarray(groups_list)
    X_combined = np.concatenate([X_curv, X_bench], axis=1)

    n_groups = int(len(np.unique(groups_arr)))
    logger.info(
        "Classifier samples: %d rows, %d classes, %d groups (dyads); group-CV=%s",
        X_curv.shape[0],
        len(np.unique(y)),
        n_groups,
        use_group_cv,
    )
    groups_for_cv = groups_arr if (use_group_cv and n_groups >= 2) else None
    cv = max(2, min(5, n_groups)) if groups_for_cv is not None else 5

    return classify_curvature_vs_benchmarks(
        X_curvature=X_curv,
        X_benchmarks=X_bench,
        y=y,
        groups=groups_for_cv,
        cv=cv,
        X_combined=X_combined,
    )


def run_null_model_sanity(long_df: pd.DataFrame, freq_bands: list[str], n_perms: int, seed: int) -> dict:
    """Baseline: rerun hierarchical permutation with scrambled dyad labels."""
    rng = np.random.default_rng(seed)
    shuffled = dyad_label_shuffle(long_df["dyad"].to_numpy(), rng=rng)
    scrambled_df = long_df.assign(dyad=shuffled)
    results: dict[str, Any] = {}
    for band in freq_bands:
        d_band = scrambled_df[scrambled_df["freq"] == band].copy()
        if d_band["dyad"].nunique() < 2:
            continue
        res = hierarchical_permutation_test(
            data=d_band,
            value_col="entropy",
            condition_col="condition",
            dyad_col="dyad",
            trial_col="trial_id",
            n_perms=n_perms,
            seed=seed,
        )
        results[str(band)] = {"p_value_under_dyad_scramble": res["p_value"]}
    return results


# ---------------------
# Output helpers
# ---------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_results_json(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, default=_json_default)
    logger.info("Report saved: %s", out_path)


# ---------------------
# Main
# ---------------------


def main(config_file: str, curvature: str, strict: bool = False) -> dict:
    """Entry point."""
    assert curvature in {"FRC", "AFRC"}, f"Curvature must be FRC or AFRC; got {curvature!r}"

    config_path = os.path.join(paths.experiments.configs, config_file)
    config = load_config(config_path)
    freq_bands = list(
        config.get("freq_bands", [f"band{i}" for i in range(int(config.get("num_freqs", 8)))])
    )

    # --- Load (cheap) ------------------------------------------------------
    entropies = load_entropy_arrays(config, curvature)

    # --- Preflight (crucial gate) -----------------------------------------
    caps = preflight_check(entropies, config)
    print(preflight_banner(caps, curvature, config))

    if caps.n_dyads_loaded == 0:
        msg = "No dyads loaded — aborting."
        logger.error(msg)
        return {"error": "no_entropy_data", "searched_under": str(Path(config["result_loc"]).resolve())}

    if strict and not (caps.can_hierarchical_perm or caps.can_mixed_effects):
        logger.error("--strict: no inferential analyses are runnable at N=%d; aborting.", caps.n_dyads_loaded)
        return {"error": "strict_no_inference_possible", "capabilities": caps.to_json()}

    n_perms = int(config.get("nhst_perm", 2000))

    # --- Hierarchical permutation / energy distance -----------------------
    if caps.can_hierarchical_perm:
        hier_res, long_df = run_hierarchical_tests(entropies, freq_bands, n_perms, seed=42)
    else:
        hier_res = {"skipped": True, "reason": "preflight: <2 conditions or 0 dyads"}
        long_df = entropy_to_long_df(entropies, freq_bands=freq_bands, value_col="entropy")

    # --- Mixed-effects ----------------------------------------------------
    if caps.can_mixed_effects:
        mixed_res = run_mixed_effects(long_df, freq_bands)
    else:
        mixed_res = {"skipped": True, "reason": f"preflight: n_dyads_loaded={caps.n_dyads_loaded} < 2"}

    # --- Effect sizes + power --------------------------------------------
    if caps.can_effect_sizes:
        effect_res = run_effect_sizes(entropies, config["trial_types"], freq_bands)
    else:
        effect_res = {"skipped": True, "reason": "preflight: insufficient observations per condition"}

    # --- Null-model sanity ------------------------------------------------
    if caps.can_dyad_null_sanity:
        null_res = run_null_model_sanity(long_df, freq_bands, n_perms=min(n_perms, 500), seed=42)
    else:
        null_res = {"skipped": True, "reason": f"preflight: n_dyads_loaded={caps.n_dyads_loaded} < 2"}

    # --- Classifier comparison -------------------------------------------
    curvatures = load_curvature_arrays(config, curvature)
    n_ch_per_subject = int(config.get("num_channels", 128)) // 2
    if curvatures:
        try:
            clf_res = run_classifier_comparison(
                entropies=entropies,
                curvatures=curvatures,
                config=config,
                n_ch_per_subject=n_ch_per_subject,
                use_group_cv=caps.can_group_cv,
            )
        except Exception as e:
            logger.warning("Classifier step failed: %s", e)
            clf_res = {"error": str(e)}
    else:
        clf_res = {"skipped": True, "reason": "no curvature .npy files found"}

    # --- Report -----------------------------------------------------------
    report = {
        "description": "Proof-of-concept hierarchical statistical analysis (reviewer addendum, 2026-04).",
        "framing": regime_framing(caps),
        "capabilities": caps.to_json(),
        "n_permutations": n_perms,
        "curvature": curvature,
        "config_id": config["config_id"],
        "hierarchical_tests": hier_res,
        "mixed_effects": mixed_res,
        "effect_sizes_and_power": effect_res,
        "null_model_sanity": null_res,
        "classifier_comparison": clf_res,
    }

    out_dir = Path(config.get("pooled_result_loc", paths.experiments.configs))
    out_path = out_dir / (
        f"hierarchical_stats_{curvature}_n_perm_{n_perms}_config_{config['config_id']}.json"
    )
    save_results_json(report, out_path)
    return report


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical proof-of-concept stats runner for HyPhi.")
    parser.add_argument("config_file", help="TOML config filename under experiments/configs/.")
    parser.add_argument("curvature", choices=["FRC", "AFRC"], help="Curvature type.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if preflight finds no inferential analyses are runnable.",
    )
    args = parser.parse_args()
    main(args.config_file, args.curvature, strict=args.strict)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
