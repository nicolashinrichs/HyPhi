# HyPhi Codebase Refactoring & Upgrade Walkthrough

This document outlines the major architectural changes, dependencies, tests, and new statistical/benchmark tools that were added to the HyPhi repository to improve reproducibility, maintainability, and academic rigor based on reviewer feedback.

## Phase 1: Codebase Reproducibility & Architecture

### Dependency Management
To ensure a reproducible environment, the following files were added:
- `requirements.txt` — Pins all Python dependencies used across the notebooks and scripts, including the critical `GraphRicciCurvature==0.5.3.1`.
- `environment.yml` — A Conda environment configuration file that matches `requirements.txt`.

### The `src/hyphi/` Shared Package
We completely DRY-refactored the analytical operations into a centralized, importable Python package: `src/hyphi/`.
The highly duplicated code across scripts like `HyperCCORRFRC.py` and `HyperCCORRAugFRC.py` can now rely on these shared modules.

| Module | Purpose |
|--------|---------|
| `curvatures.py` | Consolidates Forman-Ricci (FRC), Augmented FRC (AFRC), and Ollivier-Ricci (ORC) computation, along with helper functions for extracting curvature matrices from graph edges. |
| `entropies.py` | Contains all differential entropy estimators (Vasicek, Van Es, KDE plugin, Kozachenko, Rényi, Tsallis) and vectorised helpers. |
| `windowing.py` | Centralises sliding-window Phase Locking Value (PLV) graph construction (previously embedded within notebooks and simulations). |
| `simulations.py` | Provides scaffold functions for the Connectome-informed Kuramoto model with delays and the Watts-Strogatz small-world sweep. |
| `density.py` | KDE estimator factory functions. |
| `io_utils.py` | Configuration loading, directory creation, and pickle I/O helpers. |

### Entry Point & Build Automation
- `Makefile` — Added standard targets: `make install`, `make test`, `make run-simulations`, and `make clean`.
- `main.py` — A new top-level master script documenting the chronological execution of the full analysis pipeline. (Steps are commented out, waiting for empirical data).

---

## Phase 2: Unit Testing Suite

To ensure correctness and catch regressions when dependencies evolve, a complete `pytest` testing suite was added in the `tests/` directory.

- **Conventions:** Tested against lightweight, mathematically known toy graphs (`K_5`, `C_10`, `Star_6`).
- **`test_curvatures.py` (8 tests):** Confirms analytical Forman curvature values (e.g., K₅ edges -> -4, C₁₀ edges -> 0). Checks vectorised extraction helpers.
- **`test_entropies.py` (6 tests):** Validates the Vasicek and KDE Plugin estimators against exact differential entropies for Uniform and Gaussian distributions.
- **`test_windowing.py` (10 tests):** Checks sliding-window math, making sure perfectly phase-locked signals yield PLV=1.0 and uncorrelated ones approach ~0.

*All 24 pytest tests are functioning and passing successfully.*

---

## Phase 3: Statistical & Benchmark Implementations

To address pseudo-replication concerns and provide standard baselines, new statistical tools were implemented:

- **Benchmarks (`benchmarks.py`)**: Included standard connectivity metrics to serve as baselines against curvature metrics: PLV, weighted Phase Lag Index (wPLI), imaginary coherence, network modularity, and global efficiency. Also includes a skeleton for a cross-validated Support Vector Classifier.
- **Hierarchical/Mixed-Effects Stats (`stats.py`)**: 
  - `mixed_effects_test()` uses statsmodels `MixedLM` to properly nest variance (dyad -> trial -> window).
  - `hierarchical_permutation_test()` permutes conditions only within the upper group hierarchy (e.g. keeping dyads together) to form the null distribution.
  - `cohens_d()` effect size calculator.
- **Null Models (`null_models.py`)**: Added utilities to generate surrogate datasets, including amplitude-preserving FFT phase randomization, circular time-shifting per channel, and dyad-label shuffling.

## How to Work with the Updated Codebase

1. **Install:** Run `make install` to ensure you are using the precise versions (e.g., `GraphRicciCurvature 0.5.3.1`).
2. **Test:** Run `make test` locally before committing to verify core math operations.
3. **Execute:** Check `main.py` for the high-level outline of the simulation and data-analysis pipeline steps. Modify the paths in `CCORRconfig_001.toml` to point to your data before uncommenting functions in `main.py`.
