# Refactoring HyPhiRole & Orchestration Protocol

This plan details the code-level refactoring of HyPhi to address academic reviewer feedback, comparing current state to HyPyP and Curvature-FCN-ASD.

## Proposed Changes

### Configuration & Entry Points
#### [NEW] `pyproject.toml`
Modernized build system defining metadata, dependencies (`GraphRicciCurvature`, `mne`, `KDEpy`, `pytest`, `statsmodels`), and scripts.
#### [NEW] `Makefile`
Defines commands like `make test`, `make build`, `make format`, and `make pipeline`.
#### [NEW] `main.py`
Chronological outline script running the full pipeline end-to-end, with execution lines commented out (as requested).

---

### Phase 1: Core Logic Consolidation (`src/hyphi/`)
Moving logic from `software_module/` into a standard Python package.

#### [NEW] `src/hyphi/__init__.py`
#### [NEW] `src/hyphi/io.py`
Reading and writing connectivity graphs and datasets.
#### [NEW] `src/hyphi/analyses.py`
Contains the core FRC/AFRC computation, geometric entropies, and the sliding-window graph construction.
#### [NEW] `src/hyphi/stats.py`
Mixed-effect models, hierarchical permutation schemes, effect size (Cohen's $d$).
#### [NEW] `src/hyphi/benchmarks.py`
Classical hyperscanning metrics: PLV, wPLI, imaginary coherence, modularity, and classifier skeletons.
#### [NEW] `src/hyphi/null_models.py`
Phase randomization, circular time-shift surrogates, and dyad-shuffling.
#### [NEW] `src/hyphi/simulations.py`
Placeholders for "Connectome-informed Kuramoto with delays" and "Watts-Strogatz sweep".

#### [DELETE] `software_module/*.py`
Will deprecate/delete flat scripts inside `software_module/` once migrated logically.

---

### Phase 2: Unit Testing Scaffolding (`tests/`)

#### [NEW] `tests/conftest.py`
Fixtures for toy graphs (complete graph, ring lattice, star graph).
#### [NEW] `tests/test_analyses.py`
Tests for curvature calculations (FRC/AFRC) and entropy estimation on toy graphs with mathematical guarantees.
#### [NEW] `tests/test_sliding_window.py`
Tests for windowing array shapes and graph emission.
#### [NEW] `tests/test_stats.py`
Sanity checks for hierarchical modeling output shapes and test APIs.

## Verification Plan

### Automated Tests
- Run `pytest tests/` to verify that toy graphs produce the exact expected curvature mathematically.
- Ensure `pyproject.toml` installation works correctly via `pip install .`.

### Manual Verification
- Review `main.py` and `Makefile` to confirm the pipeline is accurately structured for reviewers.
- Verify that no heavy simulations or true data processing are run during execution (all have `TODO` placeholders or are commented out).
