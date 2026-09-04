# HyPhi — session change log

This document records everything changed from the original source code during a single
working session: the initial clone, four escalating rounds of adversarial code review with
their fixes, and a final homogenization / decluttering pass.

Two states are distinguished throughout:

- **Committed** changes live as 18 local commits on the branch `fix/coherence-pass`
  (ahead of `master`). They are the review rounds and their fixes.
- **Uncommitted** changes are the final optimization pass (this document's Part 3). They are
  staged/working-tree only. **Nothing in this session was pushed to the remote**, per
  instruction (to avoid colliding with Simon's parallel refactor).

Headline outcome:

| Metric | Original | After session |
|---|---|---|
| `ruff check code/hyphi` | 279 violations | **0** (now a gating CI step) |
| `ty check code/hyphi` | 43 diagnostics | 24 (only the optional-import plotly/seaborn pattern) |
| Test suite | 104 collected (one hung pytest collection for 6+ min) | **315 passed + 1 xfail** |
| TODO/FIXME in `*.py` | ~17 | 0 in package code |
| Docstrings | mixed styles, many missing | NumPy style throughout |
| Duplicated `load_pickle_adjacency` | 3 copies | 1 canonical (`hyphi.io`) + re-export shims |
| CI | none | install + import-sweep + tests (3.11/3.12/3.13 matrix) + gating lint + ty |

---

## Part 1 — Clone and assessment

- Cloned `https://github.com/nicolashinrichs/HyPhi.git` as a sibling repo.
- Initial smoke assessment found: a `pytest` collection that hung for 6+ minutes (a
  script-style `test_hyphi.py` ran a full-size simulation at import), two modules
  unimportable (`pynndescent` undeclared), fixture drift breaking two tests, and a
  scientifically invalid statistic in `stats.py`.

## Part 2 — Adversarial review rounds and fixes (committed, branch `fix/coherence-pass`)

Four escalating review passes were run: an inline adversarial round, then two multi-agent
review workflows (a 265-agent and a 740-agent run), each finding bugs the previous missed.
Every confirmed bug was fixed with a regression test. The 18 commits:

### Statistics (`code/hyphi/stats.py`) — the load-bearing repairs

- **Repaired the hierarchical permutation test null distribution** (`c997c92`, `3759de9`).
  `entropy_to_long_df` built `trial_id` as `dyad__trial`; the per-condition trial index
  restarts, so every `(dyad, trial)` block carried both conditions, the null distribution
  collapsed, and the test returned the minimum possible p for ANY data (a measured 100%
  false-positive rate). Fixed the id to `dyad__condition__trial`; the function now validates
  the invariant and raises instead of degenerating. Also fixed a label-vs-position bug that
  made results depend on dataframe row order.
- **Hardened input validation** (`d56cf94`, `cf01b04`, `b234317`): refuse missing/`inf`
  values in any of the value/condition/dyad/trial columns (each silently corrupted the null
  in a different way), refuse `n_perms < 1` and empty frames, sort trial ids by string form
  so mixed-type ids do not crash.
- **Refuse degenerate designs, warn on underpowered ones** (`6eeb5ef`): a between-subjects
  design (each dyad in one condition) silently returned p=1.0 for any effect; it is now
  refused. The within-dyad permutation space is checked, and the function warns when
  significance is structurally unreachable (the dual-EEG tutorial's one-trial-per-condition
  design floors p near 0.5 regardless of effect size).

### Configuration (`code/hyphi/configs.py`)

- **`init()` is cwd-neutral; added `bootstrap()`** (`d80850b`, `88291c2`): config
  initialization no longer changes the caller's working directory (it silently broke every
  relative path downstream, and on one machine chdir'd into a same-named sibling checkout).
  `bootstrap()` is the single explicit opt-in that changes directory.
- **Symlink-safe project containment** (`b234317`): `_set_wd` used string `startswith`, which
  both false-matched sibling dirs sharing a name prefix and false-mismatched symlinked paths
  (macOS `/tmp` -> `/private/tmp`). Now compares resolved paths component-wise.
- **Order-independent foreign-config validation** (`d56cf94`, `cf01b04`): validation ran on
  an accumulating module singleton, so a foreign config could pass on a previous project's
  stale attributes; it now validates the freshly merged dict and only touches globals after
  validation passes. Required table-keys must be tables, not scalars.

### Tutorial, visualization, modeling

- **Dual-EEG tutorial made runnable and honest** (`290df62`, `0b6b0d1`, `cf01b04`): the
  documented pipeline could not run (wrong dict shape into `entropy_to_long_df`) and its
  design could not detect an effect; rewritten to a within-dyad design with a power caveat.
  Fixed the section-6 plot (hard-coded palette KeyError; `np.stack` crash on
  variable-length recordings), corrected the curvature swap table (`compute_orc_vec` writes
  `ricciCurvature`, not `formanCurvature`), and split the null-model table into
  signal-domain vs design-domain (only the former are drop-ins).
- **Curvature-blind plot fixed** (`0b6b0d1`): the partition plot read only `ricciCurvature`,
  so Forman-flow graphs rendered gray edges; it now reads either Ricci variant.
- **`compute_frc_vec` keyword + `sim_graph` self-loop** (`1b39fb7`): two experiment scripts
  called `compute_frc_vec(method_val=...)` (the wrong kwarg, a guaranteed `TypeError`); and
  an unguarded branch re-added self-loops in `sim_graph`.
- **Ricci-flow `--enable-visualization` wired** (`9818412`): the flag was a no-op referencing
  undefined variables; now wired correctly with a smoke test.

### Packaging and CI

- **Declared missing deps, docs extra, version sync** (`322b63f`): `pynndescent`,
  `tqdm-joblib`; a working `docs` extra; `CITATION.cff` synced to 2.2.0 and tracked by
  bumpver.
- **Added GitHub Actions CI** (`1bedc51`, `49c042a`, later rounds): install, an import-sweep
  over every submodule, the test suite, lint and (non-gating) type-check; a Python
  3.11/3.12/3.13 matrix; `uv sync --frozen`; `ty` no longer excludes `configs.py`.

## Part 3 — Homogenization and decluttering pass (uncommitted, local working tree)

Aligned with Simon's parallel library-first refactor (which removed the hollow placeholder
`main.py`): HyPhi is a library, and the end-to-end example is the dual-EEG tutorial, not a
run-everything script. `main.py` was intentionally **not** restored.

### Lint and type baseline (the highest-consequence change)

- Declared the scientific naming convention in `pyproject` (`ignore-names` for `G`, `A`, `L`,
  `FRC`, `Gt`, ... and curvature acronyms) and turned on NumPy docstring enforcement
  (`pydocstyle convention = "numpy"`). This alone removed ~114 N803/N806 violations without
  risky renames.
- A 23-agent per-file homogenization workflow then converted every module's docstrings to
  NumPy style, resolved TODOs, removed dead/commented code, and cleared the remaining lint
  and fixable type issues — strictly behavior-preserving, each agent self-verifying with
  `ruff` + `ty` + import + its tests. Result: `ruff check code/hyphi` went **279 -> 0**.
- The CI lint step is now **gating** (was report-only). The type check stays non-gating
  because the only residual diagnostics are the optional `plotly`/`seaborn` import pattern
  (`try/except ModuleNotFoundError` produces a `Module | None` union that static analysis
  cannot narrow). Fixed the invalid `numpy.typing` annotations in `entropies.py`
  (`npt.number`/`npt.ndarray` do not exist).

### Decluttering

- **`_CompatUnpickler` / `load_pickle_adjacency` consolidated 3 -> 1.** The networkx-compat
  adjacency loader was duplicated in `io_brainhack.py` and both `adjacency_from_pickle.py`
  modules. One canonical implementation now lives in `hyphi.io` (refactored to pass
  complexity checks); the other modules are thin re-export shims. Verified with a new
  round-trip test (`code/tests/test_io.py`).
- **`io.py` decluttered.** Removed the unused `load_connectivity_data` and `save_network_pkl`
  (0 callers); kept `CompatUnpickler` (numpy-version pickle compat) and added the canonical
  `load_pickle_adjacency`; declared `__all__`.
- **Curated public API.** Resolved the `__init__.py` TODO and added an explicit `__all__`
  (the user-facing submodules); per-module `__all__` and `_underscore` privatization were
  applied throughout by the homogenization pass.
- **Module renames** (`N999`): `GDD_FRc_helpers.py` -> `gdd_frc_helpers.py` and
  `GDD_FRc_visualization.py` -> `gdd_frc_visualization.py` (no importers; cross-references
  updated). `hyphi.io` keeps its name (renaming would break 11 importers) with a scoped
  `A005` ignore.

### Structure, configs, docs

- **Notebooks split by polish** (`git mv`): `code/notebooks/hyphi.ipynb` -> `tutorials/`;
  the research-exploration notebooks (Kuramoto, GDD-FRC, network checks, marimo explore) ->
  `experiments/notebooks/`. `code/notebooks/` removed; `pyproject` `src`/`include` updated.
- **Config paths made portable.** The machine-specific absolute paths in
  `experiments/configs/CCORRconfig_001.toml` (`/media/noahguzman/...`,
  `/raven/u/ntorbati/...`) are now relative.
- **Ricci-Flow doc moved into the site.** `code/README_Ricci-Flow.md` -> `docs/ricci-flow.md`,
  wired into the mkdocs nav (which builds `--strict`); README link updated.
- **README placeholders replaced.** The scilaunch template italics in `code/README.md`
  ("describe this", "fill this in", stale notebook path) were replaced with real content;
  the top-level README's stale link and a markdown artifact were fixed.

### Reconciliation with Felipe's PR #37 (Declare pynndescent and lazy-import NNDescent)

- Our independent `pynndescent>=0.5,<1.0` declaration matched PR #37 exactly (both resolve to
  0.6.0), corroborating the fix. We then adopted the two ideas the PR added on top of ours:
  - **Lazy-import `NNDescent`** inside `nearest_neighbor_graph` (was a top-level import). Measured:
    `import hyphi.modeling.curvatures` dropped from 5.5 s to 3.2 s and no longer loads `numba`
    (pynndescent pulls in numba/llvmlite, ~3 s); the cost is now paid only when the approximate-kNN
    path runs. Also speeds up `import hyphi` and the CI import-sweep.
  - **`viz` optional extra** (`plotly`, `seaborn`) for the figure code currently behind
    `try/except` guards; `uv.lock` regenerated (plotly 6.8.0, seaborn 0.13.2).

### Audit-driven follow-ups (online-vs-local review)

- Added `hyphi.configs.project_path()` (joins raw relative config paths against the absolute
  detected project root) and routed all 33 `config["*_loc"]` usages across the 10 experiment
  scripts through it, so they are launch-directory-independent (the scripts previously only worked
  when run from the project root).
- Fixed the heat-kernel sign in `spectral/diffusion_distance.py` (`exp(-t * Lambda)`; the `+` sign
  overflowed) — see the SIM-SPECTRAL note above.
- Finished the notebook-move follow-through: repointed the remaining `code/notebooks/` references in
  `README.md`, `CONTRIBUTING.md` (including its runnable commands), and the
  `gdd_frc_visualization` docstring; fixed a stale `GDD_FRc_helpers` `:mod:` cross-reference; and
  corrected the `laplace.py` gap-sign docstring.
- Modernized two deprecated `plt.cm.get_cmap` calls; renamed a non-ASCII `epsilon` argument (so
  `ruff format` could not displace a fragile `# noqa`); made `make check` honestly mirror CI (lint
  gates, ty + format report-only).
- Removed a stray `experiments/scripts/canonical_calibration_figs.py` left by an earlier
  verification agent (operator decision).
- Two review/audit guides written at the repo root: `ONLINE-VS-LOCAL.md` (how local differs from
  `origin/master`) and `RESOLUTION-PLAN.md` (the improvement audit + resolution status).

### Tests

- Added `code/tests/test_io.py` (config loading, the consolidated adjacency loader,
  round-trip and weight/symmetrize/return-nodes behavior, and that the shims re-export the
  one canonical object).
- Added test coverage for the previously-untested modules via a 7-agent test-writing pass
  (known-answer tests where the math is determinable, smoke tests for plotting):
  `test_graph_simulations`, `test_kuramoto_simulations`, `test_simulations`, `test_laplace`,
  `test_diffusion_distance`, `test_spectral_visualization` (all three visualization modules),
  and `test_transform_curvature`. About +180 tests; full suite now **315 passed + 1 xfail**.
- **Two latent bugs were found by writing those tests and then fixed:**
  - `spectral/diffusion_distance.py`: `_heat_exp` passed the 1-D eigenvalue vector from
    `laplace()` straight to `scipy.linalg.expm` (which needs a 2-D matrix), so
    `diffusion_distance` raised `LinAlgError` on every call. Fixed to build the diagonal heat
    operator `U @ diag(exp(-t * Lambda)) @ U^-1` (the physical diffusion kernel decays with the
    negative sign, matching `gdd_frc_helpers`; the positive sign overflowed). The function now
    behaves as a non-negative graph metric (d(a,a)=0, symmetric), which its tests assert.
  - `visualization/curvature_visualization.py`: `visualize_graph_with_curvature` and
    `visualize_graph_partitions_colors` called `plt.colorbar(sm)` without an `ax=`, which
    raises `ValueError` on matplotlib >= 3.10. Added `ax=plt.gca()`; the functions run again.

---

## Deliberately deferred (researcher's call or out of branch scope)

These were surfaced during review but not changed, because they are research-methodology
decisions or live in modules this work did not own:

- **Statistical methodology**: the global-pooled (band-averaged) default statistic vs a
  within-dyad paired contrast; whether `phase_randomize` (which flips the DC/Nyquist Fourier
  sign and operates on wrapped Hilbert phases) is the intended null. Surfaced, not redesigned.
- **Pre-existing module issues**: entropy estimators crash on degenerate (edgeless/constant)
  inputs (tied to issue #28); `windowing.py` PLV graphs carry self-loops; `required_sample_size`
  reports the wrong total N for unbalanced (`ratio != 1`) designs.
- **Per-module test coverage** now spans `stats`, `io`, `analyses`, `benchmarks`,
  `curvatures`, `entropies`, `windowing`, `null_models`, config behavior, all three
  `simulation` modules, both `spectral` modules, all three `visualization` modules, and
  `transform_curvature`. Remaining lighter-coverage spots: `communities_centrality` and the
  GDD helper modules.
- The 24 `ty` diagnostics from the optional-import pattern (kept non-gating).
- **Lint scope**: CI gates `ruff check` on `code/hyphi` only (it is 0). `code/tests/` carries ~200
  cosmetic violations (mostly magic-value thresholds inherent to assertions), not gated; the new
  test files are `ruff check` and `ruff format` clean.

## How to review these changes

```shell
# committed review rounds + fixes
git -C HyPhi log --oneline master..HEAD

# uncommitted optimization pass
git -C HyPhi status
git -C HyPhi diff           # working-tree changes
git -C HyPhi diff --staged  # renames/moves

# verify
uv run --extra develop pytest -q
uv run --extra develop ruff check code/hyphi
uv run --extra develop ty check code/hyphi
uv run --extra docs mkdocs build --strict
```
