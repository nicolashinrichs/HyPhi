# HyPhi: how the local working tree differs from `origin/master`

**Scope.** Branch `fix/coherence-pass`. `origin/master` is `ae159c2` (the clone base `1030489` plus one trivial Simon commit fixing the README experiments-section formatting). The local tree is a **large superset** of online: clone base + **18 committed review/fix commits** (`git log master..HEAD`) + a **large uncommitted optimization/homogenization pass** + **9 new untracked test files** + an uncommitted `CHANGES.md`.

**Headline diff (verified):** `git diff origin/master --stat` = **54 files changed, +2977 / -1079**. Nothing in the optimization pass is committed or pushed.

**The one online commit is already matched locally.** Simon's README experiments-section formatting fix exists on `origin/master`; the local tree already contains an equivalent state independently, so it produces **no merge conflict** and does not appear as a divergence to resolve.

**One stray artifact.** `experiments/scripts/canonical_calibration_figs.py` (260 lines, ~10KB) is **untracked** (`git ls-files` = 0), referenced nowhere in any `.py`/`.md`/`.toml`/`.yml`, absent from `CHANGES.md`, and depends on `jitcdde`/`symengine` (declared) but defaults `--outdir` to a personal path. It was left by an earlier verification agent and is the single item flagged across every domain. Treat it as a keep/remove decision, not part of the documented pass.

---

## Differences by theme

### 1. STATS (committed: `c997c92`/`3759de9`/`d56cf94`/`b234317`/`6eeb5ef`/`cf01b04`) — highest-consequence change
`stats.py` +172, `test_stats.py` +172, `test_hyphi.py` +171. The online hierarchical permutation test built `trial_id = dyad__trial`, which collapsed the within-dyad permutation null (the per-condition trial index restarts at 0, so both conditions merged into one block) and produced a **measured ~100% false-positive rate**. Local fixes the id to `dyad__condition__trial`, validates the one-condition-per-block invariant and **raises** instead of degenerating, fixes a label-vs-position order-dependence bug (`reset_index(drop=True)` + sort trial_ids by `str`), hardens input validation (refuses missing/inf/NaN/empty/`n_perms<1`), refuses between-subjects/degenerate designs that silently returned `p=1.0`, and **warns** when the permutation space is structurally too small for `p<0.05`. All committed (not in the uncommitted pass).

### 2. CONFIGS (committed: `d80850b`/`88291c2`/`b234317`/`d56cf94`/`6e4d219` + uncommitted polish)
`configs.py` +334 (largest single-file diff). `init()` is now **cwd-neutral** (the old implicit `os.chdir(PROJECT_ROOT)` could chdir into a same-named sibling checkout); the chdir moved into an explicit opt-in `bootstrap()`. Adds symlink-safe component-wise project containment in `_set_wd` (replacing `str.startswith`, which false-matched sibling-prefix dirs and false-mismatched `/tmp`->`/private/tmp`), order-independent foreign-config validation against a freshly-merged dict before touching globals, and table-key type validation. `ty` no longer excludes `configs.py`. New `docs/api/configs.md` wired into mkdocs.

### 3. IO-DEDUP (uncommitted)
Consolidated the `load_pickle_adjacency` networkx-compat loader from **3 copies to 1** canonical implementation in `hyphi.io`. The two `adjacency_from_pickle.py` modules and `io_brainhack.py` become thin re-export shims (the large negative deltas in the stat are these shim reductions, not lost functionality). Removed unused `load_connectivity_data` / `save_network_pkl` (0 callers); declared `__all__`. New `test_io.py` asserts round-trip + shim object-identity.

### 4. MODELING (committed: `9818412`/`1b39fb7` + uncommitted homogenization)
Two committed correctness fixes: wired the ricci-flow `--enable-visualization` flag (was a no-op referencing undefined vars), and fixed a wrong `compute_frc_vec` keyword (`method_val`, a guaranteed `TypeError`) plus a `sim_graph` self-loop guard. Uncommitted homogenization converted docstrings to NumPy style, removed dead code/TODOs, added per-module `__all__`, and fixed invalid `numpy.typing` annotations in `entropies.py` (`npt.number`/`npt.ndarray` do not exist). `GDD_FRc_helpers.py` renamed to `gdd_frc_helpers.py` (N999, no importers).

### 5. SIM-SPECTRAL (uncommitted) — two latent bugs surfaced
`diffusion_distance.py`: `_heat_exp` previously passed the 1-D eigenvalue vector straight to `scipy.linalg.expm` (needs 2-D), so the function **raised `LinAlgError` on every call**; now builds `U @ diag(exp(t*Lambda)) @ U^-1`. Plus NumPy-style docstring/lint homogenization across simulation and spectral modules, and new known-answer/smoke tests. The fix re-activated a dead function and thereby **exposed a pre-existing sign bug** (see open items): the heat operator uses `exp(+t*L)` where the physical kernel is `exp(-t*L)`.

### 6. VISUALIZATION (committed: `0b6b0d1` + uncommitted) — one latent bug fixed
Committed: curvature-blind partition plot now reads either Ricci variant (Forman-flow graphs previously rendered gray edges); ORC/FRC attribute-swap correction. Uncommitted: `plt.colorbar(sm)` calls lacked `ax=`, raising `ValueError` on matplotlib >=3.10; added `ax=plt.gca()`. `GDD_FRc_visualization.py` renamed to `gdd_frc_visualization.py`. New smoke tests for all three viz modules.

### 7. LINT-DOCSTRINGS (committed baseline + uncommitted 23-agent homogenization)
Declared the scientific naming convention (`ignore-names` for `G`, `A`, `L`, `FRC`/`ORC`, ...) and turned on NumPy docstring enforcement in `pyproject`. A per-file pass converted every module's docstrings, resolved TODOs (~17 -> 0 in package code), and cleared lint. **Verified locally: `ruff check code/hyphi` = "All checks passed!" (0).** This theme is the bulk of the line-count churn.

### 8. STRUCTURE-MOVES (uncommitted `git mv`)
Curated the public API (`__init__.py` `__all__`). Module renames for N999 (`GDD_FRc_*` -> `gdd_frc_*`; `hyphi.io` kept its name to avoid breaking 11 importers, with a scoped `A005` ignore). Notebooks split: `hyphi.ipynb` -> `tutorials/`, research notebooks -> `experiments/notebooks/`; `code/notebooks/` removed. `main.py` intentionally NOT restored (it was deleted long before this work). `README_Ricci-Flow.md` -> `docs/ricci-flow.md`.

### 9. CI-PACKAGING (committed `322b63f`/`1bedc51`/`49c042a`/etc. + uncommitted)
New GitHub Actions CI (install, import-sweep, test suite, gating `ruff check code/hyphi`, **non-gating** `ty check`; Python 3.11/3.12/3.13 matrix; `uv sync --frozen`; triggers on `master`). Declared previously-missing deps (`pynndescent` made two modules unimportable; `tqdm-joblib`), real `docs` extra, removed dead `[tool.mypy]`, synced `CITATION.cff` 1.2.0 -> 2.2.0, regenerated `uv.lock` (+331). `.github/workflows/ci.yml` and `docs/api/configs.md` are net-new vs online.

### 10. TESTS
Committed rounds converted the script-style `test_hyphi.py` (which ran a full-size simulation at import and **hung pytest collection 6+ min**) to proper pytest and added stats/curvature regression tests. The uncommitted pass added ~180 tests across previously-untested modules; **9 of these test files are untracked**. Reported suite: 104 collected (one hung) -> 313 passed + 1 xfail.

### 11. TUTORIAL-README (committed `290df62`/`0b6b0d1`/`cf01b04` + uncommitted)
`tutorials/02_dual_eeg_pipeline.md` +157 (largest doc diff): made runnable (fixed the wrong dict shape into `entropy_to_long_df`) and statistically honest (within-dyad design + power caveat; curvature-swap table corrected). README placeholders replaced with real content.

### 12. EXPERIMENT-CONFIGS (uncommitted + small committed script fixes)
`CCORRconfig_001.toml`: replaced machine-specific absolute paths (`/media/noahguzman/...`, `/raven/u/ntorbati/...`) with portable project-relative `./data` / `./results`. The two experiment scripts received the committed `compute_frc_vec` keyword fix.

---

**Net:** local is online + 18 reviewed commits + a coherent uncommitted optimization/test pass. The only true "new vs online" files are `.github/workflows/ci.yml`, `docs/api/configs.md`, `docs/ricci-flow.md`, the 9 test files, `CHANGES.md` — and the stray `canonical_calibration_figs.py`.
