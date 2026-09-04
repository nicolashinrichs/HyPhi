> **STATUS (updated after acting on this plan).** Group A is fully resolved: the notebook-move
> dangling references (README, CONTRIBUTING runnable commands, gdd_frc_visualization docstring),
> the stale `:mod:` cross-ref, the `laplace.py` gap-sign docstring, and the deprecated
> `plt.cm.get_cmap` calls are all fixed; `ruff format` cleaned the source + test files (and a
> `ruff format`-displaced `# noqa` was resolved by renaming `epsilon` to ASCII, keeping ruff
> gating at 0). B2 (the `diffusion_distance` heat-kernel sign `exp(-t*L)`) and B3 (`make check`
> now honestly mirrors CI: lint gates, ty/format report-only) are done. C1 added a
> `CompatUnpickler` round-trip test. Verified: **314 passed + 1 xfail**, `ruff check code/hyphi`
> = 0 (gating), `make check` exits 0, strict docs build passes.
>
> **B1 RESOLVED** (operator chose remove): the stray `canonical_calibration_figs.py` was deleted.
> **B4 RESOLVED** (operator chose robustify): added `hyphi.configs.project_path()` (joins raw
> relative config paths against the absolute project root) with a test, and routed all 33
> `config["*_loc"]` usages across the 10 experiment scripts through it, so they are now
> launch-directory-independent (verified: all compile, 0 raw `Path(config["*_loc"])` remain).
>
> **STILL OPEN (genuinely your long-term call, not blockers):** the deferred research-methodology
> items only — the band-pooling default statistic and the null-model domain (see "deliberately
> deferred" in CHANGES.md). Everything mechanical/structural in this audit is resolved; the branch
> is mergeable. Final state: **315 passed + 1 xfail**, `ruff check code/hyphi` = 0 (gating),
> `make check` exits 0, strict docs build passes.

# HyPhi audit summary: are all local changes improvements?

**Short answer: almost.** The overwhelming majority of the diff is a genuine improvement or behavior-preserving cleanup. The pass fixes **four real latent bugs** that were total breakages online (a 100%-false-positive permutation null, a `diffusion_distance` that raised on every call, a matplotlib-3.10 colorbar `ValueError`, and invalid `numpy.typing` annotations) and adds substantial test coverage. **One REGRESSION** and a handful of LOW/MEDIUM RISKs remain, all introduced by incompleteness (dangling paths the same pass should have updated) or surfaced (a pre-existing sign bug newly reachable). None block a careful, scoped merge.

## Per-domain verdict table

| Domain | Verdict | Basis |
|--------|---------|-------|
| STATS | **IMPROVEMENT** | Repairs the measured ~100% false-positive null (`trial_id` -> `dyad__condition__trial`), hardens validation, 35/35 tests pass. One LOW RISK (warning `stacklevel` mis-attribution). |
| CONFIGS | **IMPROVEMENT** | `init()` cwd-neutral, symlink-safe containment, order-independent foreign-config validation. One MEDIUM RISK (experiment scripts still rely on the removed chdir via raw relative path keys). |
| IO-DEDUP | **IMPROVEMENT** | 3->1 canonical loader, behavior-preserving (round-trip + object-identity tests). One LOW RISK (`CompatUnpickler` now 0-caller, 0-coverage public surface). |
| MODELING | **IMPROVEMENT** | Two committed correctness fixes (viz flag, self-loop guard) + invalid-annotation fix + behavior-preserving homogenization. Issue-#28 degenerate-input crashes deliberately deferred. |
| SIM-SPECTRAL | **IMPROVEMENT (with a surfaced bug)** | Fixes the total `diffusion_distance` breakage; the fix exposes a pre-existing `exp(+tL)` sign bug -> overflow on realistic graphs (MEDIUM RISK). |
| VISUALIZATION | **IMPROVEMENT** | Colorbar `ax=` fix + curvature-blind-plot fix, test-covered. One MEDIUM RISK (two `plt.cm.get_cmap` deprecations left unfixed in the same edited function). |
| LINT-DOCSTRINGS | **IMPROVEMENT** | `ruff check code/hyphi` = 0 (verified). One LOW RISK (`laplace.py` "Modulus" docstring is sign-inaccurate). |
| STRUCTURE-MOVES | **IMPROVEMENT, but incomplete** | Clean renames/moves + curated API; the moves left dangling doc/path references (the REGRESSION + RISKs below). |
| CI-PACKAGING | **IMPROVEMENT** | Real CI, declared missing deps, version sync. One MEDIUM RISK (`make check` is red out-of-the-box and overclaims "matches CI"). |
| TESTS | **IMPROVEMENT** | 313 passed + 1 strict xfail; new coverage surfaced two latent bugs. One LOW RISK (test files not lint-gated; 9 still untracked). |
| TUTORIAL-README | **IMPROVEMENT** | Tutorial made runnable + honest; verified by execution. Stale README/CONTRIBUTING paths are the REGRESSION/RISK below. |
| EXPERIMENT-CONFIGS | **IMPROVEMENT** | Portable paths; committed `compute_frc_vec` keyword fix. |
| ARTIFACTS (stray script) | **NEUTRAL (unresolved)** | Untracked, unreferenced, undocumented; keep/remove is a human decision. |

## Confirmed open items, ranked by severity

**No HIGH-severity open items.** Count: **0 high**, 4 medium, 9 low (de-duplicated below; the stray-script and the dangling-path families recur across several domain reports but are one underlying item each).

### MEDIUM (4)
1. **diffusion_distance heat-kernel sign / overflow** — `code/hyphi/spectral/diffusion_distance.py:36`. Uses `exp(+t*eigvals)`; the repo's own sibling `curvatures.py:578-579` uses `exp(-evals*ti)` (verified). With `L` PSD and EDP's hardcoded `time_limit=100` (verified at line 127), `EDP` overflows on `K_4` (`lambda_max=4`) and **hard-crashes** (`ValueError: array must not contain infs or NaNs`) on a realistic Watts-Strogatz `n=20` graph. Latent, not live: grep finds **no caller** of `EDP`/`diffusion_distance` outside the module + its test. Tests use `time_limit=1.0` (or `P_3`), so they give false confidence.
2. **`make check` red out-of-the-box + overclaims "matches CI"** — `Makefile` / `.github/workflows/ci.yml`. Verified: gated `ruff check code/hyphi` is clean, but `ruff format --check` is red (10 files, one inside the gated tree: `graph_simulations.py`). `make check` also makes `ty` a hard prerequisite while CI runs it `continue-on-error: true`, so `make check` aborts at typecheck before reaching format. Self-inconsistent contract; a contributor hits failure day one.
3. **Experiment scripts depend on the removed chdir** — `experiments/scripts/*.py`. ~14 scripts still call `init()` (none migrated to `bootstrap()`) and bind relative `*_loc` keys (e.g. `./data`, `./results`) to CWD. Byte-identical and correct **only** when launched from the project root (the documented workflow); from a subdirectory they silently bind I/O to the wrong directory with no error. Not a regression in the intended workflow; a footgun the cwd-neutral change exposes.
4. **Notebook-move dangling references** — **REGRESSION**. `CONTRIBUTING.md` (lines 286/290/291/298/304) still points at `code/notebooks/`, including two **runnable** commands (`jupyter lab code/notebooks/hyphi.ipynb`, `marimo edit code/notebooks/hyphi_explore.py`) that now fail; `README.md:96` and `code/hyphi/visualization/gdd_frc_visualization.py:6` carry the same stale path (all verified present). Worse than online because online's paths were valid; the move silently invalidated them while the same pass edited these files.

### LOW (9)
5. **Stray script** `experiments/scripts/canonical_calibration_figs.py` — untracked, unreferenced, undocumented, 66 ruff violations, personal default `--outdir`. Keep/remove decision.
6. **Stale `:mod:` cross-ref** `diffusion_distance.py:5` -> `hyphi.modeling.GDD_FRc_helpers` (renamed to `gdd_frc_helpers`); verified present. One-word doc fix; sibling files were updated, this one missed.
7. **`laplace.py` "Modulus" docstring** — the code returns `eigenvalues[0]-eigenvalues[1]` (non-positive, no `abs`); a passing test asserts `-5`. Fix the docstring/comment, **not** the code.
8. **`CompatUnpickler` 0-caller, 0-coverage public symbol** — kept deliberately; add a ~6-line round-trip test or consciously accept.
9. **STATS warning `stacklevel` mis-attribution** — points at `stats.py` instead of user code via the `energy_distance_hierarchical` path; one-line robustness fix.
10. **`curvature_visualization.py` deprecated `plt.cm.get_cmap`** — lines 432 and 623 (line 623 untested); drop-in `plt.get_cmap` to survive matplotlib 3.11.
11. **Test files not lint-gated** — `code/tests` has 201 cosmetic ruff violations; CI gates only `code/hyphi`. Document the asymmetry or finish.
12. **9 test files still untracked** — must be staged explicitly (path-scoped `git add`).
13. **CHANGES.md omits the stray script** — resolves itself if the script is removed; otherwise name it.

---

# HyPhi resolution plan (branch `fix/coherence-pass`)

Read-only audit: nothing below was applied. All paths absolute. Group A = mechanical, safe, do before merge. Group B = needs a human (Nico/steward) decision. Group C = defer or accept.

---

## A. Quick-fixes (mechanical, low-risk, finish before merge)

**A1. Fix the notebook-move dangling references (the REGRESSION).** Doc/comment-only, no behavior risk.
- `/Users/nicolashinrichs/github/HyPhi/README.md:96` — `code/notebooks` -> `experiments/notebooks`.
- `/Users/nicolashinrichs/github/HyPhi/CONTRIBUTING.md` lines 286, 290, 291, **298, 304** — repoint to `tutorials/hyphi.ipynb` (Jupyter) and `experiments/notebooks/hyphi_explore.py` (marimo). Lines 298/304 are runnable commands that currently fail verbatim; highest priority in this group.
- `/Users/nicolashinrichs/github/HyPhi/code/hyphi/visualization/gdd_frc_visualization.py:6` — `code/notebooks/` -> `experiments/notebooks/`.

**A2. Fix the stale `:mod:` cross-reference.** `/Users/nicolashinrichs/github/HyPhi/code/hyphi/spectral/diffusion_distance.py:5` — `hyphi.modeling.GDD_FRc_helpers` -> `hyphi.modeling.gdd_frc_helpers`. Makes CHANGES.md's "cross-references updated" claim true.

**A3. Fix the `laplace.py` docstring sign inaccuracy.** `/Users/nicolashinrichs/github/HyPhi/code/hyphi/spectral/laplace.py:61` — replace "Modulus of the gap between the two smallest eigenvalues" with wording reflecting the actual non-positive value (e.g. "Negated gap `eigenvalues[0] - eigenvalues[1]` (<= 0)"); also fix the stale inline comment at line ~65 ("gap between smallest and largest"). **Do NOT add `abs()`** — `code/tests/test_laplace.py` asserts the gap equals `-5`.

**A4. Modernize the deprecated colormap calls.** `/Users/nicolashinrichs/github/HyPhi/code/hyphi/visualization/curvature_visualization.py` lines 432 and 623 — `plt.cm.get_cmap("tab20", n)` -> `plt.get_cmap("tab20", n)` (verified zero-risk drop-in, identical `ListedColormap`). Survives matplotlib 3.11; matches the sibling `visualize_graph_partitions_markers`.

**A5. Run `ruff format` on the dirty files (or the contract is red).** `cd /Users/nicolashinrichs/github/HyPhi && uv run --extra develop ruff format code/hyphi code/tests` — fixes the in-gated-tree `graph_simulations.py` plus the new test files so `make check`'s format axis passes. (Leave the stray script alone; see B1.)

---

## B. Human decisions (Nico / steward sign-off required)

**B1. The stray script `experiments/scripts/canonical_calibration_figs.py` — keep or remove.** Verified untracked, unreferenced, undocumented, 66 ruff violations, personal default `--outdir`. **Recommendation: remove** (it is regenerable agent scaffolding; its calibration logic overlaps a tracked notebook). If kept, it must first be `ruff format`+lint-cleaned, given a repo-relative `--outdir`, wired into `experiments` + named in `CHANGES.md`. **Either way: stage the commit path-scoped, never `git add -A`** — it is not gitignored and would otherwise ride along with the 9 legitimate test files. Per the never-delete contract, surface for Nico rather than `rm`.

**B2. The `diffusion_distance` heat-kernel sign flip (changes a research metric's numeric output).** In `/Users/nicolashinrichs/github/HyPhi/code/hyphi/spectral/diffusion_distance.py:36`, change `np.exp(t * eigvals)` -> `np.exp(-t * eigvals)`. The direction is not in doubt: the repo's own `code/hyphi/modeling/curvatures.py:578-579` uses the negative sign (verified). This fixes both the overflow and the physics. Also recommend un-hardcoding `time_limit=100` at line 127 into a parameter, and adding a realistic-scale test (`EDP` on a `gen_weighted_sw(n=20,...)` graph asserting `np.isfinite(chi)`) since current tests use `time_limit=1.0`/`P_3` and never reach the overflow regime. Latent (no live caller), so not a blocker — but it ships a function that currently crashes on realistic input. Nico signs off because it alters numeric output.

**B3. `make check` vs CI parity.** Pick one consistent pair: either (a) make `ty` non-gating in the `Makefile` to match CI's `continue-on-error: true` AND keep A5's format fix, then soften the "matches CI" docstring (format remains local-only); or (b) add a gating `ruff format --check code/hyphi` to `.github/workflows/ci.yml` so the two genuinely converge. Minimum: drop the false "matches CI" claim. (Files: `/Users/nicolashinrichs/github/HyPhi/Makefile`, `/Users/nicolashinrichs/github/HyPhi/.github/workflows/ci.yml`.)

**B4. Experiment-script CWD footgun.** Minimum (before ship): document the project-root-CWD requirement in `code/README.md` and at the top of each experiment script, and tighten the misleading `CCORRconfig_001.toml` comment. Robust follow-up: resolve the `*_loc` keys against `config.PROJECT_ROOT` at point of use (e.g. `Path(hyphi_config.paths.PROJECT_ROOT, config["data_loc"])`) so scripts are launch-dir-independent. Do **not** migrate scripts back to `bootstrap()` — that would restore the chdir the refactor deliberately removed.

---

## C. Deferred / optional (acceptable to accept-as-is with a note)

- **C1. `CompatUnpickler` 0-caller / 0-coverage** (`code/hyphi/io.py`). Add a ~6-line round-trip test (pickle a `numpy._core`-prefixed array, assert recovery) to promote the contract to verified, or consciously accept the unexercised public shim.
- **C2. STATS warning `stacklevel`** (`code/hyphi/stats.py`). One-line robustness fix: emit the underpowered-design warning from the public entry points with a per-caller `stacklevel`, or use `warnings.warn(..., skip_file_prefixes=...)` (Python 3.12+). Add a test asserting the warning's `filename`/`lineno` via the `energy_distance_hierarchical` path.
- **C3. Test-file lint asymmetry.** Add one line to `CHANGES.md`'s deferred section stating CI gates only `code/hyphi` and `code/tests` has 201 cosmetic violations — or widen CI + finish the cleanup. Cheapest: document.
- **C4. `CHANGES.md` completeness.** Resolves itself if B1 removes the stray script; if kept, name it there.
- **C5. Issue-#28 degenerate-input crashes** — already deliberately deferred (strict `xfail` in `test_hyphi.py`); leave as-is.

---

## Sequencing
1. Do all of **A** (mechanical, no decisions). After A5, `make check`'s format axis is clean.
2. Get Nico's calls on **B1** (stray script) and **B2** (sign flip) — the two that touch what ships and what numbers come out.
3. Decide **B3/B4** (process/footgun) — can land in this pass or a fast follow-up.
4. Stage path-scoped (the 9 test files + `CHANGES.md` + the edited sources), exclude the stray script, commit on `fix/coherence-pass`, open a **draft** PR. **Do not merge** — steward merges. Identity from `.env`; no AI attribution.
