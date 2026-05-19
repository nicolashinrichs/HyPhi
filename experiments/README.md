# Experiments based on `HyPhi`

End-to-end worked examples illustrating the canonical HyPhi workflow.

## Layout

- `configs/` — TOML and JSON configurations consumed by the scripts below.
- `scripts/` — runnable analysis and plotting entry points.

## Scripts

- `kuramoto_frc.py` — run Kuramoto simulations and compute Forman-Ricci curvatures on the resulting PLV graph series.
- `hyper_ccorr_frc.py` — compute per-window Forman-Ricci curvatures on CCORR inter-brain graphs and write per-trial entropies.
- `hyper_ccorr_aug_frc.py` — same workflow with Augmented Forman-Ricci curvature.
- `hyper_ccorr_orc.py` — sweep Ollivier-Ricci parameters over CCORR inter-brain matrices.
- `hyper_ccor_ragg_frc.py` — aggregated FRC permutation testing on CCORR matrices via energy distance and dcor statistics.
- `hierarchical_stats_runner.py` — hierarchical proof-of-concept stats runner driven by a TOML config under `configs/`.
- `neureps_simulations.py` — reproduce the weighted Watts-Strogatz curvature-entropy sweep used in the NeurReps 2025 submission.
- `demonstrate_ccorr_adjusted.py` — minimal demonstration of adjusted CCORR computation.
- `plot_*.py` — companion plotting scripts that consume the outputs of the analysis scripts above.
