# Ricci-Flow Analysis

This folder contains a standalone workflow script for Kuramoto-based PLV graph analysis with Forman curvature and Ricci flow:

- `ricci_flow_analysis.py`

The script is adapted to run from either:
- project root (`HyPhi_base_refactor/`), or
- this folder (`HyPhi_base_refactor/Ricci-Flow/`).

## What The Script Does

For each time window:
1. Loads per-run Kuramoto phases (`*_kuramoto_phases.npy`) if available.
2. Simulates missing phase files using `software_module/KuramotoSimulations.py`.
3. Builds PLV graphs using one of two modes:
   - `merge_signals_plv` (default): concatenate signals across runs first, then compute PLV, then build one merged graph.
   - `merge_graphs`: load per-run precomputed graph windows and disjoint-union them.
4. Computes Forman curvature (`compute_frc` from `src/hyphi/curvatures.py`).
5. Runs iterative Forman-Ricci-flow updates.
6. Saves per-window numeric outputs and a final `summary.json`.

## Inputs

### Required (directly or via config)
- `--config` (default: `experiments/analysis/CCORRconfig_001.toml`)
- Run IDs (`--runs` or `num_kuramotos` in config)
- Phase location (`--phase-dir` or `kuramoto_loc` in config)
- Target windows (`--target-windows` or `kuramoto_time` in config)

### Data files
- Per-run phases (expected pattern by default):
  - `data/<run>_kuramoto_phases.npy`
- Optional precomputed per-run graph windows (for `merge_graphs` mode):
  - `data/<run>_connectome_kuramoto.pkl`
- Connectome file (optional, only if `--attach-connectome-weights`):
  - `software_module/connectivity_data.pkl`

## Outputs

Written under `--output-dir` (default: `results/ricci_flow_analysis`):

- `window_XX_forman_values.npy`
- `summary.json`

Optional outputs (if `--save-graphs`):
- `window_XX_aggregated_graph.pkl`
- `window_XX_forman_graph.pkl`
- `window_XX_ricci_flow_graph.pkl`

Optional visualization (if `--enable-visualization`):
- PNGs via `curvature_visualisation.py`

## Core Functions (quick map)

- `load_config_file`: load TOML config.
- `load_connectome_matrix`: load structural connectivity matrix.
- `compute_plv_matrix_window`: wrapper to repo PLV function.
- `compute_frc_graph`: wrapper to repo Forman curvature function.
- `simulate_missing_phase_files`: generate missing phase trajectories with existing Kuramoto simulation code.
- `load_phase_windows` / `phase_windows_from_array`: normalize phase arrays into per-window matrices.
- `build_merged_plv_graph`: build merged PLV graph per window.
- `forman_ricci_flow`: iterative Forman-based Ricci-flow update loop.
- `main`: orchestrates load/simulate -> graph build -> curvature -> flow -> save.

## Performance Notes

The pipeline can be expensive for dense merged graphs.

Main speed controls:
- `--flow-iterations` (lower is faster)
- `--plv-threshold` (sparsifies PLV graph)
- `--max-windows` (debug with fewer windows)
- keep visualization disabled (default)
- keep graph pickle saving disabled (default)

## Example Commands

### 1) Fast smoke test (1 window, 1 run)
```bash
python Ricci-Flow/ricci_flow_analysis.py \
  --runs 1 \
  --phase-dir data_8node \
  --data-dir data_8node \
  --target-windows 1 \
  --flow-iterations 1
```

### 2) Full merged-signal analysis (24 windows)
```bash
python Ricci-Flow/ricci_flow_analysis.py \
  --graph-construction merge_signals_plv \
  --runs 1 2 3 4 5 6 7 8 \
  --phase-dir data_8node \
  --data-dir data_8node \
  --target-windows 24 \
  --flow-iterations 10 \
  --plv-threshold 0.2
```

### 3) Force re-simulation of phases
```bash
python Ricci-Flow/ricci_flow_analysis.py \
  --runs 1 2 3 4 5 6 7 8 \
  --phase-dir data_8node \
  --data-dir data_8node \
  --force-simulate-phases \
  --sim-n-osc 8 \
  --target-windows 24
```

## Dependencies

At minimum:
- `numpy`, `networkx`
- `GraphRicciCurvature`
- `jax`, `jaxlib` (if phase simulation is needed)
- TOML parser (`tomllib` on Python 3.11+, otherwise `tomli`)

If visualization is enabled, ensure plotting dependencies used by `curvature_visualisation.py` are installed.
