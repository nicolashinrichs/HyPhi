# hyphi – **code**

    Last update:    June 4, 2026
    Status:         work in progress

***

## Description

`hyphi` analyses hyperscanning (dual-brain EEG) data through the geometry of
inter-brain connectivity graphs. It builds sliding-window phase-locking (PLV)
graphs, computes their Forman-Ricci (and Ollivier-Ricci) curvature, summarises
the curvature distribution as a geometric entropy, and tests condition
differences with hierarchical permutation and mixed-effects models. The package
also provides null models, Kuramoto and small-world simulations, and benchmark
comparisons against standard hyperscanning metrics (PLV, wPLI).

## Codebase

The analysis lives in the `hyphi` package; runnable end-to-end examples live in
`tutorials/` (start with `tutorials/02_dual_eeg_pipeline.md`) and `experiments/`.

### `hyphi` Python package

Python code (in the structure of a python package) is stored in `./code/hyphi/`

To install the `hyphi` package, run the following code in the project root directory:

```shell
uv sync [--extra develop] [--extra notebook]
```

Or use other package management tools (e.g., `conda`, `pip`, or `pixi`) to install the package in editable mode.


### Notebooks and examples

Polished, instructional material is in `./tutorials/` (the quickstart and the
dual-EEG pipeline walkthrough). Raw research-exploration notebooks (Kuramoto,
GDD-FRC, network checks) are in `./experiments/notebooks/`, alongside the
experiment scripts in `./experiments/scripts/`.

### Configs

Paths to data, parameter settings, etc. are stored in the config file: `./configs/config.toml`

Private config files (starting with `_`) that contain, e.g., passwords, and therefore should not be shared,
or mirrored to a remote repository can be listed in: `./configs/_private_config.toml`

Both public and private config files will be read out by the script in `./code/hyphi/configs.py`.
Keep config toml files and the script in the places where they are.

To use your configs in your `Python` scripts, do the following:

```python
from hyphi.configs import config

# initialize the config (will be created if it does not exist)
config.init()

# check out which paths are set in config.toml
config.paths.show()

# get the path to data
path_to_data: str = config.paths.DATA

# Get parameter from config (example)
weight_decay = config.params.weight_decay

# Get private parameter from config (example)
api_key = config.service_x.api_key
```

Point the paths in `configs/config.toml` (and any private `_config.toml`) at
your own data before running an analysis.

For other programming languages, corresponding scripts must be implemented to use these `*config.toml` files in a similar way.

## COPYRIGHT/LICENSE

See the [LICENSE](../LICENSE) file for details.
