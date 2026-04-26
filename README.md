# hyphi – **code**

    Last update:    April 14, 2026
    Status:         work in progress

***
A Python package for hyperscanning data analysis by tracking inter-brain network curvature and its entropy distribution.

## Overview

`HyPhi` implements a geometry-driven alternative to traditional synchrony-based hyperscanning analysis.

The pipeline includes:

- A ground-truth simulation framework based on coupled Kuramoto oscillators
- Empirical dual-EEG analysis
- Comparison between Forman-Ricci and Augmented Forman-Ricci curvature metrics
- Sliding-window dynamic network construction
- Phase transition detection using curvature distributions, entropy, and quantiles

## Conceptual Workflow

Across simulated and empirical use cases, `HyPhi` follows the same high-level workflow:

1. **Network construction**
   Static or time-resolved graphs are constructed from simulations or empirical connectivity measures.

2. **Discrete curvature computation**
   Edge-wise Ricci curvature (Forman-Ricci and variants) is computed on each network.

3. **Distributional analysis**
   Curvature values are treated as distributions and analyzed via kernel density estimation.

4. **Information-theoretic tracking**
   Entropy and quantiles of curvature distributions are used to detect regime shifts and phase transitions.

Concrete implementations of this workflow are provided in the `experiments` and `tutorials` directories.

## Scientific Motivation

Traditional synchrony metrics collapse rich network structure into low-dimensional summaries and often miss critical topological transitions.

HyPhi instead treats inter-brain coupling as a **dynamic geometric object**, where curvature captures higher-order structural reorganization.
This enables principled detection of coupling and decoupling regimes beyond synchrony alone.

## Project structure

The repository is split into the following main directories, each with a dedicated `README.md`:

### `code`

Source folder of the Python toolbox `hyphi`, which implements the core analysis modules and pipelines for the pipeline.

- Network simulations
- Ricci curvature computation
- Ricci-Flow (for details, see [README_Ricci-Flow.md](code/README_Ricci-Flow.md)).
- Density estimation
- Entropy and quantile analysis

### `data`

## Description

*List relevant information one needs to know about the code of this research project.
For instance, one could describe the computational model that was applied,
and which statistical approach has been chosen for.*

## Codebase

*Refer to the corresponding code/scripts written for the analysis/simulation/etc.*

### `hyphi` Python package

Python code (in the structure of a python package) is stored in `./code/hyphi/`

To install the `hyphi` package, run the following code in the project root directory:

```shell
uv sync [--extra develop] [--extra notebook]
```

Or use other package management tools (e.g., `conda`, `pip`, or `pixi`) to install the package in editable mode.


### Notebooks

`Jupyter` | `marimo` notebooks are stored in `./code/notebooks/`

### Configs

Paths to data, parameter settings, etc. are stored in the config file: `./code/configs/config.toml`

Private config files that contain, e.g., passwords, and therefore should not be shared,
or mirrored to a remote repository can be listed in: `./code/configs/private_config.toml`

Both files will be read out by the script in `./code/hyphi/configs.py`.
Keep both config toml files and the script in the places where they are.

To use your configs in your `Python` scripts, do the following:

```python
from hyphi.configs import config, paths

# check out which paths are set in config.toml
paths.show()

# get the path to data
path_to_data: str = paths.DATA

# Get parameter from config (example)
weight_decay = config.params.weight_decay

# Get private parameter from config (example)
api_key = config.service_x.api_key
```

*Fill the corresponding `*config.toml` files with your data.*

For other programming languages, corresponding scripts must be implemented to use these `*config.toml` files in a similar way.

## COPYRIGHT/LICENSE

See the [LICENSE](../LICENSE) file for details.
