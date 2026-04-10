# hyphi – **code**

    Last update:    April 10, 2026
    Status:         work in progress

***

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

#### Notebooks

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
