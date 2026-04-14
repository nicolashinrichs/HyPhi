"""
Configurations for the hyphi project.

Note:
----
    * store private configs in the same folder as 'config.toml', namely: "./[PRIVATE_PREFIX]_configs.toml"
    * keep the prefix such that it is ignored by git

Alternatives:
-----------
    * load *configs.toml's with the toml module (see below) and process it further with the package python-box
    * configs could also be set using an .env file together with the python-dotenv package.

Author: Simon M. Hofmann
GitHub: SHEscher
Years: 2023

"""

# %% Imports
from __future__ import annotations

import os
import sys

if sys.version_info >= (3, 11):
    # since python 3.11, there is also tomllib
    import tomllib as toml
else:
    import toml

import logging.config
from pathlib import Path
from typing import Any

# %% Config class & functions ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _iter_nested_dicts(nested_dict: dict[str, Any]) -> Any:
    """Create a generator iterating over values in nested dicts."""
    for value in nested_dict.values():
        if isinstance(value, dict):
            yield from _iter_nested_dicts(value)
        else:
            yield value


def _create_parent_dirs(config_as_dict: dict[str, Any]) -> None:
    """Create parent dirs of log files."""
    for value in _iter_nested_dicts(config_as_dict):
        if isinstance(value, str) and value.endswith(".log"):
            Path(PROJECT_ROOT, value).parent.mkdir(parents=True, exist_ok=True)


class _CONFIG:
    """Configuration object."""

    def __init__(self, config_dict: dict | None = None):
        """Initialise _CONFIG class object."""
        if config_dict is not None:
            self.update(config_dict)

    def __repr__(self):
        """Implement __repr__ of _CONFIG."""
        str_out = "_CONFIG("
        list_attr = [k for k in self.__dict__ if not k.startswith("_")]
        ctn = 0  # counter for visible attributes only
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                # ignore hidden attributes
                continue
            ctn += 1
            str_out += f"{key}="
            if isinstance(val, _CONFIG):
                str_out += val.__str__()
            else:
                str_out += f"'{val}'" if isinstance(val, str) else f"{val}"

            str_out += ", " if ctn < len(list_attr) else ""
        return str_out + ")"

    def update(self, new_configs: dict[str, Any]):
        """Update the config object with new entries."""
        for k, val in new_configs.items():
            if isinstance(val, (list, tuple)):
                setattr(self, k, [_CONFIG(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, k, _CONFIG(val) if isinstance(val, dict) else val)

    def show(self, indent: int = 0):
        """
        Display the nested configuration information.

        :param indent: The number of tabs to use for indentation (default: 0)
        :type indent: int
        :return: None
        """
        for key, val in self.__dict__.items():
            if isinstance(val, _CONFIG):
                print("\t" * indent + f"{key}:")
                val.show(indent=indent + 1)
            else:
                _val = val.replace("\n", "\\n").replace("\t", "\\t") if isinstance(val, str) else val
                print("\t" * indent + f"{key}: " + (f"'{_val}'" if isinstance(val, str) else f"{val}"))

    def asdict(self):
        """Convert config object to dict."""
        dict_out = {}
        for key, val in self.__dict__.items():
            if isinstance(val, _CONFIG):
                dict_out.update({key: val.asdict()})
            else:
                dict_out.update({key: val})
        return dict_out

    def update_paths(self, parent_path: str | None = None, for_logging: bool = False):
        """Update relative paths to PROJECT_ROOT dir."""
        # Use the project root dir as the parent path if not specified
        parent_path = self.PROJECT_ROOT if hasattr(self, "PROJECT_ROOT") else parent_path

        if parent_path is not None:
            parent_path = str(Path(parent_path).absolute())

            for key, path in self.__dict__.items():
                if isinstance(path, str) and not Path(path).is_absolute():
                    if for_logging and key != "filename":
                        # In the case of logging configs, apply only on filename
                        continue
                    self.__dict__.update({key: str(Path(parent_path).joinpath(path))})

                elif isinstance(path, _CONFIG):
                    path.update_paths(parent_path=parent_path, for_logging=for_logging)

        else:
            print("\033[91mPaths can't be converted to absolute paths, since no PROJECT_ROOT is found!\033[0m")  # red


def _set_wd(new_dir: str | Path) -> None:
    """
    Set the given directory as the new working directory of the project.

    :param new_dir: name of new working directory (must be in the project folder)
    """
    if PROJECT_NAME not in str(Path.cwd()):
        msg = f"Current working dir '{Path.cwd()}' is outside of project '{PROJECT_NAME}'."
        raise OSError(msg)

    print("\033[94m" + f"Current working dir:\t{Path.cwd()}" + "\033[0m")  # print blue

    # Check if new_dir is a folder path or just a folder name
    new_dir = Path(new_dir)
    if new_dir.is_absolute():
        found = new_dir.is_dir()
        change_dir = new_dir.absolute() != Path.cwd()
        if found and change_dir:
            os.chdir(new_dir)

    else:
        # Remove '/' if new_dir == 'folder/' OR '/folder'
        new_dir = new_dir.name

        # Check if new_dir is the current dir
        found = new_dir == Path.cwd().name
        change_dir = not found

        # First look down the tree
        if not found:
            # Note: This works only for unique folder names
            paths_found = sorted(Path(PROJECT_ROOT).parent.glob(f"**/{new_dir}"), key=lambda x: len(x.parents))
            if len(paths_found) > 1:
                msg = (
                    f"Found multiple folders with name '{new_dir}' in project '{PROJECT_NAME}'!\n\n"
                    f"Please specify the absolute path to the desired folder:\n\n{[str(p) for p in paths_found]}"
                )
                raise ValueError(msg)

            if len(paths_found) == 1:
                found = True
                os.chdir(paths_found.pop())

    if found and change_dir:
        print("\033[93m" + f"New working dir: '{Path.cwd()}'\n" + "\033[0m")  # yellow print
    elif found and not change_dir:
        pass
    else:
        print("\033[91m" + f"Given folder not found. Working dir remains: '{Path.cwd()}'\n" + "\033[0m")  # red print


# %% Setup configuration object < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Instantiate config object
config = _CONFIG()  # ready for import in other scripts

# Load config file(s)
for config_file in Path(__file__).parent.glob("../configs/*config.toml"):
    if sys.version_info >= (3, 11):
        with config_file.open("rb") as f:
            config.update(new_configs=toml.load(f))
    else:
        config.update(new_configs=toml.load(str(config_file)))

# Extract some useful globals
PROJECT_NAME = config.PROJECT_NAME  # ready for import in other scripts

# Get the project root path
if hasattr(config.paths, "PROJECT_ROOT"):
    PROJECT_ROOT = config.paths.PROJECT_ROOT  # ready for import in other scripts
else:
    PROJECT_ROOT = __file__[: __file__.find(PROJECT_NAME) + len(PROJECT_NAME)]
    # Set the root path to the config file & update paths
    config.paths.PROJECT_ROOT = PROJECT_ROOT
    config.paths.update_paths()

# Prepare logging
config.logging.update_paths(parent_path=PROJECT_ROOT, for_logging=True)
_create_parent_dirs(config_as_dict=config.logging.asdict())


# Extract paths & params and set logging configs
paths = config.paths  # ready for import in other scripts
params = config.params  # ready for import in other scripts
logging.config.dictConfig(config.logging.asdict())  # in other scripts: import logging & logging.getLogger(__name__)

# Welcome
_w = 95
print("\n" + ("*" * _w + "\n") * 2 + "\n" + f"{PROJECT_NAME:^{_w}}" + "\n" * 2 + ("*" * _w + "\n") * 2)

# Set the project working directory
_set_wd(PROJECT_ROOT)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
