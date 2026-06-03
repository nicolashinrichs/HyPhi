# Contributing to HyPhi

Thanks for your interest in contributing.
This document collects the project conventions,
plus the commands you need to develop locally.

## Setting up the repository

Clone to a desired location on your device

```shell
git clone https://github.com/nicolashinrichs/HyPhi
cd HyPhi
```

### Install dependencies

We recommend using the [`uv`](https://docs.astral.sh/uv/) package manager
(if not installed yet, see: https://docs.astral.sh/uv/getting-started/installation/)

Use either the `make` command

```shell
make install
```

or

```shell
uv sync --extra develop --extra notebook
```

*(leave out the `--extra notebook` flag, if not required)*

Add `--extra experiments` if you plan to run anything under `experiments/scripts/`.

### Create a feature branch

Create a branch to develop your feature:

```shell
git checkout -b my-feature-branch
```

## Adding modules to the `hyphi` package

The `hyphi` package code resides in `./code/hyphi/`.

To add new functionality, there are three ways:

### 1. Add a function to an existing module

For instance, add a function to the existing `./code/hyphi/stats.py`

```python
# ./code/hyphi/stats.py
def my_new_function():
    # do something great
    ...
```

This can be used in another script or notebook in the following way:

```python
from hyphi.stats import my_new_function

# Use the new function
my_new_function()
```

### 2. Add a new submodule with its own functionality in an existing module

Create a new Python file, becoming a submodule of `hyphi` with several functions and or classes.
Place the file where it makes sense:

```shell
touch code/hyphi/visualization/my_viz_tools.py
```

Here, we add a submodule to the `visualization` module of `hyphi`.


Provide functionality in this new submodule:

```python
# code/hyphi/visualization/my_viz_tools.py

def my_viz_function1():
    # great visualization 1
    ...

def my_viz_function2():
    # great visualization 2
    ...
```

Again, this new submodule can be now used in other scripts or notebooks in the following way:

```python
from hyphi.visualization.my_viz_tools import my_viz_function1, my_viz_function2

# Use the new functions
my_viz_function1()
my_viz_function2()
```

### 3. Create a separate module

First, create the container for the new module, that is just a folder:

```shell
mkdir code/hyphi/my_module
```

Then drop an `__init__.py` file in that folder such that Python recognizes the folder as module.

```shell
touch code/hyphi/my_module/__init__.py
```

Then, similar to Option 2 (see [above](#2-add-a-new-submodule-with-its-own-functionality-in-an-existing-module)), create a submodule with its functionality.

```shell
touch code/hyphi/my_module/my_sub_module.py
```

And add functions to it:

```python
# code/hyphi/my_module/my_sub_module.py

def foo():
    # great foo function
    ...

def bar():
    # great bar function
    ...
```

To expose the module directly via `import hyphi` (so that `hyphi.my_module` works without an explicit import), add one line to the main `__init__.py`
at the root of the package folder `./code/hyphi/__init__.py`.

*Note: explicit imports like `from hyphi.my_module.my_sub_module import foo` will work regardless of this step.*

```python
# ./code/hyphi/__init__.py
...

# add this line
import hyphi.my_module  # matches the folder containing the module

...
```

Again, this new module and its submodules can be now used in other scripts or notebooks in the following way:

> **Note:** When adding new functionality, make sure to also add tests (see [below](#add-tests-for-added-functionality)).

```python
from hyphi.my_module.my_sub_module import foo, bar

# Use the new functions
foo()
bar()
```

### Module template

Files in `code/hyphi/` use a lightweight section-marker template:

```python
"""Single-line description of what the module is for."""

# %% Import
import ...

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def public_function(...):
    """Add a docstring here."""
    ...

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
```

Keep the markers when editing existing files.
New files do not have to introduce them.

## Add tests for added functionality

Tests live in `./code/tests/` and use [pytest](https://docs.pytest.org/) conventions.

### 1. Create a test file

For each new (sub)module, create a corresponding test file prefixed with `test_`:

```shell
touch code/tests/test_my_sub_module.py
```

### 2. Write test functions

Test functions must also be prefixed with `test_`. Import the functionality you want to test and assert expected behaviour:

```python
# code/tests/test_my_sub_module.py

from hyphi.my_module.my_sub_module import foo, bar


def test_foo():
    result = foo()
    assert result is not None


def test_bar():
    result = bar()
    assert result == expected_value
```

For related tests, you can group them in a class:

```python
# code/tests/test_my_sub_module.py

class TestFoo:
    """Tests for the foo function."""

    def test_foo_returns_value(self):
        assert foo() is not None

    def test_foo_edge_case(self):
        assert foo(edge_input) == expected_output
```

### 3. Run the tests

Use the `make` command:

```shell
make test
```

Or run pytest directly:

```shell
uv run --extra develop pytest code/tests --cov-report=html -v
```

To run only a specific test file:

```shell
uv run pytest code/tests/test_my_sub_module.py -v
```

## Code style

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/), configured in
`pyproject.toml`.  Key rules:

- Line length 119.
- PEP 8 naming throughout (no `mixed_Case` filenames, no all-caps function names, ASCII
  parameter names only).
- Internal helpers use a leading underscore; the public surface of every submodule is
  defined by its `__init__.py` `__all__` list.
- `from __future__ import annotations` is fine in new files but not required.

Run `make format`, `make lint`, and `make typecheck` (or for all shortly: `make check`) before opening a PR.

## Docstrings

NumPy style.  Match the terse register already in the codebase:

- One-liner when the function name + signature already make the behaviour obvious
  (`compute_global_efficiency` is a good example in `hyphi/benchmarks.py`).
- Full block with `Parameters` / `Returns` only when there are non-obvious shapes,
  parameter semantics, or edge cases (`hyphi/null_models.py:phase_randomize` is the
  reference for full blocks).
- No `Examples` section unless the surrounding module already uses them.


## Adding, extending, using `HyPhi` notebooks

The repository includes notebooks for interactive exploration and analysis in `./code/notebooks/`:

| Notebook | Format |
|---|---|
| `code/notebooks/hyphi.ipynb` | Jupyter notebook |
| `code/notebooks/hyphi_explore.py` | [marimo](https://marimo.io/) notebook |

### Running notebooks

**Jupyter:**

```shell
uv run --extra notebook jupyter lab code/notebooks/hyphi.ipynb
```

**marimo:**

```shell
uv run --extra notebook marimo edit code/notebooks/hyphi_explore.py
```

> **Note:** Make sure the notebook kernel points to the project's virtual environment (`.venv`) so that `import hyphi` works.




## Configuration

> *UNDER CONSTRUCTION*: This might change soon (TODO)

Two loaders, two scopes:

- `hyphi.configs.config` / `paths` / `params` — project-wide singleton populated from
  `code/configs/*config.toml`.  Library code can read it without side effects; entry
  points (`hyphi.main`, tutorial notebooks, experiment scripts) call
  `hyphi.configs.bootstrap()` once at startup to chdir, configure logging, and print the
  banner.
- `hyphi.io.load_config(path)` — per-file TOML loader for arbitrary user configs (e.g.
  `experiments/configs/*.toml`).  Returns a plain dict.

Do not introduce a third loader.

## Tests

```shell
make test
```

Tests live in `code/tests/`; shared fixtures go in `code/tests/conftest.py`.  New code
should ship with a corresponding `test_<module>.py`.

## Versioning

Versions are bumped with [bumpver](https://github.com/mbarkhau/bumpver):

```shell
uv run bumpver update --patch --no-fetch --dry   # preview
uv run bumpver update --patch                    # actually bump
```

The patterns in `[tool.bumpver.file_patterns]` keep `pyproject.toml`, the README badge and
citation, and `CITATION.cff` in sync.

## Pull request checklist

- `make check` clean
- `make test` green
- `make tutorial` still executes the quickstart notebook end-to-end
- Public API additions appear in the relevant submodule `__all__`
- Docstrings present on new public functions
- No new TODO / FIXME comments without a tracked issue
