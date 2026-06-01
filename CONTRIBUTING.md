# Contributing to HyPhi

Thanks for your interest in contributing.
This document collects the project conventions
that tooling does not enforce, plus the commands you need to develop locally.

## Setup

```shell
uv sync --extra develop --extra notebook
```

Add `--extra experiments` if you plan to run anything under `experiments/scripts/`, and
`--extra viz` for the optional `seaborn` / `plotly` plotting paths.

## Code style

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/), configured in
`pyproject.toml`.  Key rules:

- Line length 119.
- PEP 8 naming throughout (no `mixed_Case` filenames, no all-caps function names, ASCII
  parameter names only).
- Internal helpers use a leading underscore; the public surface of every submodule is
  defined by its `__init__.py` `__all__` list.
- `from __future__ import annotations` is fine in new files but not required.

Run `make format` and `make lint` before opening a PR.

## Docstrings

NumPy style.  Match the terse register already in the codebase:

- One-liner when the function name + signature already make the behaviour obvious
  (`compute_global_efficiency` is a good example in `hyphi/benchmarks.py`).
- Full block with `Parameters` / `Returns` only when there are non-obvious shapes,
  parameter semantics, or edge cases (`hyphi/null_models.py:phase_randomize` is the
  reference for full blocks).
- No `Examples` section unless the surrounding module already uses them.

## Module template

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

Keep the markers when editing existing files.  New files do not have to introduce them.

## Configuration

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

- `make lint` clean
- `make test` green
- `make tutorial` still executes the quickstart notebook end-to-end
- Public API additions appear in the relevant submodule `__all__`
- Docstrings present on new public functions
- No new TODO / FIXME comments without a tracked issue
