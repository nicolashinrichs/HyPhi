## What this changes

A short, scientific-register description of the change and why it is needed. Link the issue it
closes (`Closes #N`).

## How I validated it

The exact commands run and their results. Validate by running, not by reading: paste the test
output, the import check, or the figure you inspected. If the change touches the analysis
pipeline, confirm the quickstart still executes end to end.

## Checklist

This mirrors the pull request checklist in `CONTRIBUTING.md`:

- [ ] `make check` is clean (ruff format, ty typecheck, ruff lint)
- [ ] `make test` is green
- [ ] `make tutorial` still executes the quickstart end to end (if the pipeline was touched)
- [ ] New public API additions appear in the relevant submodule `__all__`
- [ ] NumPy-style docstrings are present on new public functions, with the method reference
- [ ] No new `TODO` or `FIXME` comments without a tracked issue
