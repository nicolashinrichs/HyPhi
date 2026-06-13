## What this changes

A short, scientific-register description of the change and why it is needed. Link the issue it
closes (`Closes #N`).

## How I validated it

The exact commands run and their results. Validate by running, not by reading: paste the test
output or the import check. If the change touches the analysis pipeline, confirm it still runs
end to end.

## Checklist

These mirror the CI gates:

- [ ] `uv run --extra develop ruff format --check code/hyphi code/tests` is clean
- [ ] `uv run --extra develop pytest code/tests` is green
- [ ] Every module still imports (CI runs an import-every-module smoke check)
- [ ] `uv run --extra develop ruff check code/hyphi` introduces no new lint (advisory gate)
- [ ] New public API additions appear in the relevant submodule `__all__`
- [ ] NumPy-style docstrings are present on new public functions, with the method reference
- [ ] No new `TODO` or `FIXME` comments without a tracked issue
