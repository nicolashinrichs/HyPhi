"""
Template marimo notebook for the hyphi toolbox.

Years: 2026
"""
import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploration of `hyphi`

    Running analyses and simulations with `hyphi`

    *Note, this is a placeholder `marimo` notebook.*

    ___

        Years:   2026

    ___
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    # Set global vars & paths
    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyse
    """)
    return


@app.cell
def _(Path):
    print("Current WD:", Path.cwd())
    return


@app.cell
def _(Path, paths):
    # List all nested paths in `code/configs/config.toml`
    paths.show()

    # Get specific path
    print(f"\n\n{paths.results.visualizations = }")
    print(Path(paths.results.visualizations).exists())
    return


@app.cell
def _(params):
    # Look for a specific parameter
    params.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Intermediate summary
    This is an intermediate summary of the analysis.
    We find this and that ...
    """)
    return


@app.cell
def _():
    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    # Import my hyphi package and other modules
    import marimo as mo
    from pathlib import Path

    import numpy as np
    from hyphi.configs import config, params, paths

    return Path, mo, params, paths


if __name__ == "__main__":
    app.run()
