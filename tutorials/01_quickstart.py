"""User-facing quickstart marimo notebook for the hyphi toolbox.

Walks through the canonical HyPhi workflow on a small shipped Kuramoto connectome:
config, adjacency loading, Forman-Ricci curvature, entropy, and a basic
network plot.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HyPhi quickstart

    Minimal end-to-end run of the package on the shipped 8-subject Kuramoto connectome.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Bootstrap project config
    """)
    return


@app.cell
def _():
    # Load hyphi config
    from hyphi.configs import config

    config.init()
    return (config,)


@app.cell
def _(config):
    config.paths.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load shipped connectomes
    """)
    return


@app.cell
def _(config):
    from hyphi.io_brainhack import load_all_kuramoto_adjacencies

    bundle = load_all_kuramoto_adjacencies(indices=range(1, 9), base_dir=config.paths.data.connectome)
    adjacencies = bundle["adjacencies"]
    print(f"Loaded {len(adjacencies)} connectomes; first shape: {adjacencies[1].shape}")
    return (adjacencies,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Forman-Ricci curvature on one connectome
    """)
    return


@app.cell
def _(adjacencies):
    import networkx as nx
    from hyphi.modeling.graph_curvatures import compute_frc, extract_curvatures

    G = nx.from_numpy_array(adjacencies[1])
    G_frc = compute_frc(G, method="1d")
    curvatures = extract_curvatures(G_frc)
    print(f"FRC: n_edges={curvatures.size}, mean={curvatures.mean():.3f}, std={curvatures.std():.3f}")
    return (G_frc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Entropy of the curvature distribution
    """)
    return


@app.cell
def _(G_frc):
    from hyphi.modeling.entropies import entropy_kde_plugin

    h = entropy_kde_plugin(G_frc, curvature="formanCurvature")
    print(f"plug-in KDE entropy: {h:.3f} nats")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Plot
    """)
    return


@app.cell
def _(G_frc):
    from hyphi.visualization.network_plots import plot_curvature_network

    fig = plot_curvature_network(G_frc, curvature_attr="formanCurvature")
    fig
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
