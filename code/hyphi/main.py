"""
End-to-end HyPhi pipeline.

One runnable command for the canonical workflow: a connectivity series (windows of an N x N
connectivity matrix) becomes a series of graphs, each annotated with Forman-Ricci curvature,
whose curvature distribution is summarised by an entropy and quantiles, giving the
curvature-entropy time series HyPhi reads phase transitions from.

Run it with ``python -m hyphi.main`` (or ``make pipeline``). With no input it runs on a demo
connectivity series built from a Watts-Strogatz sweep, so it is self-contained; point ``--input``
at a saved ``(windows, nodes, nodes)`` ``.npy`` array to run on your own connectivity.
"""

# %% Import
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from hyphi.analyses import build_sliding_window_graphs, compute_entropy, compute_windowed_curvatures
from hyphi.modeling.entropies import DEFAULT_ENTROPY_METHOD, get_estimator, vec_quantiles
from hyphi.simulation.graph_simulations import gen_weighted_sw

_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_VALID_CURVATURE_METHODS = ("1d", "augmented")  # Forman-Ricci variants compute_frc accepts
_WS_DEGREE = 4  # Watts-Strogatz ring degree for the demo connectivity
_MATRIX_NDIM = 2  # a single (nodes, nodes) connectivity matrix
_SERIES_NDIM = 3  # a (windows, nodes, nodes) connectivity series


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def demo_connectivity(n_nodes: int = 20, n_windows: int = 8, seed: int = 0) -> np.ndarray:
    """
    Build a small demo connectivity series from a Watts-Strogatz sweep.

    Each window is the weighted adjacency of a Watts-Strogatz graph at an increasing rewiring
    probability, so the series passes through the small-world structural transition, a ground
    truth the curvature-entropy trace should respond to.

    Parameters
    ----------
    n_nodes : int
        Nodes per window.
    n_windows : int
        Number of windows (rewiring-probability grid points).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Connectivity tensor of shape ``(n_windows, n_nodes, n_nodes)``.

    Raises
    ------
    ValueError
        If ``n_windows < 1`` or ``n_nodes`` does not exceed the Watts-Strogatz degree.

    """
    if n_windows < 1:
        msg = f"n_windows must be at least 1, got {n_windows}."
        raise ValueError(msg)
    if n_nodes <= _WS_DEGREE:
        msg = f"n_nodes must exceed the Watts-Strogatz degree {_WS_DEGREE} (got {n_nodes})."
        raise ValueError(msg)
    probabilities = np.logspace(-2, 0, n_windows)
    mats = [
        nx.to_numpy_array(gen_weighted_sw(n_nodes, _WS_DEGREE, float(p), 1.0, seed_val=seed)) for p in probabilities
    ]
    return np.stack(mats, axis=0)


def run_pipeline(
    connectivity: np.ndarray,
    entropy_method: str = DEFAULT_ENTROPY_METHOD,
    curvature_method: str = "1d",
) -> dict[str, Any]:
    """
    Run the connectivity-to-curvature-entropy pipeline.

    Parameters
    ----------
    connectivity : np.ndarray
        Connectivity of shape ``(windows, nodes, nodes)`` (or ``(nodes, nodes)`` for one window).
    entropy_method : str
        Entropy estimator name (any key of ``hyphi.modeling.entropies.ESTIMATORS``).
    curvature_method : str
        Forman-Ricci method, ``"1d"`` (Forman) or ``"augmented"`` (augmented Forman).

    Returns
    -------
    dict
        ``{"graphs": curvature-annotated graphs, "entropy": (n_windows,) trace,
        "quantiles": (n_windows, 5) curvature quantiles}``.

    Raises
    ------
    ValueError
        If ``connectivity`` is not a square ``(nodes, nodes)`` matrix or ``(windows, nodes, nodes)``
        series; if ``curvature_method`` is unknown; or if ``entropy_method`` is unknown.

    """
    arr = np.asarray(connectivity, dtype=float)
    if arr.ndim not in (_MATRIX_NDIM, _SERIES_NDIM):
        msg = f"connectivity must be (nodes, nodes) or (windows, nodes, nodes), got shape {arr.shape}."
        raise ValueError(msg)
    if arr.shape[-1] != arr.shape[-2]:
        msg = f"connectivity matrices must be square, got last two dims {arr.shape[-2:]}."
        raise ValueError(msg)
    if curvature_method not in _VALID_CURVATURE_METHODS:
        # FormanRicci silently writes no curvature attribute for an unknown method, which fails far
        # downstream as an opaque KeyError; reject it here, mirroring the entropy estimator check.
        valid = ", ".join(_VALID_CURVATURE_METHODS)
        msg = f"Unknown curvature method {curvature_method!r}; choose from: {valid}."
        raise ValueError(msg)
    get_estimator(entropy_method)  # fail fast with a clear error before the expensive curvature pass

    graphs = build_sliding_window_graphs(arr)
    curved = compute_windowed_curvatures(graphs, method=curvature_method)
    entropy = compute_entropy(curved, method=entropy_method)
    quantiles = vec_quantiles(curved, qs=list(_QUANTILES))
    return {"graphs": curved, "entropy": entropy, "quantiles": quantiles}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: run the pipeline and write the entropy series and a figure."""
    parser = argparse.ArgumentParser(description="Run the HyPhi curvature-entropy pipeline.")
    parser.add_argument("--input", type=str, default=None, help="path to a (windows, nodes, nodes) .npy file")
    parser.add_argument("--output", type=str, default="results", help="directory for the outputs")
    parser.add_argument("--method", type=str, default=DEFAULT_ENTROPY_METHOD, help="entropy estimator name")
    parser.add_argument("--nodes", type=int, default=20, help="nodes per window for the demo series")
    parser.add_argument("--windows", type=int, default=8, help="number of windows for the demo series")
    args = parser.parse_args(argv)

    connectivity = np.load(args.input) if args.input else demo_connectivity(args.nodes, args.windows)
    results = run_pipeline(connectivity, entropy_method=args.method)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "entropy.csv", results["entropy"], delimiter=",")
    np.savetxt(out_dir / "quantiles.csv", results["quantiles"], delimiter=",")

    # The figure uses the viz package; import it here so importing hyphi.main (and run_pipeline)
    # does not pull Matplotlib until a figure is actually produced.
    from hyphi.viz import plot_curvature_entropy  # noqa: PLC0415

    figure = plot_curvature_entropy(results["entropy"])
    figure.savefig(out_dir / "entropy.png", dpi=150, bbox_inches="tight")

    print(f"Pipeline complete over {len(results['entropy'])} windows. Wrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
