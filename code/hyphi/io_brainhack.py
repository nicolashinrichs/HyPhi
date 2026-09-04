"""Brainhack I/O helpers for loading adjacency matrices.

This module bundles loaders for the two graph-pickle datasets used by the
GDD / Forman-Ricci transformation notebooks (``GDD_FRc_Kuramoto.ipynb``,
``GDD_FRc_prebase.ipynb``):

- the per-window Kuramoto connectomes shipped in ``data/connectome/``
- the per-dyad prebase graphs (external; see ``data/README.md`` for download
  instructions; these pickles are not committed to the repo)

Each loader returns plain NumPy adjacency matrices keyed by integer index, so
downstream code (:mod:`hyphi.modeling.gdd_frc_helpers`) can build NetworkX graphs
and compute curvatures without re-implementing the unpickling logic. The
cross-version graph-pickle loader itself lives in :func:`hyphi.io.load_pickle_adjacency`.

Years: 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hyphi.io import load_pickle_adjacency

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "build_kuramoto_paths",
    "build_prebase_paths",
    "load_adjacencies_from_paths",
    "load_all_kuramoto_adjacencies",
    "load_all_prebase_adjacencies",
    "load_pickle_adjacency",
]

# Default run indices (immutable range objects shared as defaults to avoid
# evaluating range() in the signature, per ruff B008).
_DEFAULT_KURAMOTO_INDICES = range(1, 9)
_DEFAULT_PREBASE_INDICES = range(1, 30)


# ------------------------------------------------------------------------------------------
# general
# ------------------------------------------------------------------------------------------


def load_adjacencies_from_paths(
    paths,
    weight_key="weight",
    symmetrize=False,
    return_nodes=False,
    verbose=False,
):
    """Load adjacency matrices for a dictionary of paths keyed by index.

    Parameters
    ----------
    paths : dict
        Dictionary like ``{1: Path(...), 2: Path(...), ...}``.
    weight_key : str, default="weight"
        Edge attribute to use as the adjacency weight.
    symmetrize : bool, default=False
        If True, return ``max(A, A.T)`` for each adjacency matrix.
    return_nodes : bool, default=False
        If True, also return the node order used for each adjacency matrix.
    verbose : bool, default=False
        If True, print basic loading information.

    Returns
    -------
    dict or tuple of (dict, dict)
        ``{idx: adjacency}``, and additionally ``{idx: nodes}`` when
        ``return_nodes`` is True.
    """
    adjacencies = {}
    node_orders = {} if return_nodes else None

    for idx, path in paths.items():
        loaded = load_pickle_adjacency(
            path,
            weight_key=weight_key,
            symmetrize=symmetrize,
            return_nodes=return_nodes,
        )

        if return_nodes:
            adjacency, nodes = loaded
            adjacencies[idx] = adjacency
            node_orders[idx] = nodes
        else:
            adjacencies[idx] = loaded

        if verbose:
            print(f"[{idx}] loaded {path}")
            print("   adjacency shape:", adjacencies[idx].shape)

    if return_nodes:
        return adjacencies, node_orders
    return adjacencies


# ------------------------------------------------------------------------------------------
# kuramoto
# ------------------------------------------------------------------------------------------


def build_kuramoto_paths(
    indices: Iterable[int] = _DEFAULT_KURAMOTO_INDICES,
    base_dir=Path("HyPhi") / "data" / "connectome",
    suffix="connectome_kuramoto.pkl",
):
    """Build per-run Kuramoto connectome pickle paths and check they exist.

    For example, build paths ``1_connectome_kuramoto.pkl`` ...
    ``8_connectome_kuramoto.pkl`` and verify each one is present.

    Parameters
    ----------
    indices : iterable of int, optional
        Run indices to build paths for.
    base_dir : Path, optional
        Directory containing the pickle files.
    suffix : str, optional
        Filename suffix appended after ``{idx}_``.

    Returns
    -------
    dict
        ``{idx: Path}`` for every requested index.

    Raises
    ------
    FileNotFoundError
        If any requested pickle file is missing.
    """
    paths = {idx: Path(base_dir) / f"{idx}_{suffix}" for idx in indices}

    missing = [idx for idx, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(str(idx) for idx in missing)
        raise FileNotFoundError(f"Missing Kuramoto pickle file(s) for index/indices: {missing_str}")

    return paths


def load_all_kuramoto_adjacencies(
    indices: Iterable[int] = _DEFAULT_KURAMOTO_INDICES,
    base_dir=Path("HyPhi") / "data" / "connectome",
    suffix="connectome_kuramoto.pkl",
    weight_key="weight",
    symmetrize=False,
    return_nodes=False,
    verbose=False,
):
    """Load every Kuramoto adjacency matrix and return them with their paths."""
    paths = build_kuramoto_paths(
        indices=indices,
        base_dir=base_dir,
        suffix=suffix,
    )

    loaded = load_adjacencies_from_paths(
        paths,
        weight_key=weight_key,
        symmetrize=symmetrize,
        return_nodes=return_nodes,
        verbose=verbose,
    )

    if return_nodes:
        adjacencies, node_orders = loaded
        return {
            "paths": paths,
            "adjacencies": adjacencies,
            "nodes": node_orders,
        }

    return {
        "paths": paths,
        "adjacencies": loaded,
    }


# ------------------------------------------------------------------------------------------
# prebase
# ------------------------------------------------------------------------------------------


def build_prebase_paths(
    indices: Iterable[int] = _DEFAULT_PREBASE_INDICES,
    base_dir=Path("HyPhi") / "data" / "prebase",
    suffix="_prebase_graph.pkl",
):
    """Build per-dyad prebase graph pickle paths and check they exist.

    For example, build paths ``01_prebase_graph.pkl`` ... ``29_prebase_graph.pkl``
    and verify each one is present.

    Parameters
    ----------
    indices : iterable of int, optional
        Dyad indices to build paths for.
    base_dir : Path, optional
        Directory containing the pickle files.
    suffix : str, optional
        Filename suffix appended after the zero-padded index.

    Returns
    -------
    dict
        ``{idx: Path}`` for every requested index.

    Raises
    ------
    FileNotFoundError
        If any requested pickle file is missing.
    """
    paths = {idx: Path(base_dir) / f"{idx:02d}{suffix}" for idx in indices}

    missing = [idx for idx, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(f"{idx:02d}" for idx in missing)
        raise FileNotFoundError(f"Missing prebase pickle file(s) for index/indices: {missing_str}")

    return paths


def load_all_prebase_adjacencies(
    indices: Iterable[int] = _DEFAULT_PREBASE_INDICES,
    base_dir=Path("HyPhi") / "data" / "prebase",
    suffix="_prebase_graph.pkl",
    weight_key="weight",
    symmetrize=False,
    return_nodes=False,
    verbose=False,
):
    """Load every prebase adjacency matrix and return them with their paths."""
    paths = build_prebase_paths(
        indices=indices,
        base_dir=base_dir,
        suffix=suffix,
    )

    loaded = load_adjacencies_from_paths(
        paths,
        weight_key=weight_key,
        symmetrize=symmetrize,
        return_nodes=return_nodes,
        verbose=verbose,
    )

    if return_nodes:
        adjacencies, node_orders = loaded
        return {
            "paths": paths,
            "adjacencies": adjacencies,
            "nodes": node_orders,
        }

    return {
        "paths": paths,
        "adjacencies": loaded,
    }
