"""
Plotting helpers for the Forman-Ricci / Graph Diffusion Distance (GDD) pipeline.

Companion module to :mod:`hyphi.modeling.GDD_FRc_helpers` and
:mod:`hyphi.modeling.transform_curvature`.  The two ``GDD_FRc_*.ipynb``
notebooks in ``code/notebooks/`` use these functions to render:

- per-matrix histograms of (transformed) curvature distributions,
- a successive-GDD line plot for time-ordered graph sequences,
- a pairwise-GDD heatmap for unordered graph collections.

All functions return matplotlib objects; nothing is shown automatically.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_1d_array(values) -> np.ndarray:
    """Return ``values`` as a 1D float array (flattening dicts of edge values)."""
    if isinstance(values, Mapping):
        values = list(values.values())
    arr = np.asarray(values, dtype=float).ravel()
    return arr


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_weight_distributions_by_matrix(
    curvature_dict: Mapping,
    bins: int | str = "fd",
    ncols: int = 2,
    main_title: str | None = None,
    figsize: tuple[float, float] | None = None,
    color: str = "steelblue",
    alpha: float = 0.85,
):
    """
    Plot one histogram per matrix in a grid.

    Parameters
    ----------
    curvature_dict
        Mapping ``{key: values}`` where ``values`` is an array of curvature/weight
        values or a ``{(u, v): value}`` edge dict.  One subplot is drawn per key.
    bins
        Passed straight through to :func:`matplotlib.pyplot.hist` (default ``"fd"``
        = Freedman-Diaconis).
    ncols
        Number of columns in the subplot grid.
    main_title
        Optional figure-level title (``suptitle``).
    figsize
        Figure size in inches.  Defaults to ``(5 * ncols, 3 * nrows)``.
    color, alpha
        Histogram styling.

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = list(curvature_dict.keys())
    n = len(keys)
    if n == 0:
        raise ValueError("curvature_dict is empty; nothing to plot.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.0 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, keys):
        arr = _as_1d_array(curvature_dict[key])
        if arr.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(arr, bins=bins, color=color, alpha=alpha, edgecolor="black", linewidth=0.4)
        ax.set_title(f"matrix {key}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)

    # Hide any leftover axes if n < nrows*ncols
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if main_title:
        fig.suptitle(main_title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    return fig


def plot_successive_gdd(
    successive_gdd: Mapping,
    *,
    title: str = "Successive graph diffusion distance",
    xlabel: str = "previous-graph index",
    ylabel: str = "GDD(prev, next)",
    figsize: tuple[float, float] = (8.0, 4.0),
    marker: str = "o",
    color: str = "tab:blue",
):
    """
    Line plot of successive GDD values keyed by the index of the *previous* graph.

    Expected input shape comes from
    :func:`hyphi.modeling.GDD_FRc_helpers.compute_successive_gdd`:
    ``{prev_idx: distance, ...}``.

    Parameters
    ----------
    successive_gdd
        ``{prev_idx: distance}`` dict.  Plotted in sorted-key order.
    title, xlabel, ylabel
        Axis labels.
    figsize
        Figure size in inches.
    marker, color
        Line styling.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if not successive_gdd:
        raise ValueError("successive_gdd is empty; nothing to plot.")

    keys = sorted(successive_gdd.keys())
    values = [float(successive_gdd[k]) for k in keys]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(keys, values, marker=marker, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_gdd_heatmap(
    pairwise_gdd: np.ndarray,
    labels: Sequence | None = None,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
    cmap: str = "viridis",
    title: str | None = "Pairwise GDD heatmap",
    annotate: bool = False,
    annotation_fmt: str = "{:.2g}",
    cbar_label: str = "GDD",
):
    """
    Heatmap of a symmetric pairwise GDD distance matrix.

    Parameters
    ----------
    pairwise_gdd
        2D array of shape ``(n, n)`` with non-negative entries; expected to be
        symmetric and zero on the diagonal (as produced by
        :func:`hyphi.modeling.GDD_FRc_helpers.compute_pairwise_gdd_matrix`).
    labels
        Optional length-``n`` axis tick labels.
    figsize
        Figure size in inches.
    cmap
        Matplotlib colormap.
    title
        Plot title.
    annotate
        If True, write the numeric value in each cell using ``annotation_fmt``.
    annotation_fmt
        Python format string applied to each cell value when ``annotate=True``.
    cbar_label
        Label drawn next to the colorbar.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    arr = np.asarray(pairwise_gdd, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"pairwise_gdd must be a square 2D array; got shape {arr.shape}."
        )

    n = arr.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, aspect="auto")

    if labels is not None:
        if len(labels) != n:
            raise ValueError(
                f"labels length ({len(labels)}) does not match matrix size ({n})."
            )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    annotation_fmt.format(arr[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )

    fig.tight_layout()
    return fig, ax
