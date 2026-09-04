"""Temporal-tensor helpers for stacking and visualising adjacency matrices.

Stack per-window adjacency matrices into a single ``(T, N, N)`` array and
visualise inter-window similarity as a heatmap.

The original contributor version executed the heatmap-rendering block at
module import time using a free-floating ``temporal_tensor`` variable, which
raised ``NameError`` on import.  The same code now lives in
:func:`plot_state_stability_heatmap`, which takes the tensor as an argument.
"""

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def create_temporal_tensor(folder_path, file_pattern="*_matrix.npy"):
    """Stack 2D adjacency matrix files into a single 3D temporal tensor.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing the ``.npy`` matrix files.
    file_pattern : str, optional
        Glob pattern used to match files inside *folder_path*.
        Defaults to ``"*_matrix.npy"``.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, N, N)`` where *T* is the number of matched files
        and *N* is the node count inferred from the first file.
    """
    matrix_files = sorted(Path(folder_path).glob(file_pattern))
    return np.stack([np.load(f) for f in matrix_files])


def plot_state_stability_heatmap(
    temporal_tensor,
    *,
    cmap="coolwarm",
    figsize=(9, 7),
    style="seaborn-v0_8-muted",
    title="Hyper-Brain State Stability Over Time",
    show=False,
):
    """Compute frame-by-frame correlation similarity and render it as a heatmap.

    Compute frame-by-frame correlation similarity from a ``(T, N, N)`` tensor
    of per-window adjacency matrices and render it as a heatmap.

    Parameters
    ----------
    temporal_tensor : np.ndarray
        ``(num_frames, n_nodes, n_nodes)`` tensor produced by
        :func:`create_temporal_tensor`.
    cmap : str, optional
        Colormap name passed to :func:`matplotlib.pyplot.imshow`.
        Defaults to ``"coolwarm"``.
    figsize : tuple of int, optional
        Figure dimensions ``(width, height)`` in inches.  Defaults to
        ``(9, 7)``.
    style : str, optional
        Matplotlib style name.  If the style does not exist in the installed
        version, it is silently ignored.  Defaults to
        ``"seaborn-v0_8-muted"``.
    title : str, optional
        Title placed above the heatmap.
    show : bool, optional
        If ``True``, also call :func:`matplotlib.pyplot.show`.
        Defaults to ``False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    similarity_matrix : np.ndarray
        ``(num_frames, num_frames)`` pairwise correlation-based similarity
        matrix.
    """
    if style:
        with contextlib.suppress(OSError, ValueError):
            plt.style.use(style)

    num_frames = temporal_tensor.shape[0]
    flat_tensor = temporal_tensor.reshape(num_frames, -1)

    dist_matrix = squareform(pdist(flat_tensor, metric="correlation"))
    similarity_matrix = 1 - dist_matrix

    fig = plt.figure(figsize=figsize)
    im = plt.imshow(similarity_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    cbar = plt.colorbar(im)
    cbar.set_label("Network Similarity (Correlation)", rotation=270, labelpad=15)

    plt.title(title, fontsize=16)
    plt.xlabel("Frame (time point)", fontsize=12)
    plt.ylabel("Frame (time point)", fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, similarity_matrix
