"""
Temporal-tensor helpers: stack per-window adjacency matrices into a single
``(T, N, N)`` array and visualise inter-window similarity as a heatmap.

The original contributor version executed the heatmap-rendering block at
module import time using a free-floating ``temporal_tensor`` variable, which
raised ``NameError`` on import.  The same code now lives in
:func:`plot_state_stability_heatmap`, which takes the tensor as an argument.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def create_temporal_tensor(folder_path, file_pattern="*_matrix.npy"):
    """
    Finds and stacks 2D adjacency matrices into a single 3D temporal tensor.
    """
    search_path = os.path.join(folder_path, file_pattern)
    matrix_files = sorted(glob.glob(search_path))

    matrix_list = [np.load(f) for f in matrix_files]
    temporal_tensor = np.stack(matrix_list)

    return temporal_tensor


def plot_state_stability_heatmap(
    temporal_tensor,
    *,
    cmap='coolwarm',
    figsize=(9, 7),
    style='seaborn-v0_8-muted',
    title="Hyper-Brain State Stability Over Time",
    show=False,
):
    """
    Compute frame-by-frame correlation similarity from a ``(T, N, N)`` tensor
    of per-window adjacency matrices and render it as a heatmap.

    Parameters
    ----------
    temporal_tensor
        ``(num_frames, n_nodes, n_nodes)`` tensor produced by
        :func:`create_temporal_tensor`.
    cmap, figsize, style, title
        Plot styling.
    show
        If True, also call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    (matplotlib.figure.Figure, np.ndarray)
        The figure and the ``(num_frames, num_frames)`` similarity matrix.
    """
    if style:
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            # Style may not exist in older matplotlib; fall back silently.
            pass

    num_frames = temporal_tensor.shape[0]
    flat_tensor = temporal_tensor.reshape(num_frames, -1)

    dist_matrix = squareform(pdist(flat_tensor, metric='correlation'))
    similarity_matrix = 1 - dist_matrix

    fig = plt.figure(figsize=figsize)
    im = plt.imshow(similarity_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    cbar = plt.colorbar(im)
    cbar.set_label('Network Similarity (Correlation)', rotation=270, labelpad=15)

    plt.title(title, fontsize=16)
    plt.xlabel("Frame (time point)", fontsize=12)
    plt.ylabel("Frame (time point)", fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, similarity_matrix
