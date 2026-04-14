"""TODO: add docstring"""

# %% Import
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

from hyphi.configs import paths
from hyphi.io import load_config, make_dir

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Analysis configuration file
config_file = os.path.join(paths.experiments.configs, sys.argv[1])

# Load the configuration parameters into a dictionary
config = load_config(config_file)

# Type of curvature
curv_type = sys.argv[2]
assert curv_type in ["FRC", "AFRC"], f"Curvature type ({curv_type}) must be one of (FRC, AFRC)!"


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def fully_pooled_path_constructor(trial_type, curvature, config):
    """TODO: add docstring"""
    if curvature == "FRC":
        FRCpath = os.path.abspath(
            os.path.join(
                config["pooled_result_loc"],
                f"CCORR_FRC_matrix_fully_pooled_trial_type_{trial_type}_config_{config['config_id']}.npy",
            )
        )
    elif curvature == "AFRC":
        FRCpath = os.path.abspath(
            os.path.join(
                config["pooled_result_loc"],
                f"CCORR_aug_FRC_matrix_fully_pooled_trial_type_{trial_type}_config_{config['config_id']}.npy",
            )
        )
    return FRCpath


def downsample_fully_pooled(fully_pooled, sample_size=100_000):
    """
    Downsample all trial type matrices in fully_pooled dict.

    Parameters
    ----------
    fully_pooled : dict
        Shape: {trial_type: (8, N)} where N = 10*30*4*128*128
    sample_size : int
        Target samples per frequency band per trial type

    Returns
    -------
    fully_pooled_small : dict
        Shape: {trial_type: (8, sample_size)}
    """
    fully_pooled_small = {}

    for tt, matrix in fully_pooled.items():
        # Downsample EACH frequency band independently
        small_bands = []
        for f in range(matrix.shape[0]):
            full_band = matrix[f, :]
            idx = np.random.choice(len(full_band), size=sample_size, replace=False)
            small_bands.append(full_band[idx])

        fully_pooled_small[tt] = np.stack(small_bands)

    return fully_pooled_small


def plot_all_bands_ecdf(fully_pooled, curvature, trial_types=config["trial_types"], band_names=config["freq_bands"]):
    """Plot ECDFs."""
    n_bands = 8
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for f in range(n_bands):
        ax = axes[f]
        ax.set_xscale("symlog")
        for tt in trial_types:
            z = np.asarray(fully_pooled[tt][f, :])
            x = np.sort(z)
            y = np.linspace(0, 1, len(z), endpoint=False)
            ax.step(x, y, where="post", label=tt, alpha=0.8)
        ax.set_title(band_names[f] if band_names else f"Band {f}")
        if f % 4 == 0:
            ax.set_ylabel("ECDF")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.text(0.5, 0.02, f"{curvature} Value", ha="center", fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 0.9, 1])

    return fig


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    data = {}
    for tt in config["trial_types"]:
        FRCpath = fully_pooled_path_constructor(tt, curv_type, config)
        data[tt] = np.load(FRCpath)

    # Usage (do this ONCE at the start):
    data_downsamp = downsample_fully_pooled(data, sample_size=200_000)  # downsample first

    # Visualization path variables
    hyperviz = os.path.abspath(config["pool_viz_loc"])
    make_dir(hyperviz)

    # Colorblind friendly palette (8 colors) to set the color cycle of plots (Bang Wong's palette)
    wong = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    # Set Matplotlib rc params
    params = {
        "axes.labelsize": 20,
        "axes.unicode_minus": False,
        "axes.titlesize": 20,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "sans-serif",
        "axes.prop_cycle": mpl.cycler(color=wong),
    }
    plt.rcParams.update(params)

    # Plot ECDFs
    fullpool_ecdfs = plot_all_bands_ecdf(data_downsamp, curv_type)

    # Save figure
    figpath = os.path.abspath(os.path.join(hyperviz, f"fully_pooled_{curv_type}_ecdfs.png"))
    fullpool_ecdfs.savefig(figpath, bbox_inches="tight")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
