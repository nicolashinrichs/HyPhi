# ============= #
# Preliminaries # 
# ============= # 

import numpy as np
from FileIO import *
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# Path variables
basepath = path.dirname(__file__)
configpath = path.abspath(path.join(basepath, "..", "experiments", "analysis"))

# ========================== #
# Load CCORR Analysis Config # 
# ========================== # 

# Analysis configuration file
configfile = path.abspath(path.join(configpath, sys.argv[1]))

# Load the configuration parameters into a dictionary
config = loadConfig(configfile)

# Type of curvature
curv_type = sys.argv[2]
assert curv_type in ["FRC", "AFRC"], f"Curvature type ({curv_type}) must be one of (FRC, AFRC)!"

def fullyPooledPathConstructor(trial_type, curvature, config):
    if curvature == "FRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_matrix_fully_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    elif curvature == "AFRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_matrix_fully_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    return FRCpath


data = {}
for tt in config["trial_types"]:
    FRCpath = fullyPooledPathConstructor(tt, curv_type, config)
    data[tt] = np.load(FRCpath)

# Downsample first
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


# Usage (do this ONCE at the start):
data_downsamp = downsample_fully_pooled(data, sample_size=200_000)

# ======== #
# Plotting # 
# ======== # 

# Visualization path variables
hyperviz = path.abspath(config["pool_viz_loc"])
makeDir(hyperviz)

# Colorblind friendly palette (8 colors) to set the color cycle of plots (Bang Wong's palette)
wong = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# Set Matplotlib rc params
params = {
    'axes.labelsize': 20,
    'axes.unicode_minus': False,
    'axes.titlesize': 20,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'sans-serif',
    'axes.prop_cycle': mpl.cycler(color=wong)
}
plt.rcParams.update(params)

# Function to plot ECDFs
def plot_all_bands_ecdf(fully_pooled, curvature, trial_types=config["trial_types"], band_names=config["freq_bands"]):
    n_bands = 8
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for f in range(n_bands):
        ax = axes[f]
        ax.set_xscale('symlog')
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
    plt.tight_layout(rect=[0,0.05,0.9,1])

    return fig


# Plot ECDFs
fullpool_ecdfs = plot_all_bands_ecdf(data_downsamp, curv_type)

# Save figure
figpath = path.abspath(path.join(hyperviz, f"fully_pooled_{curv_type}_ecdfs.png"))
fullpool_ecdfs.savefig(figpath, bbox_inches="tight")
