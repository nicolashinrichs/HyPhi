# ============= #
# Preliminaries # 
# ============= # 

from FileIO import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.colors as mpc
from matplotlib.transforms import ScaledTranslation
from tqdm import tqdm

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
if curv_type == "FRC":
    cmethod = "1d"
elif curv_type == "AFRC":
    cmethod = "augmented"

makeDir(config["kuramoto_viz_loc"])

# ========= #
# Load Data # 
# ========= # 

# Array to hold entropy and quantiles of replications
Hreps = np.zeros((len(config["num_kuramotos"]), config["kuramoto_time"]))
Qreps = np.zeros((len(config["num_kuramotos"]), config["kuramoto_time"], len(config["quantiles"])))

for num in config["num_kuramotos"]:
    # Load data
    Hpath = path.abspath(path.join(config["kuramoto_result_loc"], f"Kuramoto_PLV_{cmethod}_FRC_entropy_cond_{num}_config_{config["config_id"]}.npy"))
    Qpath = path.abspath(path.join(config["kuramoto_result_loc"], f"Kuramoto_PLV_{cmethod}_FRC_quantiles_cond_{num}_config_{config["config_id"]}.npy"))
    if num == "avg":
        Havg = np.load(Hpath)
        Qavg = np.load(Qpath)
    else:
        Hreps[num, :] = np.load(Hpath)
        Qreps[num, :, :] = np.load(Qpath)

# Get the quantiles over replications
rep_qvals = np.array([0.05, 0.5, 0.95])
HQs = np.quantile(Hreps, rep_qvals, axis=0)
QQs = np.quantile(Qreps, rep_qvals, axis=0)

# ======== #
# Plotting # 
# ======== # 

def plotKuramotos(tt, HQs, Havg, QQs, Qavg, qvals):
    # Initialize figure
    fig = plt.figure(figsize=(12, 3), layout="constrained")
    # GridSpec layout: 2 rows, 4 columns
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)
    # Plot entropy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tt, HQs[1, :], c="C0")
    ax1.plot(tt, Havg, linestyle="--", c="C0")
    ax1.fill_between(tt, HQs[0, :], HQs[2, :], facecolor="C0", alpha=0.3)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_xlabel("Time Window")
    ax1.set_ylabel(f"{curv_type} Entropy")
    # Take colors at regular intervals spanning the colormap
    cmap = cm.viridis
    qcolors = cmap(np.linspace(0, 1, 5))
    # Plot quantiles
    ax2 = fig.add_subplot(gs[0, 1])
    for j in range(5):
        ax2.plot(tt, QQs[1, :, j], c=qcolors[j])
        ax2.plot(tt, Qavg[:, j], linestyle="--", c=qcolors[j])
        ax2.fill_between(tt, QQs[0, :, j], QQs[2, :, j], facecolor=qcolors[j], alpha=0.5)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.set_xlabel("Time Window")
    ax2.set_ylabel("Quantile")
    # Plot quantile colorbar
    # Create a dedicated axis for the colorbar
    qcbar_ax = fig.add_subplot(gs[0, -1])
    # Create a custom colormap from the list of colors
    qcmap = mpc.LinearSegmentedColormap.from_list("seg_viridis", qcolors, N=5)
    # Define the range of values for the colorbar
    qbounds = [0, 1, 2, 3, 4, 5]
    qnorm = mpc.BoundaryNorm(qbounds, 5)
    # Create a ScalarMappable
    q_scalar_mappable = cm.ScalarMappable(cmap=qcmap, norm=qnorm)
    q_scalar_mappable.set_array([0.5, 1.5, 2.5, 3.5, 4.5])
    # Add the colorbar to the dedicated axis
    qcbar = fig.colorbar(q_scalar_mappable, cax=qcbar_ax, orientation='vertical')
    qcbar.set_label(f"{curv_type} Quantile", labelpad=20)
    # Optional: Set ticks and labels for the colorbar
    tick_locations = [0.5, 1.5, 2.5, 3.5, 4.5]
    qcbar.set_ticks(tick_locations)
    qcbar.set_ticklabels([f"{qvals[j]}" for j in range(5)])
    return fig


# Final plot
time_axis = np.array(range(len(Havg)))
f = plotKuramotos(time_axis, HQs, Havg, QQs, Qavg, config["quantiles"])
kurviz = path.abspath(path.join(config["kuramoto_viz_loc"], f"kuramoto_PLV_{curv_type}.png"))
f.savefig(kurviz, bbox_inches="tight")