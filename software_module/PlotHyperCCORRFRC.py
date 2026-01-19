# ============= #
# Preliminaries # 
# ============= # 

from FileIO import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mpc
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
from matplotlib.lines import Line2D  # for custom legend handles
from tqdm import tqdm
from os import path
import sys

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

# Create map between dyads and dates
dyad_date_map = dict(zip(config["dyads"], config["dyad_dates"]))

# Create map between trial types and numeric identifiers
# Map to 0, 1, 2 instead of 1, 2, 3 for later
trial_type_ids = list(np.array(config["trial_type_ids"]) - 1)
trial_type_map = dict(zip(trial_type_ids, config["trial_types"]))

# Visualization path variables
hyperviz = path.abspath(config["viz_loc"])
makeDir(hyperviz)

# ======== #
# Plotting # 
# ======== # 

def plotHyperFRC(entropy, quantiles, title=None, band_labels=None, window_labels=None, q_labels=None):
    """
    entropy:   array of shape (8, 30, 4)
    quantiles: array of shape (8, 30, 4, 5)
               last dim = 5 quantiles.
    """

    n_bands, n_trials, n_windows = entropy.shape
    _, _, _, n_q = quantiles.shape
    x = np.arange(n_windows)

    # Entropy: median + IQR across trials
    ent_median = np.median(entropy, axis=1)
    ent_q1 = np.percentile(entropy, 25, axis=1)
    ent_q3 = np.percentile(entropy, 75, axis=1)

    # Quantiles: median + IQR across trials for each quantile
    q_median = np.median(quantiles, axis=1)    # (8, 4, 5)
    q_q1 = np.percentile(quantiles, 25, axis=1)
    q_q3 = np.percentile(quantiles, 75, axis=1)

    fig, axes = plt.subplots(2, n_bands, figsize=(3 * n_bands + 1.5, 6), sharex=True, sharey="row")
    if n_bands == 1:
        axes = axes.reshape(2, 1)

    # --- Set up discrete colormap for quantile index ---
    # Use first n_q Wong colors
    cmap = ListedColormap(wong[1:n_q+1])
    # Quantile indices: 0,1,2,3,4 → boundaries at -0.5,0.5,...,4.5
    boundaries = np.arange(-0.5, n_q + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    for b in range(n_bands):
        # Top: entropy
        ax_ent = axes[0, b]
        ax_ent.spines['top'].set_visible(False)
        ax_ent.spines['right'].set_visible(False)
        ax_ent.plot(x, ent_median[b], marker='o', color=wong[0])
        ax_ent.fill_between(x, ent_q1[b], ent_q3[b], color=wong[0], alpha=0.3)
        if b == 0:
            ax_ent.set_ylabel('Entropy')
        ax_ent.set_title(band_labels[b] if band_labels is not None else f'Band {b}')

        # Bottom: quantiles colored by index
        ax_q = axes[1, b]
        ax_q.spines['top'].set_visible(False)
        ax_q.spines['right'].set_visible(False)
        for q_idx in range(n_q):
            color = cmap(norm(q_idx))
            ax_q.plot(
                x,
                q_median[b, :, q_idx],
                marker='o',
                color=color
            )
            ax_q.fill_between(
                x,
                q_q1[b, :, q_idx],
                q_q3[b, :, q_idx],
                color=color,
                alpha=0.2
            )

        if b == 0:
            ax_q.set_ylabel('Quantile Value')
        if window_labels is not None:
            ax_q.set_xticks(x)
            ax_q.set_xticklabels(window_labels, rotation=45)
        else:
            ax_q.set_xticks(x)
            ax_q.set_xlabel('Time Window')

    # --- Single discrete colorbar on the right of the bottom-right quantile plot ---
    # Create a dummy ScalarMappable using the same cmap/norm
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required in some Matplotlib versions

    # Attach colorbar to the rightmost bottom axes
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.35])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=np.arange(n_q))
    # Label ticks as quantiles
    if q_labels is not None:
        cbar.set_ticklabels(q_labels)
        cbar.set_label('Quantile')
    else:
        cbar.set_ticklabels([f'Q{q_idx + 1}' for q_idx in range(n_q)])
        cbar.set_label('Quantile Index')

    # Figure title
    if title is not None:
        fig.suptitle(title)

    fig.subplots_adjust(right=0.9)  # make room for colorbar
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    return fig, axes


def plotHyperFRCFullExp(
    entropy_list,      # list of 3 arrays, each (8, 30, 4)
    quantiles_list,    # list of 3 arrays, each (8, 30, 4, 5)
    trial_type_seq,    # array-like, length 90, values in {0,1,2} or {1,2,3}
    title=None,
    band_labels=None,
    window_labels=None,
    q_labels=None,
    type_labels=None,
):
    """
    For each trial type k (0,1,2):
      entropy_list[k][b, i, w]   shape (8, 30, 4)
      quantiles_list[k][b, i, w, q] shape (8, 30, 4, 5)

    trial_type_seq[t] gives chronological trial type for t in 0..89.
    Plots:
      - 1 row per band, 2 cols: left = entropy, right = quantiles
      - x-axis is 90 * 4 = 360 windows.
    """
    # Shapes
    n_types = len(entropy_list)
    n_bands, n_trials_per_type, n_windows = entropy_list[0].shape
    _, _, _, n_q = quantiles_list[0].shape

    trial_type_seq = np.asarray(trial_type_seq)
    n_trials_total = trial_type_seq.shape[0]
    assert n_trials_total == n_types * n_trials_per_type, \
        "trial_type_seq length must equal total number of trials across types"

    # Ensure 0,1,2 labels internally
    if trial_type_seq.min() == 1:
        tt_seq = trial_type_seq - 1
    else:
        tt_seq = trial_type_seq.copy()

    # Default type labels if none provided
    if type_labels is None:
        type_labels = [f"T{k+1}" for k in range(n_types)]

    # Time axis
    total_windows = n_trials_total * n_windows
    time_index = np.arange(total_windows)

    # Combined arrays: (bands, total_windows) and (bands, total_windows, n_q)
    ent_combined = np.zeros((n_bands, total_windows))
    quant_combined = np.zeros((n_bands, total_windows, n_q))

    # Counters per trial type
    per_type_counters = np.zeros(n_types, dtype=int)

    # Fill combined arrays according to chronological order
    for t in range(n_trials_total):
        k = tt_seq[t]            # trial type index 0..2
        i = per_type_counters[k] # trial index within type
        per_type_counters[k] += 1

        start = t * n_windows
        end = (t + 1) * n_windows

        for b in range(n_bands):
            ent_combined[b, start:end] = entropy_list[k][b, i, :]
            quant_combined[b, start:end, :] = quantiles_list[k][b, i, :, :]

    # Figure: n_bands rows, 2 columns (entropy | quantiles)
    fig, axes = plt.subplots(
        n_bands, 2,
        figsize=(28, 2.0 * n_bands),
        sharex=True,
        sharey='col'
    )
    axes = np.atleast_2d(axes)

    # Discrete colormap for quantiles (skip black)
    cmap = ListedColormap(wong[1:n_q+1])
    boundaries = np.arange(-0.5, n_q + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Trial boundaries (every 4 samples)
    trial_boundaries = np.arange(0, total_windows, n_windows)

    for b in range(n_bands):
        ax_ent = axes[b, 0]
        ax_q = axes[b, 1]

        # Remove top and right spines
        for ax in (ax_ent, ax_q):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # ---- Entropy time series (left column) ----
        ax_ent.plot(time_index, ent_combined[b], color=wong[0], linewidth=1)
        ax_ent.set_ylabel('Entropy')

        if band_labels is not None:
            ax_ent.set_title(band_labels[b], loc='left')
        else:
            ax_ent.set_title(f'Band {b}', loc='left')

        # Vertical trial boundaries
        for x_trial in trial_boundaries:
            ax_ent.axvline(x=x_trial - 0.5, color='k', linestyle=':', linewidth=0.5)

        # ---- Quantiles time series (right column) ----
        for q_idx in range(n_q):
            color = cmap(norm(q_idx))
            ax_q.plot(
                time_index,
                quant_combined[b, :, q_idx],
                color=color,
                linewidth=1
            )

        for x_trial in trial_boundaries:
            ax_q.axvline(x=x_trial - 0.5, color='k', linestyle=':', linewidth=0.5)

        # Label trial types beneath the quantile axis
        for t in range(n_trials_total):
            x_center = t * n_windows + (n_windows - 1) / 2.0
            trial_type = tt_seq[t]
            label = type_labels.get(trial_type, str(trial_type))
            ax_ent.text(
                x_center,
                -0.05,
                label,
                ha='center',
                va='top',
                rotation=60,
                fontsize=8,
                transform=ax_ent.get_xaxis_transform()
            )
            ax_q.text(
                x_center,
                -0.05,
                label,
                ha='center',
                va='top',
                rotation=60,
                fontsize=8,
                transform=ax_q.get_xaxis_transform()
            )

        ax_q.set_ylabel('Quantile')

        # X-label only on last band row
        if b == n_bands - 1:
            for ax in (ax_ent, ax_q):
                ax.tick_params(axis='x', labelbottom=False)  # hide tick labels
                ax.set_xlabel('Time Window', labelpad=20) # prevent overlap 

    if title is not None:
        fig.suptitle(title, y=0.99)

    fig.tight_layout()

    # Per-band colorbars, anchored as insets to quantile axes   ### CHANGED
    for b in range(n_bands):
        ax_q = axes[b, 1]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # inset_axes sizes in relative axes coordinates (width, height)
        cax = inset_axes(
            ax_q,
            width="1.5%",   # narrow
            height="80%",   # tall
            loc="center right",
            borderpad=1.0
        )
        cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=np.arange(n_q))
        if q_labels is not None:
            cbar.set_ticklabels(q_labels)
            cbar.set_label('Quantile', fontsize=8)
        else:
            cbar.set_ticklabels([f'Q{q_idx + 1}' for q_idx in range(n_q)])
            cbar.set_label('Quantile index', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return fig, axes


def plotHyperFRCFullExp_avgWindows(
    entropy_list,      # list of 3 arrays, each (8, 30, 4)
    quantiles_list,    # list of 3 arrays, each (8, 30, 4, 5)
    trial_type_seq,    # array-like, length 90, values in {0,1,2} or {1,2,3}
    title=None,
    band_labels=None,
    window_labels=None,
    q_labels=None,
    type_labels=None,
):
    """
    Same as plotHyperFRCFullExp, but averages across the 4 windows within each trial.
    Resulting time axis has 90 points (one per trial). Marker shape encodes trial type.
    """

    # Shapes
    n_types = len(entropy_list)
    n_bands, n_trials_per_type, n_windows = entropy_list[0].shape
    _, _, _, n_q = quantiles_list[0].shape

    trial_type_seq = np.asarray(trial_type_seq)
    n_trials_total = trial_type_seq.shape[0]
    assert n_trials_total == n_types * n_trials_per_type, \
        "trial_type_seq length must equal total number of trials across types"

    # Ensure 0,1,2 labels internally
    if trial_type_seq.min() == 1:
        tt_seq = trial_type_seq - 1
    else:
        tt_seq = trial_type_seq.copy()

    # Default type labels if none provided (used in marker legend)
    if type_labels is None:
        type_labels = [f"Type {k+1}" for k in range(n_types)]

    # Time axis: one point per trial
    time_index = np.arange(n_trials_total)

    # Combined arrays: (bands, n_trials_total) and (bands, n_trials_total, n_q)
    ent_combined = np.zeros((n_bands, n_trials_total))
    quant_combined = np.zeros((n_bands, n_trials_total, n_q))

    # Counters per trial type
    per_type_counters = np.zeros(n_types, dtype=int)

    # Fill combined arrays according to chronological order,
    # averaging across the 4 windows within each trial.
    for t in range(n_trials_total):
        k = tt_seq[t]            # trial type index 0..2
        i = per_type_counters[k] # trial index within type
        per_type_counters[k] += 1

        for b in range(n_bands):
            # Average entropy across windows: (4,) -> scalar
            ent_combined[b, t] = entropy_list[k][b, i, :].mean()
            # Average quantiles across windows: (4, 5) -> (5,)
            quant_combined[b, t, :] = quantiles_list[k][b, i, :, :].mean(axis=0)

    # Marker style per trial type
    marker_map = {
        0: '*',   # type 0
        1: 'x',   # type 1
        2: 'd',   # type 2
    }

    # Figure: n_bands rows, 2 columns (entropy | quantiles)
    fig, axes = plt.subplots(
        n_bands, 2,
        figsize=(28, 2.0 * n_bands),
        sharex=True,
        sharey='col'
    )
    axes = np.atleast_2d(axes)

    # Discrete colormap for quantiles (skip black)
    cmap = ListedColormap(wong[1:n_q+1])
    boundaries = np.arange(-0.5, n_q + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    for b in range(n_bands):
        ax_ent = axes[b, 0]
        ax_q = axes[b, 1]

        # Remove top and right spines
        for ax in (ax_ent, ax_q):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # ---- Entropy time series (left column) ----
        # Single continuous line for all trials
        ax_ent.plot(
            time_index,
            ent_combined[b],
            color=wong[0],
            linewidth=1,
            linestyle='-',
        )
        # Overlay markers by trial type, but do not re-draw the line
        for k in range(n_types):
            mask = tt_seq == k
            if not np.any(mask):
                continue
            mk = marker_map.get(k, 'o')
            ax_ent.plot(
                time_index[mask],
                ent_combined[b, mask],
                color=wong[0],
                marker=mk,
                linestyle='None',
                markersize=6,
            )

        ax_ent.set_ylabel('Entropy')

        if band_labels is not None:
            ax_ent.set_title(band_labels[b], loc='left')
        else:
            ax_ent.set_title(f'Band {b}', loc='left')

        # ---- Quantiles time series (right column) ----
        # One line per quantile index across all trials
        for q_idx in range(n_q):
            color = cmap(norm(q_idx))
            ax_q.plot(
                time_index,
                quant_combined[b, :, q_idx],
                color=color,
                linewidth=1,
                linestyle='-',
            )
            # Overlay markers by trial type along that line
            for k in range(n_types):
                mask = tt_seq == k
                if not np.any(mask):
                    continue
                mk = marker_map.get(k, 'o')
                ax_q.plot(
                    time_index[mask],
                    quant_combined[b, mask, q_idx],
                    color=color,
                    marker=mk,
                    linestyle='None',
                    markersize=6,
                )

        ax_q.set_ylabel('Quantile')

        # X-label only on last band row
        if b == n_bands - 1:
            for ax in (ax_ent, ax_q):
                ax.tick_params(axis='x', labelbottom=True)
                ax.set_xlabel('Trial', labelpad=10)

    if title is not None:
        fig.suptitle(title, y=0.99)

    fig.tight_layout()

    # Per-band colorbars, anchored as insets to quantile axes
    for b in range(n_bands):
        ax_q = axes[b, 1]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = inset_axes(
            ax_q,
            width="1.5%",   # narrow
            height="80%",   # tall
            loc="center right",
            borderpad=1.0
        )
        cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=np.arange(n_q))
        if q_labels is not None:
            cbar.set_ticklabels(q_labels)
            cbar.set_label('Quantile', fontsize=8)
        else:
            cbar.set_ticklabels([f'Q{q_idx + 1}' for q_idx in range(n_q)])
            cbar.set_label('Quantile index', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # Per-row legend inset (same style as colorbars)
        ax_ent = axes[b, 0]

        legend_ax = inset_axes(
            ax_ent,                    # anchor to entropy plot (left column)
            width="3%",                # narrow width  
            height="25%",              # short height
            loc="center right",
            borderpad=1.0
        )

        # Create legend in the inset axes
        legend_handles = []
        for k in range(n_types):
            mk = marker_map.get(k, 'o')
            lbl = type_labels[k] if k < len(type_labels) else f"Type {k}"
            legend_handles.append(
                Line2D([], [], color='k', marker=mk, linestyle='None',
                    markersize=6, label=lbl)
            )

        legend_ax.legend(
            handles=legend_handles,
            title='Trial\ntype',
            fontsize=6,
            loc='center'
        )
        legend_ax.axis('off')  # hide the inset axes frame/ticks

    # # ---- Marker legend for trial types (shared) ----
    # legend_handles = []
    # for k in range(n_types):
    #     mk = marker_map.get(k, 'o')
    #     lbl = type_labels[k] if k < len(type_labels) else f"Type {k}"
    #     legend_handles.append(
    #         Line2D(
    #             [], [], color='k', marker=mk, linestyle='None',
    #             markersize=6, label=lbl
    #         )
    #     )
    # # Place legend in an empty corner; here use top-right of the first row's entropy axis
    # axes[0, 0].legend(
    #     handles=legend_handles,
    #     title='Trial type',
    #     loc='center left',
    #     bbox_to_anchor=(1.0, 0.5),  # anchor point at figure right edge
    #     fontsize=8
    # )

    return fig, axes


# ======== #
# Plotting # 
# ======== # 

# Loop over dyads
for dyad in tqdm(config["dyads"], desc="Dyads"):

    # Data path for shot times
    Spath = path.abspath(path.join(config["behav_loc"], f"exp{dyad_date_map[dyad]}"))

    # Load shot time data
    Svals = sp.io.loadmat(Spath)["trialtype"].flatten()

    # List to hold entropies and quantiles
    Hlist = []
    Qlist = []

    # Loop over trial types
    for trial_type in tqdm(config["trial_types"], desc="Trial Types"):

        # Data paths
        Hpath = path.abspath(path.join(config["result_loc"], f"CCORR_FRC_entropy_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Qpath = path.abspath(path.join(config["result_loc"], f"CCORR_FRC_quantiles_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))

        # Load data
        Hvals = np.load(Hpath)
        Hlist.append(Hvals)
        Qvals = np.load(Qpath)
        Qlist.append(Qvals)

        # Plot
        f, _ = plotHyperFRC(Hvals, Qvals, title=f"Dyad: {dyad}, Trial Type: {trial_type}", band_labels=config["freq_bands"], q_labels=config["quantiles"])

        # Plot paths
        figpath = path.abspath(path.join(hyperviz, f"CCORR_FRC_ent_quant_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.png"))

        # Save the figure
        f.savefig(figpath, bbox_inches="tight")

    # Plot full experiment
    ffull, _ = plotHyperFRCFullExp(Hlist, Qlist, Svals, title=f"Dyad: {dyad}", band_labels=config["freq_bands"], q_labels=config["quantiles"], type_labels=trial_type_map)
    ffull_avg, _ = plotHyperFRCFullExp_avgWindows(Hlist, Qlist, Svals, title=f"Dyad: {dyad}", band_labels=config["freq_bands"], q_labels=config["quantiles"], type_labels=trial_type_map)

    # Plot paths
    fullfigpath = path.abspath(path.join(hyperviz, f"CCORR_FRC_ent_quant_dyad_{dyad}_full_exp_config_{config["config_id"]}.png"))
    fullavgfigpath = path.abspath(path.join(hyperviz, f"CCORR_FRC_ent_quant_dyad_{dyad}_full_exp_trial_avg_config_{config["config_id"]}.png"))

    # Save the figure
    ffull.savefig(fullfigpath, bbox_inches="tight")
    ffull_avg.savefig(fullavgfigpath, bbox_inches="tight")