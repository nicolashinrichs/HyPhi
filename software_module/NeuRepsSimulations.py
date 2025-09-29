# ============= #
# Preliminaries # 
# ============= # 

from Entropies import *
from GraphSimulations import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.colors as mpc
from matplotlib.transforms import ScaledTranslation
from tqdm import tqdm
from os import path

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
vizpath = path.abspath(path.join(basepath, "..", "experiments", "figures"))
neurepsviz = path.abspath(path.join(vizpath, "NeuReps_2025_Figure.png"))

# ========================= #
# Weighted Small World Sims # 
# ========================= # 

# Array to hold entropy and quantiles of replications
Hreps = np.zeros((200, 100))
Qreps = np.zeros((200, 100, 5))

for n in tqdm(range(200)):
    # Generate weighted small world networks
    pt, Gt = genNeuRepsWSW(seed_val=n)

    # Compute Forman-Ricci curvatures
    FRCt = getFRCVec(Gt)

    # Get entropy
    Ht = vecEntropy(FRCt)
    Hreps[n, :] = Ht.copy()

    # Get quantiles
    qvals = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    Qt = vecQuantiles(FRCt, qs=qvals)
    Qreps[n, :, :] = Qt.copy()

# Get the quantiles over replications
rep_qvals = np.array([0.05, 0.5, 0.95])
HQs = np.quantile(Hreps, rep_qvals, axis=0)
QQs = np.quantile(Qreps, rep_qvals, axis=0)

# ======== #
# Plotting # 
# ======== # 

def NeuRepsFig():
    # Generate smaller graphs for plotting
    ptf, Gtf = genTVWeightedSW(100, 5, 1., 4, -4, 0)
    # Get edge weights
    weights = np.array([[G[u][v]['weight'] for u, v in G.edges()] for G in Gtf])
    # Normalize weights for width (e.g., scale to a desired range)
    max_weight = weights.flatten().max()
    min_weight = weights.flatten().min()
    # Normalize weights for color mapping (0 to 1)
    normalized_colors = np.array([[((w - min_weight) / (max_weight - min_weight)) for w in weight] for weight in weights])
    # Scale to 1-6
    normalized_widths = normalized_colors*5 + 1
    # Initialize figure
    fig = plt.figure(figsize=(19.2, 10.8), layout="constrained")
    # GridSpec layout: 2 rows, 4 columns
    gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], figure=fig)
    # Graph axes
    gaxs = [fig.add_subplot(gs[0, j]) for j in range(4)]
    glabs = ["A", "B", "C", "D"]
    for j in range(4):
        G = Gtf[j]
        # Layout for visualizing graphs on the ring
        pos = nx.circular_layout(G)
        # Draw the graph
        nx.draw(G, pos, ax=gaxs[j], node_size=20, width=normalized_widths[j], edge_color=normalized_colors[j], edge_cmap=cm.magma)
        # Label
        gaxs[j].set_title(fr"$p = {ptf[j]:.3f}$")
        # Use ScaledTranslation to put the label
        # - at the top left corner (axes fraction (0, 1)),
        # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
        # i.e. just outside the axes.
        gaxs[j].text(0.0, 1.0, glabs[j], transform=(
            gaxs[j].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=25, fontweight="bold", va='bottom')
    # Plot colorbar
    # Create a dedicated axis for the colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    # Define colormap and normalization
    gcmap = cm.magma
    # Define the range of values for the colorbar
    norm = mpc.Normalize(vmin=min_weight, vmax=max_weight) 
    # Create a ScalarMappable
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=gcmap)
    scalar_mappable.set_array([]) # Important: set_array to an empty array as there's no data directly linked
    # Add the colorbar to the dedicated axis
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax, orientation='vertical')
    cbar.set_label("Edge Weight", labelpad=20)
    # Plot entropy
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.semilogx(pt, HQs[1, :], c="C0")
    ax1.fill_between(pt, HQs[0, :], HQs[2, :], facecolor="C0", alpha=0.3)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_xlabel(r"$p(t)$")
    ax1.set_ylabel(r"$H_{FRC}(t)$")
    ax1.text(0.0, 1.0, "E", transform=(
            ax1.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize=25, fontweight="bold", va='bottom')
    # Take colors at regular intervals spanning the colormap
    cmap = cm.viridis
    qcolors = cmap(np.linspace(0, 1, 5))
    # Plot quantiles
    ax2 = fig.add_subplot(gs[1, 2:4])
    for j in range(5):
        ax2.semilogx(pt, QQs[1, :, j], c=qcolors[j])
        ax2.fill_between(pt, QQs[0, :, j], QQs[2, :, j], facecolor=qcolors[j], alpha=0.5)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.set_xlabel(r"$p(t)$")
    ax2.set_ylabel(r"$Q_{FRC}(t)$")
    ax2.text(0.0, 1.0, "F", transform=(
        ax2.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
        fontsize=25, fontweight="bold", va='bottom')
    # Plot quantile colorbar
    # Create a dedicated axis for the colorbar
    qcbar_ax = fig.add_subplot(gs[1, -1])
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
    qcbar.set_label("FRC Quantile", labelpad=20)
    # Optional: Set ticks and labels for the colorbar
    tick_locations = [0.5, 1.5, 2.5, 3.5, 4.5]
    qcbar.set_ticks(tick_locations)
    qcbar.set_ticklabels([f"{qvals[j]}" for j in range(5)])
    return fig


# Final plot
f = NeuRepsFig()
f.savefig(neurepsviz, bbox_inches="tight")