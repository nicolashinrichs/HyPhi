"""TODO: add docstring"""

# %% Import
import os
import numpy as np
import networkx as nx
from hyphi.modeling.entropies import vec_entropy, vec_quantiles
from hyphi.io import load_config, make_dir, load_network_pkl
import sys
from tqdm import tqdm

from hyphi.configs import paths
from hyphi.modeling.graph_curvatures import compute_frc_vec

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Analysis configuration file
config_file = os.path.join(paths.experiments.configs, sys.argv[1])

# Load the configuration parameters into a dictionary
config = load_config(config_file)

# Type of curvature
curv_type = sys.argv[2]
assert curv_type in ["FRC", "AFRC"], f"Curvature type ({curv_type}) must be one of (FRC, AFRC)!"
if curv_type == "FRC":
    cmethod = "1d"
elif curv_type == "AFRC":
    cmethod = "augmented"

make_dir(config["kuramoto_result_loc"])

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    num_nets = len(config["num_kuramotos"])

    for num in tqdm(config["num_kuramotos"], desc="Kuramoto Simulations"):
        # Load condition network time series
        pklf = os.path.abspath(os.path.join(config["kuramoto_loc"], f"{num}_connectome_kuramoto.pkl"))
        Gt = load_network_pkl(pklf)

        # Allocate memory
        FRCvals = np.zeros((len(Gt), config["kuramoto_size"], config["kuramoto_size"]))

        # Compute Forman-Ricci curvatures across windows for this trial and frequency band
        FRCt = compute_frc_vec(Gt, method_val=cmethod)

        # Convert IBCs (networkx graphs) with window curvatures to window curvature matrices
        for window in tqdm(range(len(Gt)), desc="Timepoints"):
            FRCvals[window, :, :] = nx.attr_matrix(FRCt[window], edge_attr="formanCurvature")[0]

        # Get entropy
        Ht = vec_entropy(FRCt)

        # Get quantiles
        Qt = vec_quantiles(FRCt, qs=config["quantiles"])

        # Save data by condition

        # First, construct save paths
        FRCpath = os.path.abspath(
            os.path.join(
                config["kuramoto_result_loc"],
                f"Kuramoto_PLV_{cmethod}_FRC_matrix_cond_{num}_config_{config['config_id']}.npy",
            )
        )
        Hpath = os.path.abspath(
            os.path.join(
                config["kuramoto_result_loc"],
                f"Kuramoto_PLV_{cmethod}_FRC_entropy_cond_{num}_config_{config['config_id']}.npy",
            )
        )
        Qpath = os.path.abspath(
            os.path.join(
                config["kuramoto_result_loc"],
                f"Kuramoto_PLV_{cmethod}_FRC_quantiles_cond_{num}_config_{config['config_id']}.npy",
            )
        )

        # Now save the NPY files
        np.save(FRCpath, FRCvals)
        np.save(Hpath, Ht)
        np.save(Qpath, Qt)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
