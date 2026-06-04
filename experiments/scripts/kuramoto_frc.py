"""Run Kuramoto simulations and compute Forman-Ricci curvatures on the resulting PLV graph series."""

# %% Import
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from hyphi.configs import config as hyphi_config
from hyphi.io import load_config, load_network_pkl
from hyphi.modeling.entropies import vec_entropy, vec_quantiles
from hyphi.modeling.graph_curvatures import compute_frc_vec
from tqdm import tqdm

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Load hyphi config
hyphi_config.init()

# Load the configuration parameters into a dictionary
config = load_config(Path(hyphi_config.paths.experiments.configs, sys.argv[1]))

# Type of curvature
curv_type: str = sys.argv[2]

if curv_type not in ["FRC", "AFRC"]:
    raise ValueError("Curvature type ({curv_type}) must be one of (FRC, AFRC)!")

c_method = "1d" if curv_type == "FRC" else "augmented"

Path(config["kuramoto_result_loc"]).mkdir(exist_ok=True, parents=True)

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    num_nets = len(config["num_kuramotos"])

    for num in tqdm(config["num_kuramotos"], desc="Kuramoto Simulations"):
        # Load condition network time series
        Gt: nx.Graph = load_network_pkl(Path(config["kuramoto_loc"], f"{num}_connectome_kuramoto.pkl").absolute())

        # Allocate memory
        FRCvals = np.zeros((len(Gt), config["kuramoto_size"], config["kuramoto_size"]))

        # Compute Forman-Ricci curvatures across windows for this trial and frequency band
        FRCt = compute_frc_vec(list_of_graphs=Gt, method_val=c_method)

        # Convert IBCs (networkx graphs) with window curvatures to window curvature matrices
        for window in tqdm(range(len(Gt)), desc="Timepoints"):
            FRCvals[window, :, :] = nx.attr_matrix(FRCt[window], edge_attr="formanCurvature")[0]

        # Get entropy
        Ht = vec_entropy(FRCt)

        # Get quantiles
        Qt = vec_quantiles(FRCt, qs=config["quantiles"])

        # Save data by condition

        # First, construct save paths
        FRCpath = Path(
            config["kuramoto_result_loc"],
            f"Kuramoto_PLV_{c_method}_FRC_matrix_cond_{num}_config_{config['config_id']}.npy",
        ).absolute()
        Hpath = Path(
            config["kuramoto_result_loc"],
            f"Kuramoto_PLV_{c_method}_FRC_entropy_cond_{num}_config_{config['config_id']}.npy",
        ).absolute()
        Qpath = Path(
            config["kuramoto_result_loc"],
            f"Kuramoto_PLV_{c_method}_FRC_quantiles_cond_{num}_config_{config['config_id']}.npy",
        ).absolute()

        # Now save the NPY files
        np.save(FRCpath, FRCvals)
        np.save(Hpath, Ht)
        np.save(Qpath, Qt)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
