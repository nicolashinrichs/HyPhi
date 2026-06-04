"""Plot per-trial AFRC entropy distributions from a hyper_ccorr_aug_frc.py run."""
# %% Import

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from hyphi.configs import config as hyphi_config
from hyphi.io import load_config
from plot_hyper_ccorr_frc import (
    params,
    plot_hyper_frc,
    plot_hyper_frc_full_exp,
    plot_hyper_frc_full_exp_avg_windows,
)
from tqdm import tqdm

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Load hyphi config
hyphi_config.init()

plt.rcParams.update(params)

# Load the configuration parameters into a dictionary
config = load_config(Path(hyphi_config.paths.experiments.configs, sys.argv[1]))

# Create map between dyads and dates
dyad_date_map = dict(zip(config["dyads"], config["dyad_dates"], strict=True))

# Create map between trial types and numeric identifiers
# Map to 0, 1, 2 instead of 1, 2, 3 for later
trial_type_ids = list(np.array(config["trial_type_ids"]) - 1)
trial_type_map = dict(zip(trial_type_ids, config["trial_types"], strict=True))

# Visualization path variables
hyperviz = Path(config["aug_viz_loc"]).absolute()
Path(hyperviz).mkdir(parents=True, exist_ok=True)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


if __name__ == "__main__":
    # Loop over dyads
    for dyad in tqdm(config["dyads"], desc="Dyads"):
        # Data path for shot times
        Spath = Path(config["behav_loc"], f"exp{dyad_date_map[dyad]}").absolute()

        # Load shot time data
        Svals = sp.io.loadmat(Spath)["trialtype"].flatten()

        # List to hold entropies and quantiles
        Hlist = []
        Qlist = []

        # Loop over trial types
        for trial_type in tqdm(config["trial_types"], desc="Trial Types"):
            # Data paths
            Hpath = Path(
                config["result_loc"],
                f"CCORR_aug_FRC_entropy_dyad_{dyad}_trial_type_{trial_type}_config_{config['config_id']}.npy",
            ).absolute()
            Qpath = Path(
                config["result_loc"],
                f"CCORR_aug_FRC_quantiles_dyad_{dyad}_trial_type_{trial_type}_config_{config['config_id']}.npy",
            ).absolute()

            # Load data
            Hvals = np.load(Hpath)
            Hlist.append(Hvals)
            Qvals = np.load(Qpath)
            Qlist.append(Qvals)

            # Plot
            f, _ = plot_hyper_frc(
                Hvals,
                Qvals,
                title=f"Dyad: {dyad}, Trial Type: {trial_type}",
                band_labels=config["freq_bands"],
                q_labels=config["quantiles"],
            )

            # Plot paths
            fig_path = (
                hyperviz
                / f"CCORR_aug_FRC_ent_quant_dyad_{dyad}_trial_type_{trial_type}_config_{config['config_id']}.png"
            )

            # Save the figure
            f.savefig(fig_path, bbox_inches="tight")

        # Plot full experiment
        ffull, _ = plot_hyper_frc_full_exp(
            Hlist,
            Qlist,
            Svals,
            title=f"Dyad: {dyad}",
            band_labels=config["freq_bands"],
            q_labels=config["quantiles"],
            type_labels=trial_type_map,
        )
        ffull_avg, _ = plot_hyper_frc_full_exp_avg_windows(
            Hlist,
            Qlist,
            Svals,
            title=f"Dyad: {dyad}",
            band_labels=config["freq_bands"],
            q_labels=config["quantiles"],
            type_labels=trial_type_map,
        )

        # Plot paths
        full_fig_path = hyperviz / f"CCORR_aug_FRC_ent_quant_dyad_{dyad}_full_exp_config_{config['config_id']}.png"
        full_avg_fig_path = (
            hyperviz / f"CCORR_aug_FRC_ent_quant_dyad_{dyad}_full_exp_trial_avg_config_{config['config_id']}.png"
        )

        # Save the figure
        ffull.savefig(full_fig_path, bbox_inches="tight")
        ffull_avg.savefig(full_avg_fig_path, bbox_inches="tight")


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
