# ============= #
# Preliminaries # 
# ============= # 

from Entropies import *
from FileIO import *
import sys
from tqdm import tqdm
import scipy as sp

# Path variables
basepath = path.dirname(__file__)
configpath = path.abspath(path.join(basepath, "..", "experiments", "analysis"))

# ========================== #
# Load CCORR Data and Config # 
# ========================== # 

# Analysis configuration file
configfile = path.abspath(path.join(configpath, sys.argv[1]))

# Load the configuration parameters into a dictionary
config = loadConfig(configfile)

# Load the CCORR data tensors from .mat files
# We store data for each trial type separately
ccorr_data = {}
for dyad in config["dyads"]:
    # Construct file path for dyad
    dyad_file = path.abspath(path.join(config["data_loc"], f"CCORR_{dyad}.mat"))
    # Load dyad CCORR dictionary
    dyad_ccorr = sp.io.loadmat(dyad_file)
    # Throw away unused metadata, keep only CCORR matrices by trial type
    ccorr_data[dyad] = {trial_type: dyad_ccorr[f"CCORR_{trial_type}"] for trial_type in config["trial_types"]}

# If the results directory doesn't exist, make it
makeDir(path.abspath(config["result_loc"]))

# ========================= #
# Windowed CCORR Curvatures # 
# ========================= # 

# Data structures to hold curvatures + entropy and quantiles of curvatures
# We will do it like this: 
# CCORR tensors for each trial type have the following dimensions
# 8 [frequency bands] x 120 [4 consecutive windows * 30 trials] x (128 x 128) [EEG channel dyadic CCORR matrices]
# So we will compute Forman-Ricci curvatures for each window in each frequency band for each dyad
# Curvature arrays will be stored in files by Trial Type: Trial: Window: Freq: Curvatures
# Curvature entropies for each window in each frequency band for each dyad and trial type will be computed and stored in 
# 8 [frequency bands] x 30 [trials] x 4 [consecutive windows]
# Curvature quantiles for each window in each frequency band for each dyad and trial type will be computed and stored in 
# 8 [frequency bands] x 30 [trials] x 4 [consecutive windows] x 5 [quantiles]

# Loop over dyads
for dyad in tqdm(config["dyads"], desc="Dyads"):

    # Loop over trial types
    for trial_type in tqdm(config["trial_types"], desc="Trial Types"):

        # Extract the CCORR matrices for this dyad and trial type
        ccorr_mat = ccorr_data[dyad][trial_type]
        assert ccorr_mat.shape == (config["num_freqs"], 
                                   config["num_trials"]*config["num_windows"], 
                                   config["num_channels"],
                                   config["num_channels"]), f"CCORR tensor for dyad {dyad} does not match expected shape!"

        # Start by reshaping the CCORR matrices so they can be indexed by frequency band, trial, and window
        ccorr_mat = ccorr_mat.reshape(config["num_freqs"], 
                                      config["num_trials"], 
                                      config["num_windows"], 
                                      config["num_channels"], 
                                      config["num_channels"])

        # Forman-Ricci curvature expects only positive weights, so we need to map the CCORR values
        # to positive values. There are a few ways to do this, but the one that makes the most sense
        # from the perspective of graph curvatures is to take the absolute CCORR values (r_ij --> |r_ij|)
        # The reasoning goes like this: graph curvature tries to quantify information flow between nodes
        # Two nodes that are anti-correlated (negative CCORR) still have high information flow between them
        # So for the graph curvature computation to see these nodes as still strongly connected, we need
        # to preserve the strength of the weights, while remaining agnostic to the sign of the weights

        ccorr_mat = np.abs(ccorr_mat)

        # Allocate memory
        FRCvals = np.zeros((config["num_freqs"], 
                            config["num_trials"], 
                            config["num_windows"], 
                            config["num_channels"], 
                            config["num_channels"]))

        Hvals = np.zeros((config["num_freqs"], 
                        config["num_trials"], 
                        config["num_windows"]))

        Qvals = np.zeros((config["num_freqs"], 
                        config["num_trials"], 
                        config["num_windows"],
                        len(config["quantiles"])))

        # Loop over trials
        for trial in tqdm(range(config["num_trials"]), desc="Trials"):

            # Loop over frequency bands
            for freq in tqdm(range(config["num_freqs"]), desc="Frequency Bands"):

                # CCORR matrices for this trial and frequency band
                ccorr_trial = ccorr_mat[freq, trial, :, :, :]

                # Convert CCORR matrices into Networkx graphs
                Gt = [nx.from_numpy_array(ccorr_trial[window, :, :]) for window in range(4)]

                # Compute Forman-Ricci curvatures across windows for this trial and frequency band
                FRCt = getFRCVec(Gt)

                # Convert IBCs (networkx graphs) with window curvatures to window curvature matrices
                for window in range(4):
                   FRCvals[freq, trial, window, :, :] = nx.attr_matrix(FRCt[window], edge_attr="formanCurvature")[0]

                # Get entropy
                Ht = vecEntropy(FRCt)
                Hvals[freq, trial, :] = Ht.copy()

                # Get quantiles
                Qt = vecQuantiles(FRCt, qs=config["quantiles"])
                Qvals[freq, trial, :, :] = Qt.copy()

        # Save data by dyad and trial type

        # First, construct save paths
        FRCpath = path.abspath(path.join(config["result_loc"], f"CCORR_FRC_matrix_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Hpath = path.abspath(path.join(config["result_loc"], f"CCORR_FRC_entropy_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Qpath = path.abspath(path.join(config["result_loc"], f"CCORR_FRC_quantiles_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))

        # Now save the NPY files
        np.save(FRCpath, FRCvals)
        np.save(Hpath, Hvals)
        np.save(Qpath, Qvals)