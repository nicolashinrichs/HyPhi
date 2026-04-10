# ============= #
# Preliminaries # 
# ============= # 

from Entropies import *
from FileIO import *
import sys
from tqdm import tqdm
import scipy as sp
import multiprocessing
from concurrent.futures import ThreadPoolExecutor  
import threading  
from queue import Queue  

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

    ccorr_data[dyad] = {}
    
    # Construct file path for dyad
    dyad_file = path.abspath(path.join(config["data_loc"], f"CCORR_{dyad}.mat"))

    # Load dyad CCORR dictionary
    dyad_ccorr = sp.io.loadmat(dyad_file)

    # Throw away unused metadata, keep only CCORR matrices by trial type
    for trial_type in tqdm(config["trial_types"], desc="Trial Types"):

        # Extract the CCORR matrices for this dyad and trial type
        ccorr_mat = dyad_ccorr[f"CCORR_{trial_type}"]
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

        # Ollivier-Ricci curvature expects only positive weights, so we need to map the CCORR values
        # to positive values. There are a few ways to do this, but the one that makes the most sense
        # from the perspective of graph curvatures is to take the absolute CCORR values (r_ij --> |r_ij|)
        # The reasoning goes like this: graph curvature tries to quantify information flow between nodes
        # Two nodes that are anti-correlated (negative CCORR) still have high information flow between them
        # So for the graph curvature computation to see these nodes as still strongly connected, we need
        # to preserve the strength of the weights, while remaining agnostic to the sign of the weights

        ccorr_mat = np.abs(ccorr_mat)
        ccorr_data[dyad][trial_type] = ccorr_mat

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

# Function to parallelize over frequency bands
def freqBandAnalysis(freq_data, band_idx, config, progress_queue):

    # Store outputs
    output_data = {}

    # Loop over dyads
    for dyad in config["dyads"]:

        output_data[dyad] = {}

        # Loop over trial types
        for trial_type in config["trial_types"]:

            output_data[dyad][trial_type] = {}

            # Extract the CCORR matrices for this dyad and trial type
            ccorr_mat = freq_data[dyad][trial_type]

            # Loop over alpha hyperparameters
            for alph in config["orc_alphas"]:

                output_data[dyad][trial_type][alph] = {}

                # Loop over exponent power hyperparameters
                for powr in config["orc_powers"]:

                    output_data[dyad][trial_type][alph][powr] = {}

                    # Allocate memory
                    ORCvals = np.zeros((config["num_trials"], 
                                        config["num_windows"], 
                                        config["num_channels"], 
                                        config["num_channels"]))

                    Hvals = np.zeros((config["num_trials"], 
                                    config["num_windows"]))

                    Qvals = np.zeros((config["num_trials"], 
                                    config["num_windows"],
                                    len(config["quantiles"])))

                    # Loop over trials
                    for trial in range(config["num_trials"]):

                        # CCORR matrices for this trial, dyad, and trial type
                        ccorr_trial = ccorr_mat[trial, :, :, :]

                        # Convert CCORR matrices into Networkx graphs
                        Gt = [nx.from_numpy_array(ccorr_trial[window, :, :]) for window in range(config["num_windows"])]

                        # Loop over windows
                        for window in range(config["num_windows"]):

                            # Compute Forman-Ricci curvatures across windows for this trial and frequency band
                            ORC = getORC(Gt[window], alpha_val=alph, power_val=powr)

                            # Convert IBCs (networkx graphs) with window curvatures to window curvature matrices
                            ORCvals[trial, window, :, :] = nx.attr_matrix(ORC, edge_attr="ricciCurvature")[0]

                            # Get entropy
                            H = getEntropyKozachenko(ORC, curvature="ricciCurvature")
                            Hvals[trial, window] = H.copy()

                            # Get quantiles
                            Q = getQuantiles(ORC, qs=config["quantiles"], curvature="ricciCurvature")
                            Qvals[trial, window, :] = Q.copy()

                            # Send progress update  
                            progress_queue.put(1) 

                    # Store data
                    output_data[dyad][trial_type][alph][powr]["ORC"] = ORCvals
                    output_data[dyad][trial_type][alph][powr]["Entropy"] = Hvals
                    output_data[dyad][trial_type][alph][powr]["Quantiles"] = Qvals

                    # Save data by frequency band, dyad, trial type, alpha value, and power value

                    # First, construct save paths
                    ORCpath = path.abspath(path.join(config["result_loc"], f"CCORR_ORC_matrix_freq_{config["freq_bands"][band_idx]}_dyad_{dyad}_trial_type_{trial_type}_alpha_{alph}_power_{powr}_config_{config["config_id"]}.npy"))
                    Hpath = path.abspath(path.join(config["result_loc"], f"CCORR_ORC_entropy_freq_{config["freq_bands"][band_idx]}_dyad_{dyad}_trial_type_{trial_type}_alpha_{alph}_power_{powr}_config_{config["config_id"]}.npy"))
                    Qpath = path.abspath(path.join(config["result_loc"], f"CCORR_ORC_quantiles_freq_{config["freq_bands"][band_idx]}_dyad_{dyad}_trial_type_{trial_type}_alpha_{alph}_power_{powr}_config_{config["config_id"]}.npy"))

                    # Now save the NPY files
                    np.save(ORCpath, ORCvals)
                    np.save(Hpath, Hvals)
                    np.save(Qpath, Qvals)

    return output_data 


def progress_monitor(progress_queue, total_networks, pbar):  
    """Monitor progress updates and update tqdm bar"""  
    processed = 0  
    while processed < total_networks:  
        try:  
            count = progress_queue.get()  
            processed += count  
            pbar.update(count)  
        except:  
            break  


# Prepare the data by organizing by frequency band for parallelization
tqdm.write("Organizing data by frequency band for parallelization...")

# Loop over frequency bands
freq_data = []
for freq in tqdm(range(config["num_freqs"]), desc="Frequency Bands"):

    freq_org = {}

    # Loop over dyads
    for dyad in tqdm(config["dyads"], desc="Dyads"):

        freq_org[dyad] = {}

        # Loop over trial types
        for trial_type in tqdm(config["trial_types"], desc="Trial Types"):

            freq_org[dyad][trial_type] = ccorr_data[dyad][trial_type][freq, :, :, :, :]
    
    freq_data.append(freq_org)

# Get total number of dyad-trial type-trial-window-ORC hyperparameter combos 
total_networks = len(config["dyads"]) * len(config["trial_types"]) * config["num_trials"] * config["num_windows"] * len(config["orc_alphas"]) * len(config["orc_powers"])
tqdm.write(f"Total number of network-hyperparameter combos to be evaluated: {total_networks}")

# ORC computation parallelized over frequency bands
tqdm.write("Starting ORC computations...")

# Nested progress tracking
manager = multiprocessing.Manager()  
progress_queue = manager.Queue()  

# Inner progress bar for networks within bands  
with tqdm(total=len(freq_data) * total_networks, desc="Networks") as inner_pbar:  
        
    # Start progress monitor  
    monitor = threading.Thread(target=progress_monitor,   
                                args=(progress_queue, len(freq_data) * total_networks, inner_pbar))  
    monitor.start()  
        
    with ThreadPoolExecutor(max_workers=8) as executor:  
        # Arguments to pass in
        args = [(band_data, band_idx, config, progress_queue)   
                for band_idx, band_data in enumerate(freq_data)]  
            
        # This single tqdm tracks frequency band progress  
        band_results = list(tqdm(executor.map(lambda x: freqBandAnalysis(*x), args),  
                                total=len(args), desc="Processing frequency bands"))
            
    monitor.join()