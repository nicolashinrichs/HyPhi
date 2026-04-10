# ============= #
# Preliminaries # 
# ============= # 

from Entropies import *
from FileIO import *
import sys
from tqdm import tqdm

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

makeDir(config["kuramoto_result_loc"])

# =================== #
# Load + Analyze Data # 
# =================== # 

num_nets = len(config["num_kuramotos"])

for num in tqdm(config["num_kuramotos"], desc="Kuramoto Simulations"):
    # Load condition network time series
    pklf = path.abspath(path.join(config["kuramoto_loc"], f"{num}_connectome_kuramoto.pkl"))
    Gt = loadNetworkPKL(pklf)

    # Allocate memory
    FRCvals = np.zeros((len(Gt), config["kuramoto_size"], config["kuramoto_size"]))

    # Compute Forman-Ricci curvatures across windows for this trial and frequency band
    FRCt = getFRCVec(Gt, method_val=cmethod)

    # Convert IBCs (networkx graphs) with window curvatures to window curvature matrices
    for window in tqdm(range(len(Gt)), desc="Timepoints"):
        FRCvals[window, :, :] = nx.attr_matrix(FRCt[window], edge_attr="formanCurvature")[0]

    # Get entropy
    Ht = vecEntropy(FRCt)

    # Get quantiles
    Qt = vecQuantiles(FRCt, qs=config["quantiles"])

    # Save data by condition

    # First, construct save paths
    FRCpath = path.abspath(path.join(config["kuramoto_result_loc"], f"Kuramoto_PLV_{cmethod}_FRC_matrix_cond_{num}_config_{config["config_id"]}.npy"))
    Hpath = path.abspath(path.join(config["kuramoto_result_loc"], f"Kuramoto_PLV_{cmethod}_FRC_entropy_cond_{num}_config_{config["config_id"]}.npy"))
    Qpath = path.abspath(path.join(config["kuramoto_result_loc"], f"Kuramoto_PLV_{cmethod}_FRC_quantiles_cond_{num}_config_{config["config_id"]}.npy"))

    # Now save the NPY files
    np.save(FRCpath, FRCvals)
    np.save(Hpath, Ht)
    np.save(Qpath, Qt)