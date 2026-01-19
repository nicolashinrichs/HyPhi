# ============= #
# Preliminaries # 
# ============= # 

from Entropies import *
from FileIO import *
import sys
from tqdm import tqdm
from scipy.stats import energy_distance
import dcor
from dcor import EstimationStatistic
from statsmodels.stats.multitest import multipletests
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

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

# If the pooled results directory doesn't exist, make it
makeDir(path.abspath(config["pooled_result_loc"]))

# ================================= #
# Aggregate Curvature Distributions # 
# ================================= # 

# Type of curvature
curv_type = sys.argv[2]
assert curv_type in ["FRC", "AFRC"], f"Curvature type ({curv_type}) must be one of (FRC, AFRC)!"

# Construct data path by curvature type
def dataPathConstructor(dyad, trial_type, curvature, config):
    if curvature == "FRC":
        return path.abspath(path.join(config["result_loc"], f"CCORR_FRC_matrix_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    elif curvature == "AFRC":
        return path.abspath(path.join(config["result_loc"], f"CCORR_aug_FRC_matrix_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))


# Construct data path by curvature type
def resultPathConstructor(dyad, trial_type, curvature, config, pooling):
    assert pooling in ["trial", "window"]
    if curvature == "FRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_{pooling}_pooling_matrix_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Hpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_{pooling}_pooling_entropy_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Qpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_{pooling}_pooling_quantiles_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    elif curvature == "AFRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_{pooling}_pooling_matrix_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Hpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_{pooling}_pooling_entropy_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Qpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_{pooling}_pooling_quantiles_dyad_{dyad}_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    return FRCpath, Hpath, Qpath


def dyadPooledPathConstructor(trial_type, curvature, config, pooling):
    assert pooling in ["trial", "window"]
    if curvature == "FRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_{pooling}_pooling_matrix_dyad_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Hpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_{pooling}_pooling_entropy_dyad_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Qpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_{pooling}_pooling_quantiles_dyad_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    elif curvature == "AFRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_{pooling}_pooling_matrix_dyad_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Hpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_{pooling}_pooling_entropy_dyad_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
        Qpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_{pooling}_pooling_quantiles_dyad_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    return FRCpath, Hpath, Qpath


def fullyPooledPathConstructor(trial_type, curvature, config):
    if curvature == "FRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_FRC_matrix_fully_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    elif curvature == "AFRC":
        FRCpath = path.abspath(path.join(config["pooled_result_loc"], f"CCORR_aug_FRC_matrix_fully_pooled_trial_type_{trial_type}_config_{config["config_id"]}.npy"))
    return FRCpath


# Load data
data = {
    d: {
        tt: np.load(dataPathConstructor(d, tt, curv_type, config))
        for tt in config["trial_types"]
    }
    for d in config["dyads"]
}

# # Containers for pooled data per dyad and trial type
# pooled_trials_per_dyad = {d: {} for d in dyads}        # (8, 4, 30*128*128)
# pooled_windows_per_dyad = {d: {} for d in dyads}       # (8, 30, 4*128*128)

# pooled_trials_per_dyad_entropy = {d: {tt: np.zeros((8, 4)) for tt in config["trial_types"]} for d in config["dyads"]}
# pooled_windows_per_dyad_entropy = {d: {tt: np.zeros((8, 30)) for tt in config["trial_types"]} for d in config["dyads"]}

# pooled_trials_per_dyad_quantiles = {d: {tt: np.zeros((8, 4, 5)) for tt in config["trial_types"]} for d in config["dyads"]}
# pooled_windows_per_dyad_quantiles = {d: {tt: np.zeros((8, 30, 5)) for tt in config["trial_types"]} for d in config["dyads"]}

# # ------------------------------------------------------------------
# # 1. Pool trials within each dyad × trial_type
# #    Result: (8, 4, 30 * 128 * 128)
# # ------------------------------------------------------------------
# for d in config["dyads"]:
#     for tt in config["trial_types"]:
#         # X has shape (8, 30, 4, 128, 128)
#         X = data[d][tt]

#         # Move axes so we have (freq, window, trial, elec, elec)
#         X_fwte = np.transpose(X, (0, 2, 1, 3, 4))

#         # Flatten trial and electrode dimensions: (8, 4, 30*128*128)
#         pooled_trials = X_fwte.reshape(8, 4, -1)

#         pooled_trials_per_dyad[d][tt] = pooled_trials

#         # Entropy and quantiles by frequency band
#         for f in range(8):
#             # By window
#             for w in range(4):
#                 # Kozachenko entropy
#                 pooled_trials_per_dyad_entropy[d][tt][f, w] = im.entropy(pooled_trials[f, w], approach="metric", k=4)
#                 pooled_trials_per_dyad_quantiles[d][tt][f, w] = np.quantile(pooled_trials[f, w], config["quantiles"])

#         # File paths
#         FRCpath, Hpath, Qpath = resultPathConstructor(d, tt, curv_type, config, "trial")

#         # Now save the NPY files
#         np.save(FRCpath, pooled_trials)
#         np.save(Hpath, pooled_trials_per_dyad_entropy[d][tt])
#         np.save(Qpath, pooled_trials_per_dyad_quantiles[d][tt])

# # ------------------------------------------------------------------
# # 2. Pool windows within each trial (per dyad × trial_type)
# #    Result: (8, 30, 4 * 128 * 128)
# # ------------------------------------------------------------------
# for d in dyads:
#     for tt in trial_types:
#         # X has shape (8, 30, 4, 128, 128)
#         X = data[d][tt]

#         # Shape is already (freq, trial, window, elec, elec)
#         # Flatten window and electrode dimensions: (8, 30, 4*128*128)
#         pooled_windows = X.reshape(8, 30, -1)

#         pooled_windows_per_dyad[d][tt] = pooled_windows

#         # Entropy and quantiles by frequency band
#         for f in range(8):
#             # By trial
#             for tr in range(30):
#                 # Kozachenko entropy
#                 pooled_windows_per_dyad_entropy[d][tt][f, tr] = im.entropy(pooled_windows[f, tr], approach="metric", k=4)
#                 pooled_windows_per_dyad_quantiles[d][tt][f, tr] = np.quantile(pooled_windows[f, tr], config["quantiles"])

#         # File paths
#         FRCpath, Hpath, Qpath = resultPathConstructor(d, tt, curv_type, config, "window")

#         # Now save the NPY files
#         np.save(FRCpath, pooled_windows)
#         np.save(Hpath, pooled_windows_per_dyad_entropy[d][tt])
#         np.save(Qpath, pooled_windows_per_dyad_quantiles[d][tt])

# # ------------------------------------------------------------------
# # 3. Pool across dyads, keeping trial type distinct
# #
# # Trial-pooling:  (8, 4, 10 * 30 * 128 * 128)
# # Window-pooling: (8, 30, 10 * 4 * 128 * 128)
# # ------------------------------------------------------------------

# trial_pooled_across_dyads = {}   # one array per trial type
# window_pooled_across_dyads = {}  # one array per trial type

# pooled_trials_entropy = {tt: np.zeros((8, 4)) for tt in config["trial_types"]}
# pooled_windows_entropy = {tt: np.zeros((8, 30)) for tt in config["trial_types"]}

# pooled_trials_quantiles = {tt: np.zeros((8, 4, 5)) for tt in config["trial_types"]}
# pooled_windows_quantiles = {tt: np.zeros((8, 30, 5)) for tt in config["trial_types"]}

# for tt in trial_types:
#     # Collect per-dyad arrays for this trial type
#     # trial-pooled: each is (8, 4, 30*128*128)
#     trial_list = [pooled_trials_per_dyad[d][tt] for d in dyads]
#     # Stack on a new dyad axis: (10, 8, 4, 30*128*128)
#     trial_stack = np.stack(trial_list, axis=0)
#     # Merge dyad into last dimension: (8, 4, 10*30*128*128)
#     trial_merged = np.reshape(
#         np.transpose(trial_stack, (1, 2, 0, 3)),  # (8, 4, 10, 30*128*128)
#         (8, 4, -1)
#     )
#     trial_pooled_across_dyads[tt] = trial_merged

#     # window-pooled: each is (8, 30, 4*128*128)
#     window_list = [pooled_windows_per_dyad[d][tt] for d in dyads]
#     # Stack: (10, 8, 30, 4*128*128)
#     window_stack = np.stack(window_list, axis=0)
#     # Merge dyad into last dimension: (8, 30, 10*4*128*128)
#     window_merged = np.reshape(
#         np.transpose(window_stack, (1, 2, 0, 3)),  # (8, 30, 10, 4*128*128)
#         (8, 30, -1)
#     )
#     window_pooled_across_dyads[tt] = window_merged

#     # Entropy and quantiles by frequency band
#     for f in range(8):
#         # By window
#         for w in range(4):
#             # Kozachenko entropy
#             pooled_trials_entropy[tt][f, w] = im.entropy(trial_merged[f, w, :], approach="metric", k=4)
#             pooled_trials_quantiles[tt][f, w] = np.quantile(trial_merged[f, w, :], config["quantiles"])

#         # By trial
#         for tr in range(30):
#             # Kozachenko entropy
#             pooled_windows_entropy[tt][f, tr] = im.entropy(window_merged[f, tr, :], approach="metric", k=4)
#             pooled_windows_quantiles[tt][f, tr] = np.quantile(window_merged[f, tr, :], config["quantiles"])

#     # File paths
#     FRCpath, Hpath, Qpath = resultPathConstructor(d, tt, curv_type, config, "window")

#     # Now save the NPY files
#     np.save(FRCpath, pooled_windows)
#     np.save(Hpath, pooled_windows_per_dyad_entropy[d][tt])
#     np.save(Qpath, pooled_windows_per_dyad_quantiles[d][tt])

# ------------------------------------------------------------------
# 4. Pool across ALL dimensions (dyads × trials × windows) per trial type
#    Result: (8, 10 * 30 * 4 * 128 * 128) for each trial type
# ------------------------------------------------------------------
fully_pooled_across_everything = {}

for tt in config["trial_types"]:
    # Start from original data across all dyads
    all_dyad_data = [data[d][tt] for d in config["dyads"]]  # list of (8,30,4,128,128)
    
    # Stack dyads: (10, 8, 30, 4, 128, 128)
    stacked = np.stack(all_dyad_data, axis=0)
    
    # Reorder to (8, 10, 30, 4, 128, 128)
    reordered = np.transpose(stacked, (1, 0, 2, 3, 4, 5))
    
    # Flatten all dimensions after freq: (8, 10*30*4*128*128)
    fully_pooled = reordered.reshape(8, -1)
    
    fully_pooled_across_everything[tt] = fully_pooled

    # Save data 
    FRCpath = fullyPooledPathConstructor(tt, curv_type, config)
    np.save(FRCpath, fully_pooled)

# ------------------------------------------------------------------
# At this point:
#   pooled_trials_per_dyad[d][tt]        -> (8, 4, 30*128*128)
#   pooled_windows_per_dyad[d][tt]       -> (8, 30, 4*128*128)
#   trial_pooled_across_dyads[tt]        -> (8, 4, 10*30*128*128)
#   window_pooled_across_dyads[tt]       -> (8, 30, 10*4*128*128)
# ------------------------------------------------------------------

# Example sanity checks (use assertions in real code)
# example_tt = trial_types[0]
# print("Per-dyad trial-pooled shape:", pooled_trials_per_dyad[dyads[0]][example_tt].shape)
# print("Per-dyad window-pooled shape:", pooled_windows_per_dyad[dyads[0]][example_tt].shape)
# print("Across-dyad trial-pooled shape:", trial_pooled_across_dyads[example_tt].shape)
# print("Across-dyad window-pooled shape:", window_pooled_across_dyads[example_tt].shape)
# print("Fully pooled shape (all dyads/trials/windows):", 
#       fully_pooled_across_everything[example_tt].shape)
# # Expected: (8, 10*30*4*128*128) = (8, 19660800)

# ================== #
# Hypothesis Testing # 
# ================== # 

# Assume fully_pooled_across_everything from previous script
# Shape: fully_pooled_across_everything[tt][freq, flattened_curvs]
# flattened_curvs has shape (10*30*4*128*128,)

def single_permutation_perm(all_c, split_sizes, test_statistic):
    """Single permutation computation (pure function for parallelization)"""
    np.random.shuffle(all_c)
    perm_curvs = np.split(all_c, np.cumsum(split_sizes)[:-1])
    return test_statistic(perm_curvs)


def fast_energy_test_parallel(curvs, n_perm=1000, sample_size=50000, n_jobs=-1):
    """Parallelized energy distance test"""
    # Downsample first
    sampled_curvs = [c[np.random.choice(len(c), min(sample_size, len(c)), replace=False)] 
                    for c in curvs]
    
    def test_statistic(data_list):
        dists = [energy_distance(data_list[i], data_list[j]) 
                for i in range(len(data_list)) for j in range(i+1, len(data_list))]
        return np.mean(dists)
    
    obs_stat = test_statistic(sampled_curvs)
    
    # Parallel permutations
    all_c = np.concatenate(sampled_curvs)
    split_sizes = [len(c) for c in sampled_curvs]
    
    perm_stats = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(single_permutation_perm)(all_c.copy(), split_sizes, test_statistic) 
        for _ in range(n_perm)
    )
    
    p_value = np.mean(np.array(perm_stats) >= obs_stat)
    return {'observed_statistic': obs_stat, 'p_value': p_value}


# Updated main functions
def test_trial_types_per_freq(fully_pooled, trial_types, freq_bands, 
                            n_perm=1000, sample_size=50000, n_jobs=-1):
    results = []
    for f in tqdm(range(len(freq_bands)), desc="Frequency Bands, Omnibus"):
        curvs = [fully_pooled[tt][f, :] for tt in trial_types]
        r = fast_energy_test_parallel(curvs, n_perm, sample_size, n_jobs)
        results.append({
            'frequency_band': freq_bands[f],
            'observed_statistic': r['observed_statistic'],
            'p_value': r['p_value'],
            'significant': r['p_value'] < 0.05
        })
    return results

# def test_trial_types_per_freq(fully_pooled, trial_types, freq_bands, n_perm=10000, alpha=0.05):
#     results = []
    
#     for f in tqdm(range(len(freq_bands)), desc="Frequency Bands, Omnibus"):  # frequency bands
#         # Extract curvatures for all 3 trial types at this freq
#         curvs = [fully_pooled[tt][f, :] for tt in trial_types]
        
#         # Test statistic: mean pairwise energy distance between trial types
#         def test_statistic(data_list):
#             dists = []
#             for i in range(len(data_list)):
#                 for j in range(i+1, len(data_list)):
#                     d = energy_distance(data_list[i], data_list[j])
#                     dists.append(d)
#             return np.mean(dists)
        
#         obs_stat = test_statistic(curvs)
        
#         # Permutation test: shuffle trial type labels across all data
#         all_c = np.concatenate(curvs)
#         perm_stats = []
        
#         for _ in tqdm(range(n_perm), desc="Permutations, Omnibus"):
#             np.random.shuffle(all_c)  # permutes across trial types
#             perm_curvs = np.split(all_c, np.cumsum([len(c) for c in curvs])[:-1])
#             perm_stat = test_statistic(perm_curvs)
#             perm_stats.append(perm_stat)
        
#         p_value = np.mean(np.array(perm_stats) >= obs_stat)
#         significant = p_value < alpha
        
#         results.append({
#             'frequency_band': freq_bands[f],
#             'observed_statistic': obs_stat,
#             'p_value': p_value,
#             'significant': significant
#         })
    
#     return results

# def test_trial_types_per_freq(fully_pooled, trial_types, freq_bands, n_perm=10000, alpha=0.05):
#     results = []
    
#     for f in tqdm(range(len(freq_bands)), desc="Frequency Bands, Omnibus"):
#         # Extract curvatures for all 3 trial types at this freq
#         curvs = [fully_pooled[tt][f, :] for tt in trial_types]
        
#         # Use dcor's energy_test for omnibus test (k>=2 samples)
#         test_result = dcor.homogeneity.energy_test(
#             *curvs, 
#             num_resamples=n_perm,
#             exponent=1.0,  # Euclidean distance,
#             estimation_stat=EstimationStatistic.U_STATISTIC,
#             n_jobs=8
#         )
        
#         # Extract results (matches your original format)
#         obs_stat = float(test_result.statistic)
#         p_value = float(test_result.pvalue)
#         significant = p_value < alpha
        
#         results.append({
#             'frequency_band': freq_bands[f],
#             'observed_statistic': obs_stat,
#             'p_value': p_value,
#             'significant': significant
#         })
    
#     return results


# Pairwise tests
# def pairwise_energy_tests(fully_pooled, trial_types, freq_bands, n_perm=10000, alpha=0.05):
#     """
#     Run 3 pairwise energy distance tests per frequency band (24 total tests).
#     Returns raw p-values for Holm correction across all 24.
#     """
#     results = []

#     # Define 3 pairwise comparisons
#     pairs = [(0,1), (0,2), (1,2)]
#     pair_labels = [f"{trial_types[p[0]]}-{trial_types[p[1]]}" for p in pairs]

#     # Loop over frequency bands
#     for f in tqdm(range(len(freq_bands)), desc="Frequency Bands, Pairwise"):  # frequency bands
#         # Extract curvatures for all 3 trial types at this freq
#         curvs = [fully_pooled[tt][f, :] for tt in trial_types]
        
#         for (i,j), label in tqdm(zip(pairs, pair_labels), desc="Pairs"):
#             # Observed energy distance
#             obs_d = energy_distance(curvs[i], curvs[j])
            
#             # Permutation test
#             all_c_pair = np.concatenate([curvs[i], curvs[j]])
#             perm_ds = []
            
#             for _ in tqdm(range(n_perm), desc="Permutations, Pairwise"):
#                 np.random.shuffle(all_c_pair)
#                 perm_d = energy_distance(all_c_pair[:len(curvs[i])], 
#                                        all_c_pair[len(curvs[i]):])
#                 perm_ds.append(perm_d)
            
#             p_val = np.mean(np.array(perm_ds) >= obs_d)
#             significant = p_val < alpha

#             results.append({
#             'pair': label,
#             'frequency_band': freq_bands[f],
#             'observed_statistic': obs_d,
#             'p_value': p_val,
#             'significant': significant
#         })
    
#     return results

# def pairwise_energy_tests(fully_pooled, trial_types, freq_bands, n_perm=10000, alpha=0.05):
#     """
#     Run 3 pairwise energy distance tests per frequency band (24 total tests) 
#     using dcor.energy_test. Returns same format as original.
#     """
#     results = []

#     # Define 3 pairwise comparisons
#     pairs = [(0,1), (0,2), (1,2)]
#     pair_labels = [f"{trial_types[p[0]]}-{trial_types[p[1]]}" for p in pairs]

#     # Loop over frequency bands
#     for f in tqdm(range(len(freq_bands)), desc="Frequency Bands, Pairwise"):
#         # Extract curvatures for all 3 trial types at this freq
#         curvs = [fully_pooled[tt][f, :] for tt in trial_types]
        
#         for (i,j), label in tqdm(zip(pairs, pair_labels), desc="Pairs"):
#             # Use dcor's energy_test for 2-sample test
#             test_result = dcor.homogeneity.energy_test(
#                 curvs[i], curvs[j], 
#                 num_resamples=n_perm,
#                 exponent=1.0,  # Euclidean distance
#                 estimation_stat=EstimationStatistic.U_STATISTIC,
#                 n_jobs=8
#             )
            
#             # Extract results (matches your exact output format)
#             obs_d = float(test_result.statistic)
#             p_val = float(test_result.pvalue)
#             significant = p_val < alpha

#             results.append({
#                 'pair': label,
#                 'frequency_band': freq_bands[f],
#                 'observed_statistic': obs_d,
#                 'p_value': p_val,
#                 'significant': significant
#             })
    
#     return results

def pairwise_energy_tests(fully_pooled, trial_types, freq_bands, 
                         n_perm=1000, sample_size=50000, n_jobs=-1):
    """
    24 pairwise tests with parallelization + single tqdm progress bar
    """
    def single_pair_test(f, i, j):
        # Downsample for speed
        curvs = [fully_pooled[tt][f, np.random.choice(len(fully_pooled[tt][f]), 
                                                     sample_size, replace=False)] 
                for tt in trial_types]
        
        obs_d = energy_distance(curvs[i], curvs[j])
        
        # Parallel permutations for this pair
        all_c_pair = np.concatenate([curvs[i], curvs[j]])
        na = len(curvs[i])
        
        perm_ds = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(lambda x: energy_distance(x[:na], x[na:]))(
                np.random.permutation(all_c_pair)
            ) for _ in range(n_perm)
        )
        
        p_val = np.mean(np.array(perm_ds) >= obs_d)
        return {
            'frequency_band': freq_bands[f],
            'pair': f"{trial_types[i]}-{trial_types[j]}",
            'observed_statistic': float(obs_d),
            'p_value': float(p_val),
            'significant': p_val < 0.05
        }
    
    # All 24 test combinations (8 freq × 3 pairs)
    pairs = [(0,1), (0,2), (1,2)]
    all_tests = [(f, i, j) for f in range(len(freq_bands)) for i,j in pairs]
    
    # Single progress bar for ALL 24 tests!
    with tqdm_joblib(desc="Pairwise Tests (24 total)", total=len(all_tests)):
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(single_pair_test)(*test) for test in all_tests
        )
    
    return results


# Run omnibus analysis
results = test_trial_types_per_freq(fully_pooled_across_everything, config["trial_types"], config["freq_bands"], n_perm=config["nhst_perm"], sample_size=config["nhst_subsample"])
print("Uncorrected omnibus results:")
for r in results:
    print(f"Freq {r['frequency_band']}: stat={r['observed_statistic']:.4f}, p={r['p_value']:.4f}, sig={r['significant']}")

# Correct for multiple testing while maintaining good statistical power

# Get your 8 p-values from original per-frequency tests
p_values = [r['p_value'] for r in results]

# Holm-Bonferroni (step-down)
rejected, p_corrected, _, _ = multipletests(p_values, 
                                            alpha=0.05, 
                                            method='holm')

print("Omnibus results after correcting for multiple comparisons:")
for f, (reject, p_corr) in enumerate(zip(rejected, p_corrected)):
    print(f"Freq {results[f]['frequency_band']}: p={p_values[f]:.4f} → p_corr={p_corr:.4f}, sig={reject}")

# Run pairwise tests
results_pair = pairwise_energy_tests(fully_pooled_across_everything, config["trial_types"], config["freq_bands"], n_perm=config["nhst_perm"], sample_size=config["nhst_subsample"])
print("Uncorrected pairwise results:")
for r in results_pair:
    print(f"Freq {r['frequency_band']}, Pair {r['pair']}: stat={r['observed_statistic']:.4f}, p={r['p_value']:.4f}, sig={r['significant']}")

# Correct for multiple testing while maintaining good statistical power

# Get your 3 p-values from each original per-frequency tests, 24 in total
p_values_pair = [r['p_value'] for r in results_pair]

# Holm-Bonferroni (step-down)
rejected_pair, p_corrected_pair, _, _ = multipletests(p_values_pair, 
                                                      alpha=0.05, 
                                                      method='holm')

print("Pairwise results after correcting for multiple comparisons:")
for f, (reject, p_corr) in enumerate(zip(rejected_pair, p_corrected_pair)):
    print(f"Freq {results_pair[f]['frequency_band']}, Pair {results_pair[f]['pair']}: p={p_values_pair[f]:.4f} → p_corr={p_corr:.4f}, sig={reject}")

# Summary
significant_pairs = sum(rejected_pair)
print(f"\n{significant_pairs}/24 pairwise tests significant (Holm-corrected)")

# Function to save results
def save_hypothesis_test_results_json(
    results_omnibus,
    rejected_omnibus,
    p_corrected_omnibus,
    results_pair,
    rejected_pair,
    p_corrected_pair,
    config,
    json_path
):
    """
    Save omnibus and pairwise test results (with Holm-corrected p-values) to JSON.

    Parameters
    ----------
    results_omnibus : list of dict
        Output of test_trial_types_per_freq, e.g.
        [
          {
            'frequency_band': 'theta',
            'observed_statistic': ...,
            'p_value': ...,
            'significant': ...
          },
          ...
        ]
    rejected_omnibus : array-like of bool
        Holm-rejected flags for each omnibus test (len = n_freq_bands).
    p_corrected_omnibus : array-like of float
        Holm-corrected p-values for omnibus tests.

    results_pair : list of dict
        Output of pairwise_energy_tests, e.g.
        [
          {
            'frequency_band': 'theta',
            'pair': 'type1-type2',
            'observed_statistic': ...,
            'p_value': ...,
            'significant': ...
          },
          ...
        ]
    rejected_pair : array-like of bool
        Holm-rejected flags for each pairwise test (len = n_freq_bands * 3).
    p_corrected_pair : array-like of float
        Holm-corrected p-values for pairwise tests.

    config : dict
        Config with at least "trial_types" and "freq_bands".
    json_path : str
        Path to output JSON file.
    """

    # Build omnibus results keyed by frequency band
    omnibus_out = {}
    for i, r in enumerate(results_omnibus):
        band = r["frequency_band"]
        omnibus_out[band] = {
            "observed_statistic": float(r["observed_statistic"]),
            "p_raw": float(r["p_value"]),
            "p_holm": float(p_corrected_omnibus[i]),
            "significant_raw": bool(r.get("significant", False)),
            "significant_holm": bool(rejected_omnibus[i]),
        }

    # Build pairwise results keyed by frequency band → pair
    pairwise_out = {}
    for i, r in enumerate(results_pair):
        band = r["frequency_band"]
        pair = r["pair"]
        if band not in pairwise_out:
            pairwise_out[band] = {}
        pairwise_out[band][pair] = {
            "observed_statistic": float(r["observed_statistic"]),
            "p_raw": float(r["p_value"]),
            "p_holm": float(p_corrected_pair[i]),
            "significant_raw": bool(r.get("significant", False)),
            "significant_holm": bool(rejected_pair[i]),
        }

    output = {
        "description": "Energy-distance omnibus and pairwise tests with Holm correction",
        "alpha": 0.05,
        "trial_types": config["trial_types"],
        "freq_bands": config["freq_bands"],
        "omnibus": omnibus_out,
        "pairwise": pairwise_out,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


# Save the results
fullpoolstats = path.abspath(path.join(configpath, f"fully_pooled_{curv_type}_energy_stat_trial_types_n_perm_{config["nhst_perm"]}_ sample_size_{config["nhst_subsample"]}.json"))

_ = save_hypothesis_test_results_json(
    results_omnibus=results,
    rejected_omnibus=rejected,
    p_corrected_omnibus=p_corrected,
    results_pair=results_pair,
    rejected_pair=rejected_pair,
    p_corrected_pair=p_corrected_pair,
    config=config,
    json_path=fullpoolstats
)