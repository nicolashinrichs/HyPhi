## HyPhi Example Code
##
## For all examples below, for the import statements to work,
## a script utilizing these functions needs to be located in the
## `software_module` directory of the HyPhi repo.
##
## This example illustrates a typical workflow for empirical
## hyperscanning data. The goal is to construct a time-varying
## sequence of weighted graphs and associated metadata from
## dual-fNIRS, VR kinematics, and ECG signals.
##
## All curvature- and entropy-based analyses are performed
## downstream using HyPhi (GraphCurvatures.py, Entropies.py).


# ============================================================
# Dual-fNIRS preprocessing and alignment
# ============================================================

import numpy as np
import pandas as pd
import networkx as nx
from scipy.interpolate import CubicSpline
from scipy.stats import zscore


def imputation(series, sampling_rate=10, max_gap_sec=2.0):
    """
    Interpolates NaN gaps shorter than max_gap_sec using cubic splines.
    Longer gaps are preserved as NaN and marked invalid (handling LIONirs output).
    """
    ts = series.copy()
    max_gap = int(max_gap_sec * sampling_rate)
    isnan = ts.isna()

    valid_mask = np.ones(len(ts), dtype=bool)
    # Identify groups of consecutive NaNs
    groups = (isnan != isnan.shift()).cumsum()

    for _, idx in isnan.groupby(groups).groups.items():
        if isnan.loc[idx[0]]: # If this group is NaN
            if len(idx) <= max_gap:
                # Interpolate if gap is short enough
                start_idx = idx[0] - 1
                end_idx = idx[-1] + 1
                
                # Check bounds
                if start_idx >= ts.index[0] and end_idx <= ts.index[-1]:
                     # Extract x and y for interpolation (indices as x)
                    x = [start_idx, end_idx]
                    y = [ts.loc[start_idx], ts.loc[end_idx]]
                    cs = CubicSpline(x, y)
                    ts.loc[idx] = cs(idx)
            else:
                # Mark as invalid if gap is too long
                valid_mask[ts.index.get_indexer(idx)] = False

    return ts, valid_mask


def process_dyad_fnirs(df_s1, df_s2, channels):
    """
    Applies identical preprocessing to both participants.
    Returns cleaned dyadic fNIRS data and a global validity mask.
    """
    data = {}
    masks = []

    for ch in channels:
        # Impute and get validity mask for each subject/channel
        s1, m1 = imputation(df_s1[ch])
        s2, m2 = imputation(df_s2[ch])

        # Normalize (z-score) to standardize variance before correlation
        data[f"S1_{ch}"] = zscore(s1, nan_policy="omit")
        data[f"S2_{ch}"] = zscore(s2, nan_policy="omit")

        masks.append(m1 & m2)

    # Intersection of all masks: timepoint is valid only if ALL channels are valid
    global_mask = np.logical_and.reduce(masks)
    
    # Return aligned DataFrame and the mask
    df_out = pd.DataFrame(data, index=df_s1.index)
    return df_out, global_mask


# ============================================================
# VR kinematics processing
# ============================================================

def process_kinematics(vr_file, target_timestamps):
    """
    Resample VR kinematics to the fNIRS timebase and compute
    head-motion energy as an external covariate.
    """
    df = pd.read_csv(vr_file)

    cols = [
        "P1 head rotation x", "P1 head rotation y", "P1 head rotation z",
        "P2 head rotation x", "P2 head rotation y", "P2 head rotation z",
        "P1 hand displacement", "P2 hand displacement"
    ]

    vr = pd.DataFrame(index=target_timestamps)

    # Linear interpolation to align VR frame rate to fNIRS sampling rate
    for c in cols:
        if c in df.columns:
            vr[c] = np.interp(target_timestamps, df["Time"], df[c])
        else:
            vr[c] = 0.0 # Fallback if column missing

    # Compute Head Motion Energy (Euclidean norm of velocity)
    for p in ["P1", "P2"]:
        rot_cols = [f"{p} head rotation {a}" for a in ["x", "y", "z"]]
        # Calculate velocity (diff)
        vel = vr[rot_cols].diff().fillna(0)
        # L2 Norm
        vr[f"{p}_HeadMotion"] = np.linalg.norm(vel.values, axis=1)

    return vr


# ============================================================
# ECG alignment
# ============================================================

def process_ecg(ecg_file, target_timestamps):
    """
    Align sec-by-sec ECG measures to the fNIRS timebase.
    ECG variables are treated as slow contextual fields.
    """
    df = pd.read_csv(ecg_file)

    ecg = pd.DataFrame(index=target_timestamps)
    # Key physiological metrics from "ECG data structure.pdf"
    for col in ["PNS index", "Stress index", "Mean HR"]:
        if col in df.columns:
            ecg[col] = np.interp(
                target_timestamps,
                df["Time"],
                df[col],
                left=np.nan,
                right=np.nan
            )
    return ecg


# ============================================================
# Constructing time-varying functional networks
# ============================================================

def build_graph_sequence(
    fnirs,
    vr,
    ecg,
    window_size=150, # e.g. 15 seconds at 10Hz
    step_size=10,    # e.g. 1 second step
    nan_thresh=0.1
):
    """
    Generates a time-ordered list of weighted networkx graphs
    and corresponding metadata dictionaries containing the
    fused ECG and VR annotations.
    
    These graphs are passed directly to HyPhi's
    getFRCVec and vecEntropy functions.
    """
    graphs = []
    meta = []

    n = len(fnirs)
    
    # Sliding window iteration
    for i in range(0, n - window_size, step_size):
        # 1. Slice Data
        w_fnirs = fnirs.iloc[i:i + window_size]
        w_vr = vr.iloc[i:i + window_size]
        w_ecg = ecg.iloc[i:i + window_size]

        # 2. Quality Control: Skip window if too many artifacts
        if w_fnirs.isna().mean().mean() > nan_thresh:
            continue

        # 3. Construct Weighted Graph (Functional Connectivity)
        # Using absolute Pearson correlation as edge weights
        corr = w_fnirs.corr().abs()
        np.fill_diagonal(corr.values, 0) # Remove self-loops

        G = nx.from_numpy_array(corr.values)
        graphs.append(G)

        # 4. Annotate with Context (Data Fusion)
        # Average the physiological/behavioral state over the window
        meta.append({
            "t_start": fnirs.index[i],
            "t_end": fnirs.index[i + window_size - 1],
            # Physio
            "mean_PNS": w_ecg["PNS index"].mean(),
            "mean_Stress": w_ecg["Stress index"].mean(),
            # Behavior (Force/Embodiment)
            "mean_hand_disp": w_vr.filter(like="hand displacement").mean().mean(),
            # Behavior (Artifact Control)
            "mean_head_motion": w_vr.filter(like="HeadMotion").mean().mean()
        })

    return graphs, meta


# ============================================================
# Downstream HyPhi geometry (illustrative only)
# ============================================================
#
# from GraphCurvatures import getFRCVec
# from Entropies import vecEntropy, getEntropyKozachenko
#
# # 1. Run the pipeline
# graphs, meta = build_graph_sequence(clean_fnirs, aligned_vr, aligned_ecg)
#
# # 2. Calculate Curvature (Augmented Forman-Ricci)
# # Returns a vector of edge curvatures for every edge in every graph
# FRCt = getFRCVec(graphs, method_val="augmented")
#
# # 3. Calculate Network Entropy (RNE)
# # Computes the entropy of the curvature distribution for each time window
# hKL = lambda G: getEntropyKozachenko(
#     G, curvature="formanCurvature", num_nn=4
# )
#
# Ht = vecEntropy(FRCt, estim=hKL)
#
# # 4. Integrate into Final DataFrame
# df_results = pd.DataFrame(meta)
# df_results['RNE'] = Ht
