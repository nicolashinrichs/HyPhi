## HyPhi Example Code
##
## For all examples below, for the import statements to work,
## a script utilizing these functions needs to be located in the
## `software_module` directory of the HyPhi repo.
##
## This script is tailored for the "Embodied Telepresence Connection" (ETC)
## project. It fuses 16-channel hyperscanning fNIRS (dlPFC, TPJ),
## VR Kinematics (pseudo-haptic force proxy), and ECG (PNS index).
##
## The output is a sequence of annotated graphs ready for HyPhi's
## `getFRCVec` (Curvature) and `vecEntropy` (RNE) functions.


import numpy as np
import pandas as pd
import networkx as nx
from scipy.interpolate import CubicSpline
from scipy.stats import zscore


# ============================================================
# 1. Experimental Protocol Definition (ETC Project)
# ============================================================

def get_etc_phase(timestamp, start_trigger_time=0):
    """
    Maps a timestamp to the ETC experimental phases based on the
    duration structure defined in the project slides.
    
    Structure:
    Free (60s) -> ISI (18s) -> Touch (60s) -> ISI (20s) -> 
    Touch (60s) -> ISI (18s) -> Touch (60s) -> ISI (18s) -> 
    Touch (60s) -> ISI (20s) -> Free (60s) -> ISI (15s)
    """
    t = timestamp - start_trigger_time
    
    # Cumulative durations (seconds)
    # [60, 78, 138, 158, 218, 236, 296, 314, 374, 394, 454, 469]
    boundaries = [60, 78, 138, 158, 218, 236, 296, 314, 374, 394, 454, 469]
    labels = [
        "Free_1", "ISI", "Touch_1", "ISI", "Touch_2", "ISI",
        "Touch_3", "ISI", "Touch_4", "ISI", "Free_2", "ISI"
    ]
    
    if t < 0: return "Pre-Experiment"
    
    for boundary, label in zip(boundaries, labels):
        if t <= boundary:
            return label
            
    return "Post-Experiment"


# ============================================================
# 2. Dual-fNIRS Preprocessing (LIONirs Compatibility)
# ============================================================

def imputation(series, sampling_rate=10, max_gap_sec=2.0):
    """
    Handles LIONirs artifact gaps (NaNs).
    Interpolates gaps < 2s (transient motion) using cubic splines.
    Leaves gaps > 2s as NaN (marked invalid) to prevent phase distortion.
    """
    ts = series.copy()
    max_gap = int(max_gap_sec * sampling_rate)
    isnan = ts.isna()

    valid_mask = np.ones(len(ts), dtype=bool)
    groups = (isnan != isnan.shift()).cumsum()

    for _, idx in isnan.groupby(groups).groups.items():
        if isnan.loc[idx[0]]: # Is a NaN block
            if len(idx) <= max_gap:
                # Interpolate short gaps to preserve topology
                start_idx = idx[0] - 1
                end_idx = idx[-1] + 1
                if start_idx >= ts.index[0] and end_idx <= ts.index[-1]:
                    x = [start_idx, end_idx]
                    y = [ts.loc[start_idx], ts.loc[end_idx]]
                    cs = CubicSpline(x, y)
                    ts.loc[idx] = cs(idx)
            else:
                # Mark long gaps as invalid
                valid_mask[ts.index.get_indexer(idx)] = False

    return ts, valid_mask


def process_dyad_fnirs(df_s1, df_s2, rois=None):
    """
    Inputs:
        df_s1, df_s2: DataFrames with 16 channels (S1_Ch1...S1_Ch16).
                      ROIs: dlPFC, TPJ (as noted in slides).
    Returns:
        Aligned Dyadic DataFrame, Global Validity Mask.
    """
    if rois is None:
        # Default to all 16 channels if not specified
        rois = df_s1.columns.tolist()

    data = {}
    masks = []

    for ch in rois:
        # Process S1
        s1, m1 = imputation(df_s1[ch])
        data[f"S1_{ch}"] = zscore(s1, nan_policy="omit")
        
        # Process S2
        s2, m2 = imputation(df_s2[ch])
        data[f"S2_{ch}"] = zscore(s2, nan_policy="omit")

        masks.append(m1 & m2)

    # Intersection mask: window is valid only if ALL ROIs are valid
    global_mask = np.logical_and.reduce(masks)
    
    return pd.DataFrame(data, index=df_s1.index), global_mask


# ============================================================
# 3. VR Kinematics (Force & Artifacts)
# ============================================================

def process_vr_logs(vr_file, target_timestamps):
    """
    Extracts two key metrics from VR logs:
    1. Motion Energy (Head Rotation) -> Regressor for artifacts.
    2. Hand Displacement -> Proxy for 'Force'/'Embodiment' (Pseudo-haptic effect).
    """
    df = pd.read_csv(vr_file)
    
    # Columns based on 'Kinematic data structure-2.pdf'
    head_rot_cols = [
        "P1 head rotation x", "P1 head rotation y", "P1 head rotation z",
        "P2 head rotation x", "P2 head rotation y", "P2 head rotation z"
    ]
    hand_disp_cols = ["P1 hand displacement", "P2 hand displacement"]
    
    vr_aligned = pd.DataFrame(index=target_timestamps)

    # A. Resample to fNIRS timebase (approx 10Hz)
    for c in head_rot_cols + hand_disp_cols:
        if c in df.columns:
            vr_aligned[c] = np.interp(target_timestamps, df["Time"], df[c])
        else:
            vr_aligned[c] = 0.0

    # B. Compute Metrics
    # 1. Head Motion Energy (Composite derivative of rotation)
    vel_p1 = vr_aligned[head_rot_cols[:3]].diff().fillna(0)
    vel_p2 = vr_aligned[head_rot_cols[3:]].diff().fillna(0)
    
    vr_aligned["P1_HeadEnergy"] = np.linalg.norm(vel_p1.values, axis=1)
    vr_aligned["P2_HeadEnergy"] = np.linalg.norm(vel_p2.values, axis=1)

    # 2. Hand Displacement (Raw value is already the metric)
    # Higher displacement = user pushing against virtual resistance = Stronger Illusion
    vr_aligned["Dyad_HandDisp"] = vr_aligned[hand_disp_cols].mean(axis=1)

    return vr_aligned


# ============================================================
# 4. ECG Alignment (Physiological Context)
# ============================================================

def process_ecg_physio(ecg_file, target_timestamps):
    """
    Aligns 'PNS index' and 'Stress index' (sec-by-sec) to fNIRS.
    These test the hypothesis: Does physiological regulation precede neural synchrony?
    """
    df = pd.read_csv(ecg_file)
    ecg_out = pd.DataFrame(index=target_timestamps)
    
    for col in ["PNS index", "Stress index"]:
        if col in df.columns:
            ecg_out[col] = np.interp(
                target_timestamps, df["Time"], df[col],
                left=np.nan, right=np.nan
            )
    return ecg_out


# ============================================================
# 5. Graph Construction Pipeline
# ============================================================

def build_annotated_graphs(
    fnirs_df, vr_df, ecg_df, 
    condition_type="Pseudo", # "Pseudo" or "No-Pseudo"
    start_trigger=0,
    window_size=150, # 15s window @ 10Hz
    step_size=10     # 1s step
):
    """
    Main Iterator.
    Generates weighted graphs + rich metadata for the Geometric Toolkit.
    """
    graphs = []
    metadata_list = []
    
    n_samples = len(fnirs_df)
    
    for i in range(0, n_samples - window_size, step_size):
        # Time Window Indices
        idx_start = i
        idx_end = i + window_size
        
        # 1. Check Data Quality (using imputation masks implicitly via NaNs)
        w_fnirs = fnirs_df.iloc[idx_start:idx_end]
        if w_fnirs.isna().mean().mean() > 0.1: 
            # Skip window if >10% data is missing/invalid
            continue
            
        # 2. Identify Experimental Phase
        # Uses the specific 60s/18s/20s structure from slides
        t_center = w_fnirs.index[len(w_fnirs)//2]
        phase_label = get_etc_phase(t_center, start_trigger)
        
        # Optional: Filter out 'ISI' if you only want to analyze active blocks
        # if "ISI" in phase_label: continue

        # 3. Build Connectivity Matrix (Weighted |Pearson|)
        # Standard approach for functional connectivity geometry
        corr = w_fnirs.corr().abs()
        np.fill_diagonal(corr.values, 0)
        G = nx.from_numpy_array(corr.values)
        
        # 4. Aggregate Context (Metadata)
        # This metadata allows correlating Geometry (RNE) with Behavior/Physio
        meta = {
            "timestamp": t_center,
            "condition": condition_type,     # Pseudo / No-Pseudo
            "phase": phase_label,            # Touch / Free / ISI
            
            # Hypothesis: High Hand Displacement (Force) -> High Embodiment
            "avg_hand_displacement": vr_df.iloc[idx_start:idx_end]["Dyad_HandDisp"].mean(),
            
            # Hypothesis: High PNS (Relaxation) -> Better Synchrony
            "avg_PNS_index": ecg_df.iloc[idx_start:idx_end]["PNS index"].mean(),
            
            # Control: Artifact check
            "avg_motion_energy": (
                vr_df.iloc[idx_start:idx_end]["P1_HeadEnergy"].mean() + 
                vr_df.iloc[idx_start:idx_end]["P2_HeadEnergy"].mean()
            ) / 2
        }
        
        graphs.append(G)
        metadata_list.append(meta)
        
    return graphs, metadata_list

# ============================================================
# Downstream Usage with HyPhi
# ============================================================
#
# # 1. Preprocess data streams
# fnirs_clean, mask = process_dyad_fnirs(raw_s1, raw_s2)
# vr_metrics = process_vr_logs("kinematics.csv", fnirs_clean.index)
# ecg_metrics = process_ecg_physio("ecg.csv", fnirs_clean.index)
#
# # 2. Build Graphs
# G_seq, Meta_seq = build_annotated_graphs(
#     fnirs_clean, vr_metrics, ecg_metrics, 
#     condition_type="Pseudo",
#     start_trigger=10.5 # Time when the first "Free" block started
# )
#
# # 3. HyPhi Geometry Calculation
# from GraphCurvatures import getFRCVec
# from Entropies import vecEntropy, getEntropyKozachenko
#
# # Calculate Augmented Forman-Ricci Curvature (AFRC)
# frc_vec = getFRCVec(G_seq, method_val="augmented")
#
# # Calculate Ricci Network Entropy (RNE)
# h_estimator = lambda G: getEntropyKozachenko(G, curvature="formanCurvature")
# rne_vec = vecEntropy(frc_vec, estim=h_estimator)
#
# # 4. Analysis
# # Combine RNE with Metadata to test: RNE ~ Condition * Phase + HandDisp
