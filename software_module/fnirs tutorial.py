## HyPhi x fNIRS
##
## This script implements a hybrid workflow:
## 1. Input: .csv files exported from LIONirs (Matlab) containing HbO data with NaNs.
## 2. Process: interpolates short gaps, fuses with VR/ECG.
## 3. Output: annotated geometric graphs for HyPhi analysis.


import numpy as np
import pandas as pd
import networkx as nx
from scipy.interpolate import CubicSpline
from scipy.stats import zscore


# ============================================================
# 1. Experimental Protocol (ETC Phase Logic)
# ============================================================

def get_etc_phase(timestamp, start_trigger_time=0):
    """
    Maps experiment time to phases: Free Exploration vs. Touch vs. ISI.
    """
    t = timestamp - start_trigger_time
    # Cumulative boundaries based on ETC_dual person_short.pptx
    boundaries = [60, 78, 138, 158, 218, 236, 296, 314, 374, 394, 454, 469]
    labels = [
        "Free_1", "ISI", "Touch_1", "ISI", "Touch_2", "ISI",
        "Touch_3", "ISI", "Touch_4", "ISI", "Free_2", "ISI"
    ]
    
    if t < 0: return "Pre-Experiment"
    for boundary, label in zip(boundaries, labels):
        if t <= boundary: return label
    return "Post-Experiment"


# ============================================================
# 2. LIONirs Import & Preprocessing
# ============================================================

def load_lionirs_export(filepath):
    """
    Loads the CSV exported from LIONirs/MATLAB.
    Assumes standard format: First column is Time, subsequent are Channels.
    """
    # Read CSV. 'na_values' ensures Matlab 'NaN' strings are read correctly as numpy.nan
    df = pd.read_csv(filepath, na_values=['NaN', 'nan'])
    
    # Rename first column to 'Time' if it isn't already
    if 'Time' not in df.columns:
        df.rename(columns={df.columns[0]: 'Time'}, inplace=True)
        
    # Set Time as index for easier slicing
    df.set_index('Time', inplace=True)
    return df

def imputation(series, sampling_rate=10, max_gap_sec=2.0):
    """
    The Bridge: Handles LIONirs 'NaN' gaps for geometric analysis.
    - Gaps < 2s: Interpolated (Cubic Spline) to preserve phase topology.
    - Gaps > 2s: Kept as NaN (window will be rejected later).
    """
    ts = series.copy()
    max_gap = int(max_gap_sec * sampling_rate)
    isnan = ts.isna()

    valid_mask = np.ones(len(ts), dtype=bool)
    groups = (isnan != isnan.shift()).cumsum()

    for _, idx in isnan.groupby(groups).groups.items():
        if isnan.loc[idx[0]]: # Is a NaN block
            if len(idx) <= max_gap:
                # Interpolate short gaps
                start_idx = idx[0] - 1
                end_idx = idx[-1] + 1
                # Bounds check
                if start_idx >= ts.index[0] and end_idx <= ts.index[-1]:
                    x = [start_idx, end_idx]
                    y = [ts.loc[start_idx], ts.loc[end_idx]]
                    cs = CubicSpline(x, y)
                    ts.loc[idx] = cs(idx)
            else:
                # Flag long gaps as invalid
                valid_mask[ts.index.get_indexer(idx)] = False

    return ts, valid_mask


def process_dyad_fnirs_from_csv(file_s1, file_s2, roi_list=None):
    """
    Full processing wrapper:
    1. Loads CSVs from LIONirs.
    2. Imputes short gaps.
    3. Z-scores data.
    4. Returns aligned DataFrame ready for Graph construction.
    """
    # 1. Load Data
    df_s1 = load_lionirs_export(file_s1)
    df_s2 = load_lionirs_export(file_s2)
    
    # Align time indices (find intersection of timestamps)
    common_index = df_s1.index.intersection(df_s2.index)
    df_s1 = df_s1.loc[common_index]
    df_s2 = df_s2.loc[common_index]

    if roi_list is None:
        # Auto-detect common channels (e.g., 'Ch1', 'HbO_Ch1')
        roi_list = [c for c in df_s1.columns if 'HbO' in c or 'Ch' in c]

    data = {}
    masks = []

    # 2. Process Each Channel
    for ch in roi_list:
        # Check if channel exists in both files
        if ch in df_s1.columns and ch in df_s2.columns:
            s1, m1 = imputation(df_s1[ch])
            s2, m2 = imputation(df_s2[ch])
            
            # 3. Z-Score (Normalize variance)
            data[f"S1_{ch}"] = zscore(s1, nan_policy="omit")
            data[f"S2_{ch}"] = zscore(s2, nan_policy="omit")
            
            masks.append(m1 & m2)
        else:
            print(f"Warning: Channel {ch} missing in one of the files.")

    global_mask = np.logical_and.reduce(masks)
    
    return pd.DataFrame(data, index=common_index), global_mask


# ============================================================
# 3. VR & ECG Fusion (Context)
# ============================================================

def process_vr_logs(vr_file, target_timestamps):
    """
    Reads VR logs, resamples to fNIRS time, extracts 'Hand Displacement' (Force).
    """
    df = pd.read_csv(vr_file)
    vr_out = pd.DataFrame(index=target_timestamps)
    
    cols_to_map = {
        "P1 hand displacement": "P1_HandDisp", 
        "P2 hand displacement": "P2_HandDisp",
        "P1 head rotation x": "P1_RotX",
        "P1 head rotation y": "P1_RotY", 
        "P1 head rotation z": "P1_RotZ"
    }
    
    for src, dst in cols_to_map.items():
        if src in df.columns:
            vr_out[dst] = np.interp(target_timestamps, df["Time"], df[src])
        else:
            vr_out[dst] = 0.0

    # Metric 1: Embodied Force (Mean Hand Displacement)
    vr_out["Dyad_HandDisp"] = vr_out[["P1_HandDisp", "P2_HandDisp"]].mean(axis=1)
    
    # Metric 2: Head Motion Energy (Artifact Control)
    # Calculate velocity magnitude
    vel = vr_out[["P1_RotX", "P1_RotY", "P1_RotZ"]].diff().fillna(0)
    vr_out["P1_HeadEnergy"] = np.linalg.norm(vel.values, axis=1)

    return vr_out


def process_ecg_physio(ecg_file, target_timestamps):
    """
    Aligns ECG PNS Index (Vagal Tone) to fNIRS time.
    """
    df = pd.read_csv(ecg_file)
    ecg_out = pd.DataFrame(index=target_timestamps)
    
    if "PNS index" in df.columns:
        ecg_out["PNS"] = np.interp(target_timestamps, df["Time"], df["PNS index"])
    
    return ecg_out


# ============================================================
# 4. Graph Construction (The Output)
# ============================================================

def build_annotated_graphs(fnirs_df, vr_df, ecg_df, start_time, window_size=150, step=10):
    graphs = []
    meta_list = []
    
    n = len(fnirs_df)
    
    for i in range(0, n - window_size, step):
        # Slice
        w_fnirs = fnirs_df.iloc[i:i+window_size]
        w_vr = vr_df.iloc[i:i+window_size]
        w_ecg = ecg_df.iloc[i:i+window_size]
        
        # Skip if too many NaNs (from LIONirs artifacts)
        if w_fnirs.isna().mean().mean() > 0.1: continue
        
        # Phase info
        t_mid = w_fnirs.index[len(w_fnirs)//2]
        phase = get_etc_phase(t_mid, start_time)
        
        # Build Graph
        corr = w_fnirs.corr().abs()
        np.fill_diagonal(corr.values, 0)
        G = nx.from_numpy_array(corr.values)
        
        # Meta
        meta = {
            "timestamp": t_mid,
            "phase": phase,
            "hand_disp": w_vr["Dyad_HandDisp"].mean(),
            "pns_index": w_ecg["PNS"].mean() if "PNS" in w_ecg else np.nan
        }
        
        graphs.append(G)
        meta_list.append(meta)
        
    return graphs, meta_list
