## HyPhi for fNIRS x ECG x VR
##
## This module implements a hybrid "Systemic Physiology Augmented" (SPA)
## pipeline to generate weighted, annotated connectivity graphs for
## the Embodied Telepresence Connection (ETC) project.
##
## It integrates:
## 1. LIONirs fNIRS data (artifact-corrected)
## 2. 90Hz VR Kinematics (features extracted before downsampling)
## 3. ECG Physiology (aligned with hemodynamic lag)
##
## OUTPUT: A dictionary containing lists of NetworkX graphs ready for
##         Forman-Ricci Curvature (FRC) and Network Entropy (RNE) analysis.

import numpy as np
import pandas as pd
import networkx as nx
from scipy.interpolate import CubicSpline
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. Experimental Protocol Logic
# ============================================================

def get_etc_phase(timestamp, start_trigger_time=0):
    """
    Maps experiment timestamp to specific phases: 
    Free Exploration (60s) -> Touch (60s) -> ISI (Variable).
    """
    t = timestamp - start_trigger_time
    # Boundaries based on ETC_dual person_short.pptx
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
# 2. Data Ingestion & Imputation
# ============================================================

def load_lionirs_export(filepath):
    """Parses LIONirs export format."""
    df = pd.read_csv(filepath, na_values=['NaN', 'nan'])
    if 'Time' not in df.columns:
        df.rename(columns={df.columns[0]: 'Time'}, inplace=True)
    df.set_index('Time', inplace=True)
    return df

def imputation(series, sampling_rate=10, max_gap_sec=2.0):
    """
    Intelligent Imputation for LIONirs Artifacts.
    - Short gaps (<2s): Interpolated to preserve topological shape.
    - Long gaps (>2s): Left as NaN to trigger window rejection.
    """
    ts = series.copy()
    max_gap = int(max_gap_sec * sampling_rate)
    isnan = ts.isna()

    valid_mask = np.ones(len(ts), dtype=bool)
    groups = (isnan != isnan.shift()).cumsum()

    for _, idx in isnan.groupby(groups).groups.items():
        if isnan.loc[idx[0]]:
            if len(idx) <= max_gap:
                start = idx[0] - 1
                end = idx[-1] + 1
                if start >= ts.index[0] and end <= ts.index[-1]:
                    x = [start, end]
                    y = [ts.loc[start], ts.loc[end]]
                    cs = CubicSpline(x, y)
                    ts.loc[idx] = cs(idx)
            else:
                valid_mask[ts.index.get_indexer(idx)] = False
    return ts, valid_mask

def process_dyad_fnirs_raw(file_s1, file_s2):
    """Loads raw Dyad data, aligns timestamps, and imputes."""
    df_s1 = load_lionirs_export(file_s1)
    df_s2 = load_lionirs_export(file_s2)
    
    common = df_s1.index.intersection(df_s2.index)
    df_s1, df_s2 = df_s1.loc[common], df_s2.loc[common]
    
    # Auto-detect channels
    rois = [c for c in df_s1.columns if 'HbO' in c or 'Ch' in c]
    
    data = {}
    for ch in rois:
        if ch in df_s2.columns:
            s1, _ = imputation(df_s1[ch])
            s2, _ = imputation(df_s2[ch])
            data[f"S1_{ch}"] = s1
            data[f"S2_{ch}"] = s2
            
    return pd.DataFrame(data, index=common)


# ============================================================
# 3. Feature Extraction (Extract-then-Resample)
# ============================================================

def process_vr_logs(vr_file, target_timestamps):
    """
    CRITICAL: Calculates features at 90Hz BEFORE downsampling.
    Captures rapid micro-adjustments lost in standard decimation.
    """
    df = pd.read_csv(vr_file)
    
    # 1. Native 90Hz Feature Calculation
    for p in ['P1', 'P2']:
        cols = [f'{p} head rotation {ax}' for ax in ['x', 'y', 'z']]
        # Velocity vector
        vel = df[cols].diff().fillna(0)
        # Jerk vector (derivative of acceleration)
        acc = vel.diff().fillna(0)
        jerk = acc.diff().fillna(0)
        
        # Norms
        df[f'{p}_HeadEnergy'] = np.linalg.norm(vel.values, axis=1)
        df[f'{p}_HeadJerk'] = np.linalg.norm(jerk.values, axis=1)

    # 2. Resample to 10Hz fNIRS grid
    vr_out = pd.DataFrame(index=target_timestamps)
    
    # Map features
    mappings = {
        "P1 hand displacement": "P1_HandDisp",
        "P2 hand displacement": "P2_HandDisp",
        "P1_HeadEnergy": "P1_HeadEnergy",
        "P2_HeadEnergy": "P2_HeadEnergy",
        "P1_HeadJerk": "P1_HeadJerk",
        "P2_HeadJerk": "P2_HeadJerk"
    }
    
    for src, dst in mappings.items():
        if src in df.columns:
            # Linear interpolation for continuous features
            vr_out[dst] = np.interp(target_timestamps, df["Time"], df[src])
        else:
            vr_out[dst] = 0.0
            
    # Composite Metrics for Regression/Annotation
    vr_out["Global_Motion"] = vr_out[["P1_HeadEnergy", "P2_HeadEnergy"]].mean(axis=1)
    vr_out["Dyad_HandDisp"] = vr_out[["P1_HandDisp", "P2_HandDisp"]].mean(axis=1)
    
    return vr_out

def process_ecg_physio(ecg_file, target_timestamps):
    """Aligns ECG PNS Index."""
    df = pd.read_csv(ecg_file)
    ecg_out = pd.DataFrame(index=target_timestamps)
    if "PNS index" in df.columns:
        ecg_out["PNS"] = np.interp(target_timestamps, df["Time"], df["PNS index"])
    return ecg_out


# ============================================================
# 4. Systemic Physiology Augmented (SPA) Regression
# ============================================================

def regress_systemic_noise(fnirs_df, vr_df, ecg_df):
    """
    Removes shared physiological arousal (ECG) and motion artifacts (VR)
    from the neural signal to prevent spurious 'Shared Arousal' synchrony.
    
    Model: HbO ~ Intercept + HeadMotion + PNS
    """
    cleaned_df = pd.DataFrame(index=fnirs_df.index)
    
    # Build Nuisance Regressors
    X = pd.DataFrame(index=fnirs_df.index)
    X['Motion'] = vr_df['Global_Motion']
    X['Physio'] = ecg_df['PNS'] if 'PNS' in ecg_df else 0
    X = X.fillna(0)
    
    model = LinearRegression()
    
    for col in fnirs_df.columns:
        y = fnirs_df[col]
        valid = ~y.isna()
        
        if valid.sum() > 30: # Need sufficient points
            X_val = X.loc[valid]
            y_val = y.loc[valid]
            
            model.fit(X_val, y_val)
            resid = y_val - model.predict(X_val)
            
            # Save Z-scored residual
            temp = pd.Series(index=y.index, dtype=float)
            temp.loc[valid] = resid
            cleaned_df[col] = zscore(temp, nan_policy='omit')
        else:
            cleaned_df[col] = np.nan
            
    return cleaned_df


# ============================================================
# 5. Graph Construction (The Output Engine)
# ============================================================

def build_annotated_graphs(fnirs_df, vr_df, ecg_df, start_time, 
                           window_size=150, step=10, hrf_lag=5.0):
    """
    Generates the final list of weighted, annotated NetworkX graphs.
    
    hrf_lag: Time shift (seconds) to align Neural(t) with Behavior(t-lag).
    """
    graph_list = []
    n = len(fnirs_df)
    lag_samples = int(hrf_lag * 10)
    
    for i in range(0, n - window_size, step):
        # 1. Neural Window (Current)
        w_fnirs = fnirs_df.iloc[i : i+window_size]
        
        # 2. Context Window (Lagged)
        ctx_start = max(0, i - lag_samples)
        ctx_end = max(0, i + window_size - lag_samples)
        w_vr = vr_df.iloc[ctx_start : ctx_end]
        w_ecg = ecg_df.iloc[ctx_start : ctx_end]
        
        # Quality Check
        if w_fnirs.isna().mean().mean() > 0.1: continue
        
        # 3. Weighted Connectivity Matrix
        # Absolute Pearson Correlation
        corr = w_fnirs.corr().abs()
        np.fill_diagonal(corr.values, 0) # No self-loops
        
        # Create Graph
        G = nx.from_numpy_array(corr.values)
        
        # 4. Annotation (Metadata embedding)
        t_center = w_fnirs.index[len(w_fnirs)//2]
        
        G.graph['timestamp'] = t_center
        G.graph['phase'] = get_etc_phase(t_center, start_time)
        G.graph['hand_disp'] = w_vr['Dyad_HandDisp'].mean() if not w_vr.empty else 0
        G.graph['head_jerk'] = w_vr['P1_HeadJerk'].mean() if not w_vr.empty else 0
        G.graph['pns_index'] = w_ecg['PNS'].mean() if 'PNS' in w_ecg else 0
        G.graph['is_pseudo'] = False
        
        graph_list.append(G)
        
    return graph_list


def generate_pseudo_graphs(real_bundle_list):
    """
    Control: Shuffles partners to create Pseudo-Dyads.
    Input: List of dicts {'fnirs': df, 'vr': df, 'ecg': df}
    """
    pseudo_graphs = []
    num = len(real_bundle_list)
    if num < 2: return []
    
    for i in range(num):
        # Dyad A and Dyad B
        dA = real_bundle_list[i]
        dB = real_bundle_list[(i+1) % num]
        
        # Create Hybrid Dataframe (S1 from A, S2 from B)
        # Assuming columns are named S1_... and S2_...
        s1_cols = [c for c in dA['fnirs'].columns if "S1_" in c]
        s2_cols = [c for c in dB['fnirs'].columns if "S2_" in c]
        
        min_len = min(len(dA['fnirs']), len(dB['fnirs']))
        
        hybrid_df = pd.concat([
            dA['fnirs'][s1_cols].iloc[:min_len].reset_index(drop=True),
            dB['fnirs'][s2_cols].iloc[:min_len].reset_index(drop=True)
        ], axis=1)
        
        # Build Graphs using Dyad A's context/timing
        # Note: In pseudo, context is mismatched, which is the point
        graphs = build_annotated_graphs(
            hybrid_df, dA['vr'], dA['ecg'], start_time=0
        )
        
        for G in graphs:
            G.graph['is_pseudo'] = True
            pseudo_graphs.append(G)
            
    return pseudo_graphs


# ============================================================
# 6. Master Wrapper
# ============================================================

def run_etc_pipeline(s1_file, s2_file, vr_file, ecg_file, start_trigger=0):
    """
    Executes the full pipeline for a single dyad.
    Returns a dictionary suitable for export/analysis.
    """
    # 1. Load & Align
    fnirs_raw = process_dyad_fnirs_raw(s1_file, s2_file)
    vr_proc = process_vr_logs(vr_file, fnirs_raw.index)
    ecg_proc = process_ecg_physio(ecg_file, fnirs_raw.index)
    
    # 2. Clean (SPA Regression)
    fnirs_clean = regress_systemic_noise(fnirs_raw, vr_proc, ecg_proc)
    
    # 3. Build Real Graphs
    real_graphs = build_annotated_graphs(
        fnirs_clean, vr_proc, ecg_proc, start_trigger
    )
    
    # Bundle for Pseudo generation later
    data_bundle = {
        'fnirs': fnirs_clean,
        'vr': vr_proc,
        'ecg': ecg_proc
    }
    
    return {
        "real_graphs": real_graphs,
        "data_bundle": data_bundle 
    }

# ============================================================
# Example Usage
# ============================================================
#
# dyad1 = run_etc_pipeline("s1.csv", "s2.csv", "vr.csv", "ecg.csv")
# dyad2 = run_etc_pipeline("s3.csv", "s4.csv", "vr2.csv", "ecg2.csv")
#
# # Generate controls
# all_bundles = [dyad1['data_bundle'], dyad2['data_bundle']]
# pseudo_G = generate_pseudo_graphs(all_bundles)
#
# # Final Dataset for HyPhi
# final_dataset = {
#     "real": dyad1['real_graphs'] + dyad2['real_graphs'],
#     "pseudo": pseudo_G
# }
#
# # Pass 'final_dataset["real"]' to getFRCVec()
