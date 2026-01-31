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
    Longer gaps are preserved as NaN and marked invalid.
    """
    ts = series.copy()
    max_gap = int(max_gap_sec * sampling_rate)
    isnan = ts.isna()

    valid_mask = np.ones(len(ts), dtype=bool)
    groups = (isnan != isnan.shift()).cumsum()

    for _, idx in isnan.groupby(groups).groups.items():
        if isnan.iloc[idx[0]]:
            if len(idx) <= max_gap:
                start, end = idx[0] - 1, idx[-1] + 1
                if start >= 0 and end < len(ts):
                    cs = CubicSpline([start, end],
                                     [ts.iloc[start], ts.iloc[end]])
                    ts.iloc[idx] = cs(idx)
            else:
                valid_mask[idx] = False

    return ts, valid_mask


def process_dyad_fnirs(df_s1, df_s2, channels):
    """
    Applies identical preprocessing to both participants.
    Returns cleaned dyadic fNIRS data and a global validity mask.
    """
    data = {}
    masks = []

    for ch in channels:
        s1, m1 = imputation(df_s1[ch])
        s2, m2 = imputation(df_s2[ch])

        data[f"S1_{ch}"] = zscore(s1, nan_policy="omit")
        data[f"S2_{ch}"] = zscore(s2, nan_policy="omit")

        masks.append(m1 & m2)

    global_mask = np.logical_and.reduce(masks)
    return pd.DataFrame(data), global_mask


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

    for c in cols:
        vr[c] = np.interp(target_timestamps, df["Time"], df[c])

    for p in ["P1", "P2"]:
        rot = vr[[f"{p} head rotation {a}" for a in ["x", "y", "z"]]]
        vel = rot.diff().fillna(0)
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
    for col in ["PNS index", "Stress index", "Mean HR"]:
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
    window_size=150,
    step_size=10,
    nan_thresh=0.1
):
    """
    Generates a time-ordered list of weighted networkx graphs
    and corresponding metadata dictionaries.
    These graphs are passed directly to HyPhi's
    getFRCVec and vecEntropy functions.
    """
    graphs = []
    meta = []

    n = len(fnirs)

    for i in range(0, n - window_size, step_size):
        w_fnirs = fnirs.iloc[i:i + window_size]

        if w_fnirs.isna().mean().mean() > nan_thresh:
            continue

        # Functional connectivity (absolute Pearson correlation)
        corr = w_fnirs.corr().abs()
        np.fill_diagonal(corr.values, 0)

        G = nx.from_numpy_array(corr.values)
        graphs.append(G)

        meta.append({
            "t_start": fnirs.index[i],
            "t_end": fnirs.index[i + window_size - 1],
            "mean_PNS": ecg.iloc[i:i + window_size]["PNS index"].mean(),
            "mean_Stress": ecg.iloc[i:i + window_size]["Stress index"].mean(),
            "mean_hand_disp": vr.iloc[i:i + window_size]
                                .filter(like="hand displacement")
                                .mean().mean(),
            "mean_head_motion": vr.iloc[i:i + window_size]
                                .filter(like="HeadMotion")
                                .mean().mean()
        })

    return graphs, meta


# ============================================================
# Downstream HyPhi geometry (illustrative only)
# ============================================================
#
# from GraphCurvatures import getFRCVec
# from Entropies import vecEntropy, getEntropyKozachenko
#
# graphs, meta = build_graph_sequence(...)
#
# FRCt = getFRCVec(graphs, method_val="augmented")
#
# hKL = lambda G: getEntropyKozachenko(
#     G, curvature="formanCurvature", num_nn=4
# )
#
# Ht = vecEntropy(FRCt, estim=hKL)
