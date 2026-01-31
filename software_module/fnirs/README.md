## HyPhi Example Code: fNIRS x ECG x VR Integration

For all examples below, for the import statements to work, the scripts `fnirs_ecg_vr.py` and `fnirs_ecg_vr_v2.py` utilizing these functions need to be located in the `software_module` directory of the `HyPhi` repo. Most of the examples build on the previous ones and are meant to illustrate a typical workflow when using this package for the Embodied Telepresence Connection (ETC) project.

These pipelines bridge the gap between artifact-corrected fNIRS data (`LIONirs` export) and the geometric analysis tools provided by HyPhi.

### Basic Pipeline (v1): Rapid Exploration

All functions utilized in this example are defined in `fnirs_ecg_vr.py`.

This workflow implements standard preprocessing. It is suitable for initial data exploration, visualizing raw connectivity trends, or tasks with minimal motion artifacts where systemic physiological confounders are not the primary concern. It handles `LIONirs` NaN artifacts via cubic spline interpolation and aligns VR/ECG streams directly.

```python

from fnirs_ecg_vr import *

# 1. Load and align data streams
# Handles LIONirs exports and standardizes time indices
# Returns aligned dataframe and validity mask
fnirs_df, _ = process_dyad_fnirs_from_csv("data/s1_lionirs.csv", "data/s2_lionirs.csv")

# 2. Process Contextual Data
# Simple resampling of VR and ECG to the fNIRS timebase
vr_df = process_vr_logs("data/vr_log.csv", fnirs_df.index)
ecg_df = process_ecg_physio("data/ecg_log.csv", fnirs_df.index)

# 3. Construct Annotated Graphs
# Builds weighted correlation matrices and embeds metadata (Phase, Hand Disp)
# Returns a list of NetworkX graph objects
graphs_v1 = build_annotated_graphs(
    fnirs_df, 
    vr_df, 
    ecg_df, 
    start_time=10.0, 
    window_size=150
)

```

### Optimized Pipeline (v2): Systemic Physiology Augmented (SPA)

All functions utilized in this example are defined in `fnirs_ecg_vr_v2.py`.

This workflow implements the rigorous SPA-fNIRS protocol required for social neuroscience claims. It explicitly controls for spurious correlations arising from shared arousal and motion artifacts. It features Extract-then-Resample VR processing, SPA Regression (regressing out ECG/Motion before connectivity), Hemodynamic Lag Correction, and Pseudo-Dyad generation.

```python

from fnirs_ecg_vr_v2 import *

# 1. Execute the Full Pipeline for Real Dyads
# Loads, cleans (SPA Regression), and builds graphs with 5s lag correction
# Returns a dictionary containing 'real_graphs' and 'data_bundle'
dyad_A = run_etc_pipeline(
    "data/s1.csv", "data/s2.csv", "data/vr_A.csv", "data/ecg_A.csv", start_trigger=10.0
)
dyad_B = run_etc_pipeline(
    "data/s3.csv", "data/s4.csv", "data/vr_B.csv", "data/ecg_B.csv", start_trigger=15.0
)

# Aggregate Real Graphs
graphs_real = dyad_A['real_graphs'] + dyad_B['real_graphs']

# 2. Generate Pseudo-Dyad Controls (Critical for Validity)
# Shuffles partners (Subject 1 from Dyad A + Subject 2 from Dyad B) 
# to create a null distribution controlling for task-locked coincidence.
all_bundles = [dyad_A['data_bundle'], dyad_B['data_bundle']]
graphs_pseudo = generate_pseudo_graphs(all_bundles)
```

### Downstream Geometric Analysis

Once the `real_graphs` and `pseudo_graphs` are generated, they can be fed into the standard HyPhi analysis pipeline.

For detailed documentation on `getFRCVec` (Forman-Ricci Curvature) and `vecEntropy` (Ricci Network Entropy), please refer to the main `Software Module` documentation.

Below is a brief example of how to bridge the ETC output with the core geometric modules.

```python

# Import core HyPhi modules from the parent directory
import sys
sys.path.append("..") 
from GraphCurvatures import getFRCVec
from Entropies import vecEntropy, getEntropyKozachenko
import pandas as pd

# 1. Compute Curvature Trajectories
# We use 'augmented' curvature to account for clustering (triangles) in correlation graphs
frc_real = getFRCVec(graphs_real, method_val="augmented")
frc_pseudo = getFRCVec(graphs_pseudo, method_val="augmented")

# 2. Compute Entropy Trajectories
# Using k-Nearest Neighbors (k=4) for robust estimation
hKL = lambda G: getEntropyKozachenko(G, curvature="formanCurvature", num_nn=4)
rne_real = vecEntropy(frc_real, estim=hKL)
rne_pseudo = vecEntropy(frc_pseudo, estim=hKL)

# 3. Integrate with Metadata for Statistical Modeling
# We extract the metadata embedded by the ETC pipeline (v2) and combine it with the RNE results
results_data = []

for i, G in enumerate(graphs_real):
    row = G.graph.copy() # Extracts 'phase', 'hand_disp', 'pns_index'
    row['RNE'] = rne_real[i]
    row['Condition'] = 'Real'
    results_data.append(row)

for i, G in enumerate(graphs_pseudo):
    row = G.graph.copy()
    row['RNE'] = rne_pseudo[i]
    row['Condition'] = 'Pseudo'
    results_data.append(row)

df_results = pd.DataFrame(results_data)
# df_results is now ready for Linear Mixed Models (RNE ~ Condition * Phase)

```
