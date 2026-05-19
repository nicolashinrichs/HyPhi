# Dual-EEG analysis pipeline

A complete walkthrough for analysing a hyperscanning dataset with HyPhi: from raw EEG
files to a publication-ready figure.  Cross-platform; works on macOS, Linux, and
Windows once `uv sync` has succeeded.

This tutorial assumes you have already:

- Installed the package (`uv sync --extra develop --extra notebook`)
- Verified the install with the [quickstart](01_quickstart.py)

For installation help see the project [README](../README.md).

---

## 1. What HyPhi expects

The simplest entry point is a NumPy `.npy` file containing **phase trajectories** for
one recording session, with shape:

```
(n_oscillators, n_time_samples)
```

where `n_oscillators` is the **total number of channels across both participants**
(e.g. 64 + 64 = 128 for two 64-channel recordings) and `n_time_samples` is the length
of the trimmed recording in samples.

If you instead have a precomputed connectivity matrix per window, the shape is:

```
(n_windows, n_oscillators, n_oscillators)
```

with values in `[0, 1]` (PLV) or `[-1, 1]` (correlation-based).

## 2. Recommended on-disk layout

Inside the project, create a `data/your-study/` folder with one phase file per dyad
(participant pair):

```
data/your-study/
  dyad_01_phases.npy
  dyad_02_phases.npy
  ...
  metadata.csv          # one row per dyad: dyad_id, condition, sex, age, ...
```

`data/` is already in HyPhi's path config (`paths.DATA`).

## 3. Convert raw EEG to phase time series

If your data are in **BrainVision** (`.vhdr`), **EDF**, **FIF**, or **EEGLAB** (`.set`)
format, the recipe below uses MNE-Python (installed by HyPhi).  Save it as
`prepare_phases.py` in the project root.

```python
"""Convert one dyad's raw dual-EEG to a HyPhi-ready phase array."""
from pathlib import Path

import mne
import numpy as np
from scipy.signal import hilbert

# --- EDIT THESE THREE PATHS ------------------------------------------------
SUBJECT_A_RAW = Path("/path/to/dyad_01_subjectA.vhdr")
SUBJECT_B_RAW = Path("/path/to/dyad_01_subjectB.vhdr")
OUT_FILE      = Path("data/your-study/dyad_01_phases.npy")
# ---------------------------------------------------------------------------

raw_a = mne.io.read_raw_brainvision(SUBJECT_A_RAW, preload=True)
raw_b = mne.io.read_raw_brainvision(SUBJECT_B_RAW, preload=True)

assert raw_a.info["sfreq"] == raw_b.info["sfreq"], "Sampling rates differ"
n_samples = min(raw_a.n_times, raw_b.n_times)

# Bandpass to a frequency band of interest (alpha shown here)
raw_a.filter(l_freq=8.0, h_freq=12.0, fir_design="firwin")
raw_b.filter(l_freq=8.0, h_freq=12.0, fir_design="firwin")

# Hilbert analytic-signal phase, per channel
data_a = raw_a.get_data()[:, :n_samples]
data_b = raw_b.get_data()[:, :n_samples]
phases_a = np.angle(hilbert(data_a, axis=1))
phases_b = np.angle(hilbert(data_b, axis=1))

# Stack into (n_oscillators, T) — subject A on top of subject B
phases = np.vstack([phases_a, phases_b])

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_FILE, phases)
print(f"Saved {phases.shape} → {OUT_FILE}")
```

Run it for each dyad:

```shell
uv run python prepare_phases.py
```

## 4. End-to-end pipeline

Save this as `run_pipeline.py` in the project root.  It produces per-window curvature
entropies, a null-model envelope, and a hierarchical-permutation p-value.

```python
"""End-to-end HyPhi pipeline on a prepared dual-EEG study."""
from pathlib import Path

import numpy as np
import pandas as pd

from hyphi.configs import paths
from hyphi.modeling.windowing import sliding_window_plv
from hyphi.modeling.graph_curvatures import compute_frc_vec, extract_curvatures_vec
from hyphi.modeling.entropies import vec_entropy, entropy_kde_plugin
from hyphi.null_models import phase_randomize
from hyphi.stats import entropy_to_long_df, hierarchical_permutation_test

study_dir = Path(paths.DATA) / "your-study"
out_dir   = Path(paths.RESULTS) / "your-study"
out_dir.mkdir(parents=True, exist_ok=True)

# Pipeline parameters
SFREQ        = 500          # sampling rate of the saved phase files (Hz)
WIN_SEC      = 2.0          # window length (s)
STRIDE_SEC   = 0.5          # window stride (s)
WIN_SAMPLES  = int(WIN_SEC    * SFREQ)
STRIDE       = int(STRIDE_SEC * SFREQ)
N_PERMS      = 1000
N_SURROGATES = 50
RNG          = np.random.default_rng(0)

meta = pd.read_csv(study_dir / "metadata.csv")

real_entropy: dict[str, np.ndarray] = {}
null_entropy: dict[str, np.ndarray] = {}

for _, row in meta.iterrows():
    dyad_id = row["dyad_id"]
    phases  = np.load(study_dir / f"{dyad_id}_phases.npy")

    graphs     = sliding_window_plv(phases, win_size=WIN_SAMPLES, win_stride=STRIDE)
    frc_graphs = compute_frc_vec(graphs)
    h          = vec_entropy(frc_graphs, entropy_kde_plugin)
    real_entropy[dyad_id] = h

    null_h = np.empty((N_SURROGATES, len(graphs)))
    for s in range(N_SURROGATES):
        surrogate = phase_randomize(phases, rng=RNG)
        sgraphs   = sliding_window_plv(surrogate, win_size=WIN_SAMPLES, win_stride=STRIDE)
        sfrc      = compute_frc_vec(sgraphs)
        null_h[s] = vec_entropy(sfrc, entropy_kde_plugin)
    null_entropy[dyad_id] = null_h.mean(axis=0)

np.savez(out_dir / "entropy_per_window.npz",
         **{f"real_{k}": v for k, v in real_entropy.items()},
         **{f"null_{k}": v for k, v in null_entropy.items()})

# Hierarchical permutation test
data = {(row["dyad_id"], row["condition"]): real_entropy[row["dyad_id"]]
        for _, row in meta.iterrows()}
df = entropy_to_long_df(data)
res = hierarchical_permutation_test(
    data=df, value_col="entropy", condition_col="condition",
    n_perms=N_PERMS, seed=0,
)
print(f"hierarchical permutation p = {res['p_value']:.4f}")
df.to_csv(out_dir / "entropy_long_form.csv", index=False)
```

Run it:

```shell
uv run python run_pipeline.py
```

Outputs land in `results/your-study/`.

## 5. Curvature, entropy, and null-model variants

The pipeline above defaults to Forman-Ricci curvature, KDE-plugin entropy, and the
phase-randomisation surrogate.  Swap any of these out:

### 5.1 Curvature

| Function                                          | Notes                                            |
|---------------------------------------------------|--------------------------------------------------|
| `hyphi.modeling.graph_curvatures.compute_frc_vec` | Forman-Ricci.  Default; fastest.                 |
| `hyphi.modeling.graph_curvatures.compute_afrc_vec`| Augmented Forman-Ricci (triangles + quads).      |
| `hyphi.modeling.graph_curvatures.compute_orc_vec` | Ollivier-Ricci.  Slowest (~10× FRC).             |

### 5.2 Entropy estimators

All live in `hyphi.modeling.entropies`:

| Function                    | Notes                                              |
|-----------------------------|----------------------------------------------------|
| `entropy_kde_plugin`        | Kernel-density plug-in.  Default; robust.          |
| `entropy_kozachenko`        | Kozachenko-Leonenko k-NN estimator.                |
| `entropy_vasicek`           | Vasicek spacings estimator.                        |
| `entropy_van_es`            | Van Es spacings estimator.                         |
| `entropy_renyi`             | Rényi entropy of order α.                          |
| `entropy_tsallis`           | Tsallis entropy of order q.                        |

### 5.3 Null-model surrogates

In `hyphi.null_models`:

| Function                              | Destroys                                  |
|---------------------------------------|-------------------------------------------|
| `phase_randomize`                     | Channel-by-channel phase relationships    |
| `circular_time_shift`                 | Cross-channel timing                      |
| `dyad_subject_swap`                   | Within-dyad pairing                       |
| `dyad_label_shuffle`                  | Dyad-level condition labels               |
| `condition_label_shuffle_within_dyad` | Trial-level condition labels per dyad     |
| `generate_surrogate_stack`            | Convenience wrapper for N surrogates      |

## 6. Visualise

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for _, row in meta.iterrows():
    h = real_entropy[row["dyad_id"]]
    color = {"sync": "tab:blue", "control": "tab:grey"}[row["condition"]]
    ax[0].plot(h, color=color, alpha=0.6)
ax[0].set_ylabel("Curvature entropy (nats)")

all_null = np.stack(list(null_entropy.values()))
ax[1].fill_between(
    np.arange(all_null.shape[1]),
    np.percentile(all_null, 5,  axis=0),
    np.percentile(all_null, 95, axis=0),
    alpha=0.3, color="tab:orange", label="null 5–95%",
)
ax[1].set_xlabel("Sliding window index")
ax[1].set_ylabel("Null entropy")
ax[1].legend()

fig.tight_layout()
fig.savefig(out_dir / "entropy_timecourse.png", dpi=200)
```

For a network plot at the peak window:

```python
from hyphi.visualization.network_plots import plot_curvature_network

peak_dyad = max(real_entropy, key=lambda d: real_entropy[d].max())
peak_idx  = int(np.argmax(real_entropy[peak_dyad]))

phases = np.load(study_dir / f"{peak_dyad}_phases.npy")
graphs = sliding_window_plv(phases, win_size=WIN_SAMPLES, win_stride=STRIDE)
G_frc  = compute_frc_vec([graphs[peak_idx]])[0]

fig = plot_curvature_network(G_frc, curvature_attr="formanCurvature", layout="spring")
fig.savefig(out_dir / f"network_dyad_{peak_dyad}_w{peak_idx}.png", dpi=200)
```

## 7. Benchmark against standard hyperscanning metrics

See `hyphi.benchmarks.classify_curvature_vs_benchmarks` for a `StratifiedGroupKFold`
classifier comparison between curvature features and PLV / wPLI / graph-theoretic
features computed on the same windowed graphs.

## 8. Reproducibility

For full reproducibility, version-control your `prepare_phases.py` and `run_pipeline.py`
alongside the data (or seed every `np.random.default_rng` you use).  The pipeline above
sets `seed=0` everywhere; change it only deliberately.
