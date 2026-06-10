# Dual-EEG analysis pipeline

A complete walkthrough for analyzing a hyperscanning dataset with HyPhi: from raw EEG
files to a publication-ready figure.  Cross-platform; works on macOS, Linux, and
Windows once `uv sync` has succeeded.

This tutorial assumes you have already:

- Installed the package (`uv sync --extra develop --extra notebook`)
- Verified the installation with the [quickstart](01_quickstart.py)

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

Inside the project, create a `data/your-study/` folder with one phase file per
recording, i.e. per dyad (participant pair) AND condition:

```
data/your-study/
  dyad_01_sync_phases.npy
  dyad_01_control_phases.npy
  dyad_02_sync_phases.npy
  ...
  metadata.csv          # one row per recording: dyad_id, condition, sex, age, ...
```

Each dyad should appear under **every** condition (a within-dyad design): the
hierarchical permutation test in step 4 shuffles condition labels *within* each
dyad, so a dyad recorded under only one condition contributes nothing to the test
(the test will refuse a design where no dyad spans more than one condition).

> **Power, read before you interpret a p-value.** The within-dyad permutation
> draws from a finite set of label arrangements: roughly `prod over dyads of
> C(trials_in_dyad, trials_in_condition_A)`. With a *single* recording per
> (dyad, condition) the only within-dyad move is identity-or-swap, and the
> squared-difference default statistic is unchanged by the swap, so the smallest
> reachable p-value is about `2 / 2^n_dyads`: roughly 0.5 for 2 dyads and still
> above 0.05 below ~6 dyads, **no matter how large the true effect**. To detect
> an effect you need either many dyads (>= ~6-8) or multiple trials per condition
> per dyad (so each contributes more than one entropy value). HyPhi emits a
> warning when the permutation space is too small for significance to be
> reachable; do not read a non-significant p as evidence of no effect until you
> have checked the design is powered.

For a design with several recordings of the same (dyad, condition), key the
per-recording entropy by `(dyad_id, condition, recording)` and append each as a
separate trial under that condition, rather than overwriting a single slot, so
the permutation has real trials to shuffle.

`data/` is already in HyPhi's path config (`config.paths.DATA`, an absolute path
after `config.init()`).

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
SUBJECT_A_RAW = Path("/path/to/dyad_01_sync_subjectA.vhdr")
SUBJECT_B_RAW = Path("/path/to/dyad_01_sync_subjectB.vhdr")
OUT_FILE      = Path("data/your-study/dyad_01_sync_phases.npy")
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

from hyphi.configs import config
from hyphi.modeling.windowing import sliding_window_plv
from hyphi.modeling.graph_curvatures import compute_frc_vec, extract_curvatures_vec
from hyphi.modeling.entropies import vec_entropy, entropy_kde_plugin
from hyphi.null_models import phase_randomize
from hyphi.stats import entropy_to_long_df, hierarchical_permutation_test

config.init()

study_dir = Path(config.paths.DATA) / "your-study"
out_dir   = Path(config.paths.RESULTS) / "your-study"
out_dir.mkdir(parents=True, exist_ok=True)

# Pipeline parameters
SFREQ        = 500          # sampling rate of YOUR phase files (Hz) - check it, datasets differ (e.g. 250)
WIN_SEC      = 2.0          # window length (s)
STRIDE_SEC   = 0.5          # window stride (s)
WIN_SAMPLES  = int(WIN_SEC    * SFREQ)
STRIDE       = int(STRIDE_SEC * SFREQ)
N_PERMS      = 1000
N_SURROGATES = 50
RNG          = np.random.default_rng(0)

meta = pd.read_csv(study_dir / "metadata.csv")

real_entropy: dict[tuple[str, str], np.ndarray] = {}
null_entropy: dict[tuple[str, str], np.ndarray] = {}

for _, row in meta.iterrows():
    dyad_id   = row["dyad_id"]
    condition = row["condition"]
    phases    = np.load(study_dir / f"{dyad_id}_{condition}_phases.npy")

    graphs     = sliding_window_plv(phases, win_size=WIN_SAMPLES, win_stride=STRIDE)
    frc_graphs = compute_frc_vec(graphs)
    h          = vec_entropy(frc_graphs, entropy_kde_plugin)
    real_entropy[(dyad_id, condition)] = h

    null_h = np.empty((N_SURROGATES, len(graphs)))
    for s in range(N_SURROGATES):
        surrogate = phase_randomize(phases, rng=RNG)
        sgraphs   = sliding_window_plv(surrogate, win_size=WIN_SAMPLES, win_stride=STRIDE)
        sfrc      = compute_frc_vec(sgraphs)
        null_h[s] = vec_entropy(sfrc, entropy_kde_plugin)
    null_entropy[(dyad_id, condition)] = null_h.mean(axis=0)

np.savez(out_dir / "entropy_per_window.npz",
         **{f"real_{d}_{c}": v for (d, c), v in real_entropy.items()},
         **{f"null_{d}_{c}": v for (d, c), v in null_entropy.items()})

# Hierarchical permutation test.
# entropy_to_long_df expects {dyad: {condition: array of shape (n_freq, n_trials, n_windows)}};
# here each recording is one trial of one frequency band, hence the [None, None, :].
entropy_nested: dict[str, dict[str, np.ndarray]] = {}
for (dyad_id, condition), h in real_entropy.items():
    entropy_nested.setdefault(dyad_id, {})[condition] = h[None, None, :]

df = entropy_to_long_df(entropy_nested)
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
| `hyphi.modeling.graph_curvatures.compute_frc_vec` | Forman-Ricci.  Default; fastest. Edge attr `formanCurvature`. |
| `hyphi.modeling.graph_curvatures.compute_afrc_vec`| Augmented Forman-Ricci (triangles + quads). Edge attr `formanCurvature`. |
| `hyphi.modeling.graph_curvatures.compute_orc_vec` | Ollivier-Ricci.  Slowest (~10x FRC). Edge attr `ricciCurvature`. |

> **Not a silent drop-in for Ollivier-Ricci.** The entropy estimators and the
> network plot read the `formanCurvature` edge attribute by default, but
> `compute_orc_vec` writes `ricciCurvature`. If you swap it in, pass the
> attribute name explicitly downstream, e.g.
> `vec_entropy(orc_graphs, lambda g: entropy_kde_plugin(g, curvature="ricciCurvature"))`
> and `plot_curvature_network(G, curvature_attr="ricciCurvature")`; otherwise the
> estimators raise `KeyError`. `compute_afrc_vec` keeps `formanCurvature`, so it
> is a true drop-in.

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

In `hyphi.null_models`. These come in two families and are **not** mutually
interchangeable.

**Signal-domain** surrogates take a phase/signal array `(n_oscillators, T)` and
return a surrogate array, so they are drop-in replacements for `phase_randomize`
in the section-4 surrogate loop (`surrogate = fn(phases, rng=RNG)`):

| Function              | Signature                  | Destroys                              |
|-----------------------|----------------------------|---------------------------------------|
| `phase_randomize`     | `(signal, rng)`            | Channel-by-channel phase relationships |
| `circular_time_shift` | `(signal, min_shift, rng)` | Cross-channel timing                  |
| `dyad_subject_swap`   | `(data_matrix, rng)`       | Within-dyad pairing                   |
| `generate_surrogate_stack` | `(data, method, n_surrogates)` | Convenience wrapper that builds N of the above |

**Design-domain** nulls operate on the long-form label arrays (the dyad /
condition / trial columns), **not** on a phase array. They belong at the stats
stage (permuting labels), not in the per-recording surrogate loop, and cannot be
substituted for `phase_randomize`:

| Function                              | Signature                                       | Destroys                          |
|---------------------------------------|-------------------------------------------------|-----------------------------------|
| `dyad_label_shuffle`                  | `(dyad_labels, rng)`                            | Dyad-level condition labels       |
| `condition_label_shuffle_within_dyad` | `(condition_labels, dyad_labels, trial_labels)` | Trial-level condition labels per dyad |

## 6. Visualise

Append these blocks to the end of `run_pipeline.py` (they reuse its variables:
`real_entropy`, `null_entropy`, `out_dir`, `study_dir`, `WIN_SAMPLES`, `STRIDE`).

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Build a colour per condition from the data, so any condition names work
# (not just the hard-coded "sync"/"control").
conditions = sorted({c for _, c in real_entropy})
cmap = dict(zip(conditions, plt.cm.tab10.colors))
for (dyad_id, condition), h in real_entropy.items():
    ax[0].plot(h, color=cmap[condition], alpha=0.6)
ax[0].set_ylabel("Curvature entropy (nats)")

# Recordings can have different window counts; truncate to the shortest before
# stacking (a bare np.stack would raise on ragged lengths). This band is the
# 5-95% spread ACROSS recordings of each recording's mean null, not a
# within-recording surrogate envelope.
min_len = min(v.shape[0] for v in null_entropy.values())
all_null = np.stack([v[:min_len] for v in null_entropy.values()])
ax[1].fill_between(
    np.arange(min_len),
    np.percentile(all_null, 5,  axis=0),
    np.percentile(all_null, 95, axis=0),
    alpha=0.3, color="tab:orange", label="across-recording 5-95%",
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

peak_key = max(real_entropy, key=lambda k: real_entropy[k].max())
peak_idx = int(np.argmax(real_entropy[peak_key]))
dyad_id, condition = peak_key

phases = np.load(study_dir / f"{dyad_id}_{condition}_phases.npy")
graphs = sliding_window_plv(phases, win_size=WIN_SAMPLES, win_stride=STRIDE)
G_frc  = compute_frc_vec([graphs[peak_idx]])[0]

fig = plot_curvature_network(G_frc, curvature_attr="formanCurvature", layout="spring")
fig.savefig(out_dir / f"network_{dyad_id}_{condition}_w{peak_idx}.png", dpi=200)
```

## 7. Benchmark against standard hyperscanning metrics

See `hyphi.benchmarks.classify_curvature_vs_benchmarks` for a `StratifiedGroupKFold`
classifier comparison between curvature features and PLV / wPLI / graph-theoretic
features computed on the same windowed graphs.

## 8. Reproducibility

For full reproducibility, version-control your `prepare_phases.py` and `run_pipeline.py`
alongside the data (or seed every `np.random.default_rng` you use).  The pipeline above
sets `seed=0` everywhere; change it only deliberately.
