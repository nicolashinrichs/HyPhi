# EEG CCorr → NetworkX Graphs

A lightweight and adaptable Python 3.9+ utility for converting inter-brain EEG/MEG connectivity matrices—stored in MATLAB `.mat` format—into analysis-ready NetworkX graph objects.

While originally developed for the *Duorhythm* dataset (using adjusted circular correlation), the tool is designed to be **general-purpose** and can be easily adapted for other hyperscanning datasets with similar structure.

---

## Repository Layout

```
eeg-ccorr-to-networkx/
│
├── src/
│   └── build_graphs.py        # Main CLI script (argparse + main())
│
├── examples/
│   └── README.md              # Instructions for running on toy data
│
├── sample_data/
│   └── CCORR_00.mat           # Dummy file: 1 epoch × 1 band (anonymised)
│
├── .gitignore
├── requirements.txt
└── README.md                  # Project front page
```

---

## Input Format

The script expects `.mat` files containing 4D connectivity matrices:

```
[frequency bands] × [epochs] × [sensors] × [sensors]
```

**In the Duorhythm dataset**, these files are named `CCORR_<dyad>.mat` and include:

- `CCORR_AA`, `CCORR_SA`, `CCORR_F` (conditions)
- Shape: `8 × N × 128 × 128`
- 8 canonical EEG bands
- N epochs per condition (e.g., 20 × 3s)
- Sensors 0–63 = Participant 1, 64–127 = Participant 2

**To adapt to other datasets**, edit the loading logic and condition keys in `build_graphs.py`.

---

## Output

A single `.pkl` file containing:

```
graphs[dyad_id][frequency_band][condition] → networkx.Graph
```

Each dyad includes:

- Full 64 × 64 weighted undirected graph (unless `--roi-only`)
- ROI subgraphs (20 selected electrodes)
- Surrogate null graph (if provided)
- Behavioral vectors (e.g., MARP/SDRP) if available

---

## Installation

```bash
python3 -m venv venv           # optional but recommended
source venv/bin/activate
pip install -r requirements.txt
```

Minimal dependencies: NumPy, SciPy, NetworkX.

---

## Quick Start

To run on your own dataset:

```bash
python -m src.build_graphs \
    --input     /path/to/ccorr_matrices \
    --behaviour /path/to/relativephasemeasures.mat \
    --output    graphs_networkx_plus.pkl
```

To test on the sample data:

```bash
python -m src.build_graphs \
    --input     sample_data \
    --output    sample_graphs.pkl \
    --dyads     00
```

---

## Command-Line Options

| Flag            | Default                      | Description                                         |
|-----------------|------------------------------|-----------------------------------------------------|
| `--input`       | (required)                   | Folder with `CCORR_*.mat` files                    |
| `--behaviour`   | (optional)                   | Path to `relativephasemeasures.mat`               |
| `--output`      | `graphs_networkx_plus.pkl`   | Output file                                        |
| `--dyads`       | all                          | Dyads to include: e.g. `01,02,10` or `1-5`         |
| `--roi-only`    | false                        | Only save ROI graphs                               |
| `--overwrite`   | false                        | Overwrite existing output file                     |

For full help:

```bash
python -m src.build_graphs -h
```

---

## ROI Subgraphs

ROI graphs contain only the following 20 electrodes (Duorhythm default):

- RF: 30, 31, 32, 60, 62  
- RT: 20, 22, 21, 26, 55  
- LF: 1, 3, 4, 34, 37  
- RP: 19, 23, 49, 50, 52

**To define custom ROIs**, modify `ROI_CLUSTERS` in `build_graphs.py`.

---

## Example Miniature Data

Run the full pipeline on a toy file (1 band × 1 epoch × 128 × 128):

```bash
python -m src.build_graphs --input sample_data --output toy.pkl --dyads 00
```

Expected output:

```
Saved 1 dyads → toy.pkl
```

---
