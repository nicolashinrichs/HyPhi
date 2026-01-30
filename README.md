# HyPhi(Φ)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18415664.svg)](https://doi.org/10.5281/zenodo.18415664)

The toolkit for detecting phase transitions in inter-brain networks by tracking discrete Ricci curvature and its entropy distribution.

This repository contains a modular pipeline for analyzing phase transitions in inter-brain coupling using geometric network analysis techniques, demonstrated on simulated and hyperscanning datasets.

Related benchmarks and applications of components of this toolkit are discussed in prior and ongoing work, including:

- **Hinrichs, N., Guzmán, N., & Weber, M. (2025).**  
  [*On a Geometry of Interbrain Networks.*](https://openreview.net/pdf?id=ouNpUPdUzH)  
  NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations (NeurReps).

- **Hinrichs, N., Hartwigsen, G., & Guzmán, N. (2025).**  
  [*Detecting Phase Transitions in EEG Hyperscanning Networks Using Geometric Markers.*](https://osf.io/preprints/osf/abx8u_v1)  
  Open Science Framework (OSF) Preregistration.

- **Hinrichs, N., Albarracin, M., Bolis, D., Jiang, Y., Christov-Moore, L., & Schilbach, L. (2025).**  
  [*Geometric Hyperscanning under Active Inference.*](https://doi.org/10.48550/arXiv.2506.08599)  
  6th International Workshop on Active Inference (IWAI 2025).

---

## Overview

HyPhi implements a geometry-driven alternative to traditional synchrony-based hyperscanning analysis.  
The pipeline includes:

- A ground-truth simulation framework based on coupled Kuramoto oscillators
- Empirical dual-EEG analysis
- Comparison between Forman-Ricci and Augmented Forman-Ricci curvature metrics
- Sliding-window dynamic network construction
- Phase transition detection using curvature distributions, entropy, and quantiles

---

## Conceptual Workflow

Across simulated and empirical use cases, HyPhi follows the same high-level workflow:

1. **Network construction**  
   Static or time-resolved graphs are constructed from simulations or empirical connectivity measures.

2. **Discrete curvature computation**  
   Edge-wise Ricci curvature (Forman-Ricci and variants) is computed on each network.

3. **Distributional analysis**  
   Curvature values are treated as distributions and analyzed via kernel density estimation.

4. **Information-theoretic tracking**  
   Entropy and quantiles of curvature distributions are used to detect regime shifts and phase transitions.

Concrete implementations of this workflow are provided in the `software_module` and `miscellaneous` directories.

---

## Layout

The repository is split into the following main directories, each with a dedicated `README.md`:

### `data`

Simulation and EEG-derived connectivity data below 100 MB.  
Larger files are hosted at:  
https://osf.io/yah5u

### `software_module`

Core analysis modules and executable pipelines, including:

- Network simulations
- Ricci curvature computation
- Density estimation
- Entropy and quantile analysis

This directory also contains a worked, end-to-end example illustrating the canonical HyPhi workflow on synthetic networks.

### `miscellaneous`

Supplementary documentation and tutorials, including a step-by-step protocol demonstrating Forman-Ricci curvature analysis in hyperscanning-style networks.

---

## Installation and Dependencies

Ensure Python 3.8+ is installed.  
Required Python libraries include:

- NumPy
- SciPy
- NetworkX
- Matplotlib
- Pandas
- MNE-Python (for EEG processing)
- GraphRicciCurvature (for curvature computations)

---

## Scientific Motivation

Traditional synchrony metrics collapse rich network structure into low-dimensional summaries and often miss critical topological transitions.

HyPhi instead treats inter-brain coupling as a **dynamic geometric object**, where curvature captures higher-order structural reorganization.  
This enables principled detection of coupling and decoupling regimes beyond synchrony alone.

---

## Citation

If you use this software, please cite:

Nicolás Hinrichs & Noah Guzmán (2026).  
*HyPhi(Φ): A toolkit for detecting phase transitions in inter-brain networks*.  
nicolashinrichs/HyPhi: Second release (v1.1.0). Zenodo.  
https://doi.org/10.5281/zenodo.18415663  

Latest DOI: https://doi.org/10.5281/zenodo.18415664

---

## Licensing

This repository is released under the BSD-3-Clause license.  
See `LICENSE` for details.

---

## Contact

For questions, issues, or collaboration inquiries, contact:  
Nicolás Hinrichs  
[hinrichsn@cbs.mpg.de](mailto:hinrichsn@cbs.mpg.de)
