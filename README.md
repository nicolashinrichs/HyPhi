# HyPhi(Φ)
The toolkit for detecting phase transitions in inter-brain networks by tracking discrete Ricci curvature and its entropy distribution.

This repository accompanies the manuscript "Beyond Inter-Brain Synchrony with HyPhi(Φ): A Pipeline for Characterizing Geometric Regimes in Hyperscanning Networks" by Nicolás Hinrichs, Noah Guzman, Jun Lin Liow, Merle Fairhurst, Gesa Hartwigsen, Leonhard Schilbach, Guillaume Dumas, and Melanie Weber (2026); it provides a comprehensive, modular pipeline for analyzing phase transitions in inter-brain coupling using geometric network analysis techniques applied to dual-EEG data.

## Overview

The pipeline includes:

* A ground-truth simulation model using Kuramoto oscillators.
* Empirical dual-EEG analysis (resting-state and behavioral task).
* Comparison between Forman-Ricci, and Augmented Forman-Ricci curvature metrics.
* Sliding window dynamic network construction.
* Phase transition detection using curvature distributions and entropy measures.

## Layout

The repository is split into seven main directories, each with a specific role in the research workflow. Each directory contains a README.md detailing its purpose and examples.

### `data`

Simulation and EEG recording data below 100MG. Larger files are hosted at https://osf.io/yah5u

### `miscellaneous`

Documentation and supplemental information.

### `software_module`

Contains scripts and pipelines for data processing, analysis, exploratory data analysis, and figure generation.

## Installation and Dependencies

Ensure Python 3.8+ is installed. Required Python libraries:

* NumPy
* SciPy
* NetworkX
* Matplotlib
* Pandas
* MNE-Python (for EEG processing)
* GraphRicciCurvature (for curvature computations)

## Scientific Motivation

Traditional synchrony metrics often miss critical dynamic topological changes. Geometric descriptors like the Forman-Ricci curvature capture network reconfigurations, providing insights into how brains dynamically couple and decouple during interactions.

## Licensing

This repository is released under the BSD-3-Clause license. See `LICENSE` for more details.

## Citation

Please cite:

> Hinrichs, N., et al. (2026). *Beyond Inter-Brain Synchrony with HyPhi(Φ): A Pipeline for Characterizing Geometric Regimes in Hyperscanning Networks*. \[Manuscript in preparation].

## Contact

For questions, issues, or collaboration inquiries, please contact Nicolás Hinrichs at [hinrichsn@cbs.mpg.de](mailto:hinrichsn@cbs.mpg.de).
