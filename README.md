# HiPhi(Φ): the toolkit for detecting phase transitions in inter-brain networks by tracking discrete Ricci curvature and its entropy distribution.

This repository accompanies the manuscript "Beyond Inter-Brain Synchrony: Geometric Phase Transitions in Hyperscanning Networks" by Nicolás Hinrichs, Noah Guzman, Jun Lin Liow, Dimitris Bolis, Gesa Hartwigsen, Leonhard Schilbach, Guillaume Dumas, and Melanie Weber (2025); it provides a comprehensive, modular pipeline for analyzing phase transitions in inter-brain coupling using geometric network analysis techniques applied to dual-EEG data.

## Overview

The pipeline includes:

* A ground-truth simulation model using Kuramoto oscillators.
* Empirical dual-EEG analysis (resting-state and behavioral task).
* Comparison between Ollivier-Ricci, Forman-Ricci, and Augmented Forman-Ricci curvature metrics.
* Sliding window dynamic network construction.
* Phase transition detection using curvature distributions and entropy measures.

## Layout

The repository is split into seven main directories, each with a specific role in the research workflow. Each directory contains a README.md detailing its purpose and examples.

### `experiments`

Contains scripts and pipelines for data processing, analysis, exploratory data analysis, and figure generation.

* `processing/`: Data preprocessing and sliding window creation.
* `analysis/`: Curvature computation and phase transition detection.
* `exploratory/`: Scripts and notebooks for exploratory analyses.
* `figures/`: Code and examples for generating publication-quality figures.

### `data`

Raw EEG datasets and processed data outputs.

* Includes both resting-state and task-based EEG recordings.

### `miscellaneous`

Documentation and supplemental information.

* `protocols/`: experimental protocols and data acquisition details.
* `materials/`: information about EEG devices and experimental setup.
* `software_details/`: detailed computational environment documentation.

### `tests`

Scripts for validating pipeline functions.

### `software_module`

Custom software functions and modules.

### `templates`

Blank templates for documenting experiments, simulations, and analyses.

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

> Hinrichs, N., et al. (2025). *Beyond Inter-Brain Synchrony: Geometric Phase Transitions in Hyperscanning Networks*. \[Manuscript in preparation].

## Contact

For questions, issues, or collaboration inquiries, please contact Nicolás Hinrichs at [hinrichsn@cbs.mpg.de](mailto:hinrichsn@cbs.mpg.de).
