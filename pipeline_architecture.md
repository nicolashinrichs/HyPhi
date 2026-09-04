# HyPhi: High-Performance Hyperscanning Pipeline via Geometric Entropy

This document provides a comprehensive overview of the refactored **HyPhi** repository architecture. It is designed to serve as a reference for describing the computational pipeline, statistical methods, and reproducibility frameworks in scientific manuscripts.

## 1. Architectural Overview and Reproducibility

The HyPhi repository has been structurally refactored from a collection of flat scripts into a cohesive, modular Python package (`src/hyphi/`). This architecture adheres to modern software engineering standards for neuroscientific software, ensuring high reproducibility, maintainability, and ease of use.

### 1.1 Dependency Management
To ensure a stable and reproducible environment, dependencies are strictly managed via a `pyproject.toml` configuration. This pins critical libraries such as `GraphRicciCurvature`, `mne`, `KDEpy`, and `statsmodels`. This approach prevents future API breakages from affecting the pipeline's execution.

### 1.2 Pipeline Execution (`main.py` & `Makefile`)
The pipeline's execution flow is centrally documented in `main.py`, which serves as the end-to-end entry point for the analysis. It chronologically maps the process from data ingestion to graph construction, curvature computation, statistical analysis, and theoretical simulations. Additionally, a `Makefile` provides standardized commands for installation (`make install`), testing (`make test`), and execution (`make pipeline`), simplifying the user experience and ensuring consistent execution across different environments.

## 2. Core Modules (`src/hyphi/`)

The core logic of HyPhi is distributed across specialized modules within the `src/hyphi/` package, adhering to the DRY (Don't Repeat Yourself) principle.

### 2.1 Data Management (`io.py`)
This module handles all input/output operations, abstracting away the complexities of loading raw connectivity matrices and configuration parameters. It utilizes `tomllib` for robust parsing of TOML configurations and `pickle` for loading precomputed network graphs.

### 2.2 Analytical Engine (`analyses.py`)
This is the mathematical core of the repository, responsible for translating brain connectivity into geometric topologies.
*   **Time-Varying Graph Construction**: `build_sliding_window_graphs` transforms dynamic connectivity matrices (e.g., from phase locking or correlation) into sequences of undirected, weighted `NetworkX` graphs.
*   **Geometric Curvatures**: The module interfaces with `GraphRicciCurvature` to compute edge-level curvatures, primarily Forman-Ricci Curvature (FRC) and Ollivier-Ricci Curvature (ORC).
*   **Information-Theoretic Entropy**: The distribution of edge curvatures within a network characterizes its topology. The module employs advanced density estimation techniques—such as Tree-based Kernel Density Estimation (`TreeKDE` via `KDEpy`) and Vasicek's m-spacing estimator—to robustly calculate the differential entropy of these curvature distributions.

### 2.3 Statistical Framework (`stats.py`)
To rigorously evaluate differences in geometric entropy without suffering from pseudo-replication (a common issue when pooling trials or sliding windows), HyPhi implements advanced statistical modeling.
*   **Mixed-Effects Modeling**: `mixed_effects_model` utilizes `statsmodels` to fit Linear Mixed Models (LMMs). By modeling `dyad` and `trial` as random effects, it correctly nests the variance, ensuring that the degrees of freedom are not artificially inflated by highly correlated sliding windows.
*   **Hierarchical Permutation Testing**: A skeleton for hierarchical permutation tests is provided, allowing for non-parametric significance testing that respects the nested structure of the data (e.g., shuffling at the subject or dyad level, rather than the window level).
*   **Effect Size**: `cohens_d` provides a standardized measure of practical significance for the observed differences in entropy.

### 2.4 Null Models (`null_models.py`)
Constructing appropriate surrogate datasets is critical for validating hyperscanning results. HyPhi provides specialized functions to generate valid null distributions:
*   **Phase Randomization**: `generate_phase_randomization` operates in the frequency domain to shuffle signal phases while preserving the original amplitude spectrum, destroying true temporal correlations while maintaining auto-correlative properties.
*   **Circular Time-Shifting**: `generate_circular_time_shift` offsets signals by a random temporal lag, destroying instantaneous phase-locking between subjects while perfectly preserving the intra-individual signal dynamics.
*   **Dyad Shuffling (Pseudo-Dyads)**: `generate_dyad_shuffled_null` repairs subjects across different experimental dyads to create "pseudo-dyads" that lack true interactive coupling, serving as a baseline for true social interaction.

### 2.5 Benchmarking (`benchmarks.py`)
To demonstrate the utility of geometric entropy, the package includes classical hyperscanning metrics for comparative benchmarking.
*   **Classical Synchrony**: Implementations of Phase Locking Value (PLV), weighted Phase Lag Index (wPLI), and Imaginary Coherence.
*   **Graph Theoretic Baselines**: Global efficiency and modularity computations.
*   **Predictive Modeling**: `evaluate_classifier_skeleton` provides a cross-validation framework using Support Vector Classifiers (SVCs) to directly compare the predictive power of geometric entropy against these classical baselines for distinguishing experimental conditions.

### 2.6 Theoretical Simulations (`simulations.py`)
To mathematically validate the geometric properties, the pipeline includes scaffolds for running generative simulations.
*   **Delay-Coupled Kuramoto Models**: Simulating non-linear oscillator dynamics over empirical connectomes to understand how phase synchronization yields specific geometric patterns.
*   **Watts-Strogatz Sweeps**: Exploring how small-world topological transitions affect Forman-Ricci curvature distributions.

## 3. Automated Validation (`tests/`)

HyPhi implements a robust testing suite using `pytest` to guarantee mathematical correctness and prevent regressions.
*   **Known-Answer Tests**: Using canonical graph topologies with mathematically provable curvatures (e.g., Complete Graphs $K_n$, Ring Lattices $C_n$, and Star Graphs $S_n$), the tests continuously verify that the fundamental FRC and AFRC calculations behave exactly as expected.
*   **Invariant Testing**: The suite ensures invariants hold true, such as confirming that homogeneous graphs (where all edges share identical curvature) correctly yield an estimated differential entropy of zero.
