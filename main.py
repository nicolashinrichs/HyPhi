#!/usr/bin/env python3
"""
HyPhi — Master Pipeline Entry Point
====================================
This script documents the chronological execution of the full analysis pipeline.

To run the full pipeline, uncomment the desired steps below and execute:
    python main.py

Prerequisites:
    pip install -r requirements.txt
"""

import sys
import os

# Ensure src/ is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    print("=" * 60)
    print("HyPhi — Geometric Hyperscanning Pipeline")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Data Preparation
    # ----------------------------------------------------------
    # Load raw CCORR tensors from .mat files and reshape.
    # Requires empirical data in the paths specified by the TOML config.
    #
    # TODO: Uncomment when data is available
    # from hyphi.io_utils import load_config
    # config = load_config("experiments/analysis/CCORRconfig_001.toml")
    # print("[Step 1] Data loaded.")

    # ----------------------------------------------------------
    # Step 2: CCORR Curvature Analysis (FRC / AFRC)
    # ----------------------------------------------------------
    # Compute Forman-Ricci and Augmented Forman-Ricci curvatures
    # across windowed CCORR connectivity matrices.
    #
    # TODO: Uncomment when data is available
    # os.system("cd software_module && python HyperCCORRFRC.py CCORRconfig_001.toml")
    # os.system("cd software_module && python HyperCCORRAugFRC.py CCORRconfig_001.toml")
    # print("[Step 2] FRC/AFRC curvatures computed.")

    # ----------------------------------------------------------
    # Step 3: CCORR Curvature Analysis (ORC)
    # ----------------------------------------------------------
    # Compute Ollivier-Ricci curvatures (parallelised over freq bands).
    #
    # TODO: Uncomment when data is available
    # os.system("cd software_module && python HyperCCORRORC.py CCORRconfig_001.toml")
    # print("[Step 3] ORC curvatures computed.")

    # ----------------------------------------------------------
    # Step 4: Connectome-Informed Kuramoto with Delays
    # ----------------------------------------------------------
    # Run delayed DDE Kuramoto simulation using structural connectivity.
    #
    # TODO: Uncomment when simulation logic is finalised
    # from hyphi.simulations import setup_delayed_kuramoto, run_delayed_kuramoto
    # print("[Step 4] Kuramoto simulations completed.")

    # ----------------------------------------------------------
    # Step 5: Kuramoto PLV Curvature Analysis
    # ----------------------------------------------------------
    # Compute curvatures on Kuramoto-derived PLV networks.
    #
    # TODO: Uncomment when Kuramoto data is available
    # os.system("cd software_module && python KuramotoFRC.py CCORRconfig_001.toml FRC")
    # os.system("cd software_module && python KuramotoFRC.py CCORRconfig_001.toml AFRC")
    # print("[Step 5] Kuramoto curvatures computed.")

    # ----------------------------------------------------------
    # Step 6: Watts-Strogatz Sweep
    # ----------------------------------------------------------
    # Run the small-world rewiring probability sweep.
    #
    # TODO: Uncomment to run
    # from hyphi.simulations import run_ws_sweep
    # pt, Hreps, Qreps = run_ws_sweep(n_reps=200)
    # print("[Step 6] Watts-Strogatz sweep completed.")

    # ----------------------------------------------------------
    # Step 7: Statistical Tests
    # ----------------------------------------------------------
    # Run mixed-effects models and hierarchical permutation tests.
    #
    # TODO: Uncomment when entropy data is available
    # from hyphi.stats import mixed_effects_test, hierarchical_permutation_test, cohens_d
    # print("[Step 7] Statistical tests completed.")

    # ----------------------------------------------------------
    # Step 8: Null Models & Controls
    # ----------------------------------------------------------
    # Generate surrogate data for null hypothesis testing.
    #
    # TODO: Uncomment when data is available
    # from hyphi.null_models import phase_randomize, circular_time_shift, dyad_shuffle
    # print("[Step 8] Null models generated.")

    # ----------------------------------------------------------
    # Step 9: Benchmark Comparisons
    # ----------------------------------------------------------
    # Compute standard hyperscanning metrics and compare via classification.
    #
    # TODO: Uncomment when data is available
    # from hyphi.benchmarks import classify_curvature_vs_benchmarks
    # print("[Step 9] Benchmark comparisons completed.")

    # ----------------------------------------------------------
    # Step 10: Figure Generation
    # ----------------------------------------------------------
    # Generate manuscript figures.
    #
    # TODO: Uncomment when all prior steps have been executed
    # os.system("cd software_module && python NeuRepsSimulations.py")
    # os.system("cd software_module && python PlotHyperCCORRFRC.py")
    # print("[Step 10] Figures generated.")

    print()
    print("Pipeline script finished.")
    print("Uncomment individual steps above to execute them.")


if __name__ == "__main__":
    main()
