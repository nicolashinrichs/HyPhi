"""
Main entry for hyphi.

Run the Hyphi pipeline end-to-end, orchestrating data loading, preprocessing, analyses, and benchmarking.

Years: 2026
"""

# %% Import
import logging

from hyphi.configs import config, params, paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
logger = logging.getLogger(__name__)


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_pipeline():
    """Run the whole pipeline using hyphi."""
    logger.info("Starting HyPhi end-to-end pipeline...")

    # 1. Data Preparation
    logger.info("Step 1: Data Preparation & Graph Construction")
    # TODO: Load raw EEG/timeseries data using io.py
    # TODO: Perform sliding-window filtering and phase correlation to construct graphs
    # example:
    # from hyphi.io import load_connectivity_data
    # from hyphi.analyses import build_sliding_window_graphs
    # data = load_connectivity_data("data/raw_data.pkl")
    # graphs = build_sliding_window_graphs(data, window_size=..., overlap=...)

    # 2. Curvature & Entropy Computation
    logger.info("Step 2: Curvature & Entropy Computation")
    # TODO: Compute Forman-Ricci (FRC) or Augmented FRC (AFRC) curvature
    # TODO: Compute geometric entropy on the curvature distributions
    # example:
    # from hyphi.analyses import compute_windowed_curvatures, compute_entropy
    # curvatures = compute_windowed_curvatures(graphs, method="AFRC")
    # entropies = compute_entropy(curvatures)

    # 3. Statistical Analysis
    logger.info("Step 3: Statistical Analysis & Null Models")
    # TODO: Run mixed-effects models or hierarchical permutation testing to test significance
    # TODO: Generate null models (e.g., dyad shuffling, phase randomization)
    # example:
    # from hyphi.stats import mixed_effects_model
    # from hyphi.null_models import generate_dyad_shuffled_null
    # p_values = mixed_effects_model(entropies, conditions=...)
    # null_entropies = generate_dyad_shuffled_null(data)

    # 4. Benchmarking
    logger.info("Step 4: Benchmarking against Standard Metrics")
    # TODO: Compute standard metrics (PLV, wPLI) and compare using classifying models
    # example:
    # from hyphi.benchmarks import compute_plv, evaluate_classifier
    # plv_scores = compute_plv(graphs)
    # clf_results = evaluate_classifier(entropies, plv_scores)

    # 5. Simulations (Optional)
    logger.info("Step 5: Theoretical Simulations")
    # TODO: Run Connectome-informed Kuramoto with delays and Watts-Strogatz sweep
    # example:
    # from hyphi.simulations import run_kuramoto_simulation
    # run_kuramoto_simulation(connectome_path="...", delays=...)

    logger.info("Pipeline completed successfully.")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Run main
    run_pipeline()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
