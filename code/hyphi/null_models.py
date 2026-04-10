"""
Null models module for HyPhi: Surrogate data generation for statistical testing.

Years: 2026
"""

# %% Import
import numpy as np

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def generate_phase_randomization(signal: np.ndarray) -> np.ndarray:
    """
    Generate a phase-randomized surrogate of a signal.

    *(preserves power spectrum, destroys phase relationships)*
    """
    # Convert to frequency domain
    X = np.fft.rfft(signal)

    # Generate random phases
    random_phases = np.random.uniform(0, 2 * np.pi, len(X))
    random_phases[0] = 0  # Keep DC component real
    if len(signal) % 2 == 0:
        random_phases[-1] = 0.0  # Keep Nyquist real if even

    X_surrogate = np.abs(X) * np.exp(1j * random_phases)

    # Return to time domain
    return np.fft.irfft(X_surrogate, n=len(signal))


def generate_circular_time_shift(signal: np.ndarray, min_shift: int = 10) -> np.ndarray:
    """Generate surrogate by circularly shifting the signal in time."""
    shift = np.random.randint(min_shift, len(signal) - min_shift)
    return np.roll(signal, shift)


def generate_dyad_shuffled_null(data_matrix: np.ndarray, n_dyads: int) -> np.ndarray:
    """Generate pseudo-dyads by shuffling subjects across real dyads."""
    # Assuming data_matrix is shape (n_dyads, 2_subjects, n_timepoints)
    shuffled = data_matrix.copy()

    for i in range(n_dyads):
        # Swap subject 2 of dyad i with subject 2 of a random other dyad
        swap_idx = np.random.randint(0, n_dyads)
        if swap_idx != i:
            temp = shuffled[i, 1].copy()
            shuffled[i, 1] = shuffled[swap_idx, 1]
            shuffled[swap_idx, 1] = temp

    return shuffled


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
