"""
Poincare-ball hyperbolic geometry utilities.

The building blocks for hyperbolic embeddings: the exponential map at the origin (which carries
Euclidean coordinates into the open Poincare ball of curvature -1) and the Poincare distance.
Hyperbolic space, whose volume grows exponentially with radius, can hold the nested hierarchies
that scalp and inter-brain EEG embeddings occupy, which Euclidean space (linear circumference
growth) cannot. This is the geometry motivating the HEEGNet result (Li et al. 2026,
arXiv:2601.03322).
"""

# %% Import
from __future__ import annotations

import numpy as np

__all__ = [
    "poincare_distance",
    "poincare_distance_matrix",
    "poincare_exp0",
]

_MAX_NORM = 1.0 - 1e-7  # keep points strictly inside the ball so distances stay finite
_EPS = 1e-12


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def poincare_exp0(x: np.ndarray) -> np.ndarray:
    """
    Map Euclidean coordinates into the Poincare ball via the exponential map at the origin.

    For curvature -1, ``exp_0(x) = tanh(|x|) * x / |x|``, which sends any Euclidean vector into
    the open unit ball. The result is clipped to a maximum radius just below 1 so downstream
    distances remain finite.

    Parameters
    ----------
    x : np.ndarray
        Euclidean coordinates of shape ``(..., d)``.

    Returns
    -------
    np.ndarray
        Coordinates in the open Poincare ball, same shape as ``x``.

    """
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    mapped = np.where(norm > 0, np.tanh(norm) * x / np.maximum(norm, _EPS), x)
    mapped_norm = np.linalg.norm(mapped, axis=-1, keepdims=True)
    factor = np.where(mapped_norm > _MAX_NORM, _MAX_NORM / np.maximum(mapped_norm, _EPS), 1.0)
    return mapped * factor


def poincare_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Geodesic distance between points in the Poincare ball.

    ``d(x, y) = arccosh(1 + 2 |x - y|^2 / ((1 - |x|^2)(1 - |y|^2)))``.

    Parameters
    ----------
    x, y : np.ndarray
        Points in the Poincare ball of shape ``(..., d)``.

    Returns
    -------
    np.ndarray
        The geodesic distance(s).

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_sq = np.sum(x**2, axis=-1)
    y_sq = np.sum(y**2, axis=-1)
    # The closed form is valid only strictly inside the unit ball. Outside it, (1 - |x|^2) flips sign
    # and the formula silently returns a finite but wrong distance, so reject out-of-ball points loudly
    # rather than let np.maximum mask the sign flip.
    if np.any(x_sq >= 1.0) or np.any(y_sq >= 1.0):
        msg = (
            "poincare_distance requires points strictly inside the unit ball (|x| < 1); "
            "map raw coordinates through poincare_exp0 first."
        )
        raise ValueError(msg)
    sq = np.sum((x - y) ** 2, axis=-1)
    # Both factors are now positive; _EPS only floors the legitimate near-boundary case (|x| -> 1).
    denom = (1.0 - x_sq) * (1.0 - y_sq)
    return np.arccosh(1.0 + 2.0 * sq / np.maximum(denom, _EPS))


def poincare_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Pairwise Poincare distances between rows of ``points``.

    Parameters
    ----------
    points : np.ndarray
        Points of shape ``(n_points, d)`` in the Poincare ball.

    Returns
    -------
    np.ndarray
        Symmetric ``(n_points, n_points)`` distance matrix with a zero diagonal.

    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        matrix[i] = poincare_distance(points[i][np.newaxis, :], points)
    np.fill_diagonal(matrix, 0.0)
    return matrix


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
