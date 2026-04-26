"""
This module contains the function for transformation of Foreman-Ricci curvatures
for use in Graph Diffusion Distance computation, and related helpers. 
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def as_1d_float_array(values):
    """
    Convert an array-like or an edge->value dict into a 1D float NumPy array.
    """
    if isinstance(values, dict):
        values = list(values.values())

    arr = np.asarray(values, dtype=float).ravel()

    if arr.size == 0:
        raise ValueError("Received an empty collection of values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("All values must be finite.")

    return arr


def fit_global_positive_linear_transform(
    curvature_dict,
    *,
    flip_sign=True,
    eps=1e-9,
    rescale=None,
):
    """
    Fit ONE global linear transform from the pooled curvatures of all matrices.

    Transform:
        transformed = scale * sign * curvature + shift

    Parameters
    ----------
    curvature_dict : dict
        Dict keyed by matrix index. Each value can be:
        - an array/list of curvature values, or
        - a dict like {(u, v): curvature}
    flip_sign : bool, default=True
        If True, use sign = -1.
        If False, use sign = +1.
    eps : float, default=1e-9
        Used only when rescale is None, so the minimum transformed value is
        strictly positive instead of exactly zero.
    rescale : None or tuple(float, float), default=(1e-6, 1.0)
        If None:
            use scale = 1 and choose the smallest shift that makes all
            transformed values strictly positive. 
        If (low, high):
            map the globally signed curvatures linearly into [low, high].
            Requires 0 < low < high. Rescaling is not appropriate for
            transforming Forman-Ricci curvatures!!! 

    Returns
    -------
    params : dict
        Fitted parameters and some diagnostics.
    """
    if not isinstance(curvature_dict, dict) or len(curvature_dict) == 0:
        raise ValueError("curvature_dict must be a non-empty dictionary.")

    sign = -1.0 if flip_sign else 1.0

    pooled = np.concatenate(
        [as_1d_float_array(values) for values in curvature_dict.values()]
    )
    signed = sign * pooled

    signed_min = float(np.min(signed))
    signed_max = float(np.max(signed))

    if signed_max == signed_min:
        raise ValueError(
            "All pooled curvature values are identical; cannot fit a meaningful transform."
        )

    if rescale is None:
        scale = 1.0
        shift = -signed_min + float(eps)
    else:
        if not isinstance(rescale, (tuple, list)) or len(rescale) != 2:
            raise ValueError("rescale must be None or a tuple/list (low, high).")

        low, high = map(float, rescale)
        if not (0.0 < low < high):
            raise ValueError("rescale=(low, high) must satisfy 0 < low < high.")

        scale = (high - low) / (signed_max - signed_min)
        shift = low - scale * signed_min

    transformed = scale * signed + shift

    return {
        "sign": sign,
        "scale": float(scale),
        "shift": float(shift),
        "raw_global_min": float(np.min(pooled)),
        "raw_global_max": float(np.max(pooled)),
        "signed_global_min": signed_min,
        "signed_global_max": signed_max,
        "transformed_global_min": float(np.min(transformed)),
        "transformed_global_max": float(np.max(transformed)),
    }


def apply_linear_transform(values, *, sign, scale, shift):
    """
    Apply transformed = scale * sign * value + shift.

    Preserves input container type:
    - dict -> dict with same keys
    - array-like -> NumPy array
    """
    if isinstance(values, dict):
        return {
            key: float(scale * sign * value + shift)
            for key, value in values.items()
        }

    arr = np.asarray(values, dtype=float)
    return scale * sign * arr + shift


def transform_curvature_collection(curvature_dict, params):
    """
    Apply one fitted transform to every entry in a curvature dictionary.
    """
    return {
        key: apply_linear_transform(
            values,
            sign=params["sign"],
            scale=params["scale"],
            shift=params["shift"],
        )
        for key, values in curvature_dict.items()
    }


def check_all_positive(transformed_dict):
    """
    Return per-matrix positivity diagnostics.
    """
    summary = {}
    for key, values in transformed_dict.items():
        arr = as_1d_float_array(values)
        summary[key] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "all_positive": bool(np.all(arr > 0)),
        }
    return summary


def attach_edge_weights_to_graph(graph, edge_weight_dict, attr_name="positive_weight"):
    """
    Copy a graph and attach transformed edge weights as a new edge attribute.
    """
    H = graph.copy()
    nx.set_edge_attributes(H, edge_weight_dict, name=attr_name)
    return H
