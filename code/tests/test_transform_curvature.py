"""
Known-Answer Tests for transform_curvature helpers.

These helpers fit a single global linear transform over a collection of
signed Forman-Ricci curvature values and apply it so that every output
weight is strictly positive.  All functions are pure / numeric, so exact
expected values are derived from the math.
"""

# %% Import
import networkx as nx
import numpy as np
import pytest
from hyphi.modeling.transform_curvature import (
    apply_linear_transform,
    as_1d_float_array,
    attach_edge_weights_to_graph,
    check_all_positive,
    fit_global_positive_linear_transform,
    transform_curvature_collection,
)

# %% Named constants (avoid bare magic literals in comparisons) o >><< o >><< o

_N_EDGES_K5 = 10  # C(5,2) edges in K_5
_N_EDGES_STAR = 5  # 5 leaves in star_graph_s6
_RESCALE_LOW = 0.1
_RESCALE_HIGH = 1.0
_EPS_STRICT = 1e-9  # default eps used by fit_global_positive_linear_transform
_ABS_TOL = 1e-10  # tolerance for float comparisons

# %% Tests o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><


class TestAs1dFloatArray:
    """as_1d_float_array must flatten and cast to float64."""

    def test_list_becomes_1d(self):
        """A nested list is flattened to a 1-D float array."""
        result = as_1d_float_array([[1, 2], [3, 4]])
        assert result.ndim == 1
        assert result.dtype == np.float64
        assert result.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_dict_values_extracted(self):
        """When given a dict the values are extracted and flattened."""
        data = {(0, 1): -2.0, (1, 2): 0.5, (2, 3): 3.0}
        result = as_1d_float_array(data)
        assert result.ndim == 1
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(sorted(result), [-2.0, 0.5, 3.0])

    def test_numpy_array_passthrough(self):
        """A 2-D NumPy array is ravel-ed to 1-D float."""
        arr = np.array([[1, 2], [3, 4]], dtype=int)
        result = as_1d_float_array(arr)
        assert result.ndim == 1
        assert result.dtype == np.float64

    def test_empty_raises(self):
        """An empty collection raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            as_1d_float_array([])

    def test_non_finite_raises(self):
        """An array containing NaN raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            as_1d_float_array([1.0, float("nan"), 3.0])

    def test_inf_raises(self):
        """An array containing Inf raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            as_1d_float_array([1.0, float("inf")])


class TestFitGlobalPositiveLinearTransform:
    """fit_global_positive_linear_transform must return strictly positive outputs."""

    def _simple_dict(self):
        """Toy curvature dict: two graphs with different signed ranges."""
        return {
            0: [-4.0, -4.0, -4.0],  # K5-like Forman curvatures
            1: [0.0, 0.0, 0.0, 0.0],  # C10-like Forman curvatures
        }

    def test_returns_required_keys(self):
        """The returned dict must contain sign, scale, shift, and diagnostics."""
        params = fit_global_positive_linear_transform(self._simple_dict())
        for key in ("sign", "scale", "shift", "transformed_global_min", "transformed_global_max"):
            assert key in params

    def test_default_scale_is_one(self):
        """With rescale=None the scale is always 1."""
        params = fit_global_positive_linear_transform(self._simple_dict())
        assert params["scale"] == 1.0

    def test_output_strictly_positive_no_rescale(self):
        """With rescale=None all transformed values must be > 0."""
        params = fit_global_positive_linear_transform(self._simple_dict())
        assert params["transformed_global_min"] > 0

    def test_output_in_range_with_rescale(self):
        """With rescale=(low, high) the output range must be [low, high]."""
        params = fit_global_positive_linear_transform(self._simple_dict(), rescale=(_RESCALE_LOW, _RESCALE_HIGH))
        assert abs(params["transformed_global_min"] - _RESCALE_LOW) < _ABS_TOL
        assert abs(params["transformed_global_max"] - _RESCALE_HIGH) < _ABS_TOL

    def test_flip_sign_false(self):
        """With flip_sign=False the sign field is +1."""
        params = fit_global_positive_linear_transform({0: [1.0, 2.0, 3.0]}, flip_sign=False)
        assert params["sign"] == 1.0

    def test_flip_sign_true(self):
        """With flip_sign=True (default) the sign field is -1."""
        params = fit_global_positive_linear_transform({0: [1.0, 2.0, 3.0]})
        assert params["sign"] == -1.0

    def test_identical_values_raises(self):
        """All-identical curvatures make fitting impossible; must raise ValueError."""
        with pytest.raises(ValueError, match="identical"):
            fit_global_positive_linear_transform({0: [1.0, 1.0], 1: [1.0]})

    def test_empty_dict_raises(self):
        """An empty dict must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            fit_global_positive_linear_transform({})

    def test_invalid_rescale_raises(self):
        """Rescale with low >= high must raise ValueError."""
        with pytest.raises(ValueError, match="0 < low < high"):
            fit_global_positive_linear_transform({0: [1.0, 2.0]}, rescale=(0.9, 0.1))

    def test_rescale_zero_low_raises(self):
        """Rescale with low=0 violates 0 < low < high and must raise."""
        with pytest.raises(ValueError, match="0 < low < high"):
            fit_global_positive_linear_transform({0: [1.0, 2.0]}, rescale=(0.0, 1.0))

    def test_shift_math_no_rescale(self):
        """Verify shift arithmetic for the no-rescale path.

        With flip_sign=True and rescale=None:
          sign = -1, scale = 1
          signed = -pooled
          signed_min = min(-pooled) = -max(pooled)
          shift = -signed_min + eps = max(pooled) + eps
        For pooled = [-4, 0] -> signed = [4, 0] -> signed_min = 0
          shift = 0 + eps; transformed_min = 0 + eps > 0.
        """
        params = fit_global_positive_linear_transform({0: [-4.0, 0.0]}, flip_sign=True, eps=_EPS_STRICT)
        expected_shift = 0.0 + _EPS_STRICT  # -signed_min + eps = -0 + eps
        assert abs(params["shift"] - expected_shift) < _ABS_TOL
        assert params["transformed_global_min"] > 0


class TestApplyLinearTransform:
    """apply_linear_transform must behave as an affine map."""

    _SIGN = 1.0
    _SCALE = 2.0
    _SHIFT = 3.0

    def test_array_affine_map(self):
        """Transformed = scale * sign * x + shift for each element."""
        x = np.array([0.0, 1.0, 2.0])
        result = apply_linear_transform(x, sign=self._SIGN, scale=self._SCALE, shift=self._SHIFT)
        expected = self._SCALE * self._SIGN * x + self._SHIFT
        np.testing.assert_array_almost_equal(result, expected)

    def test_dict_affine_map(self):
        """Dict input returns dict with same keys and affine-transformed values."""
        data = {(0, 1): 1.0, (1, 2): -1.0}
        result = apply_linear_transform(data, sign=self._SIGN, scale=self._SCALE, shift=self._SHIFT)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(data.keys())
        for key, val in data.items():
            expected = self._SCALE * self._SIGN * val + self._SHIFT
            assert abs(result[key] - expected) < _ABS_TOL

    def test_flip_sign_negates(self):
        """sign=-1 negates before scaling and shifting."""
        x = np.array([2.0])
        result = apply_linear_transform(x, sign=-1.0, scale=1.0, shift=0.0)
        np.testing.assert_almost_equal(result, [-2.0])

    def test_identity_transform(self):
        """scale=1, sign=1, shift=0 is the identity."""
        x = np.array([1.0, -3.0, 7.5])
        result = apply_linear_transform(x, sign=1.0, scale=1.0, shift=0.0)
        np.testing.assert_array_almost_equal(result, x)

    def test_roundtrip_with_fit(self):
        """Fitted transform applied to raw data must reproduce stored diagnostics.

        The minimum and maximum of the re-applied transform must equal
        params['transformed_global_min'] and params['transformed_global_max'].
        """
        raw = {0: [-4.0, -2.0, 0.0], 1: [1.0, 3.0]}
        params = fit_global_positive_linear_transform(raw, flip_sign=True)
        transformed = {
            k: apply_linear_transform(v, sign=params["sign"], scale=params["scale"], shift=params["shift"])
            for k, v in raw.items()
        }
        all_vals = np.concatenate([np.asarray(v, dtype=float) for v in transformed.values()])
        assert abs(float(np.min(all_vals)) - params["transformed_global_min"]) < _ABS_TOL
        assert abs(float(np.max(all_vals)) - params["transformed_global_max"]) < _ABS_TOL


class TestTransformCurvatureCollection:
    """transform_curvature_collection applies one fitted transform to every entry."""

    def test_all_keys_preserved(self):
        """Every key in the input dict must appear in the output."""
        raw = {0: [-3.0, -1.0], 1: [0.0, 2.0], 2: [-5.0]}
        params = fit_global_positive_linear_transform(raw)
        result = transform_curvature_collection(raw, params)
        assert set(result.keys()) == set(raw.keys())

    def test_output_all_positive(self):
        """After transformation every value must be strictly positive."""
        raw = {0: [-4.0, -4.0, -4.0], 1: [0.0, 0.0]}
        params = fit_global_positive_linear_transform(raw)
        result = transform_curvature_collection(raw, params)
        for values in result.values():
            arr = np.asarray(list(values.values()) if isinstance(values, dict) else values, dtype=float)
            assert np.all(arr > 0)

    def test_dict_values_survive(self):
        """Edge-keyed dicts (the common curvature format) are preserved."""
        raw = {
            0: {(0, 1): -2.0, (1, 2): 0.0, (2, 3): 1.0},
        }
        params = fit_global_positive_linear_transform(raw)
        result = transform_curvature_collection(raw, params)
        assert isinstance(result[0], dict)
        assert set(result[0].keys()) == set(raw[0].keys())


class TestCheckAllPositive:
    """check_all_positive returns per-matrix positivity diagnostics."""

    def test_all_positive_true(self):
        """A dict of all-positive arrays should have all_positive=True."""
        data = {0: [1.0, 2.0, 3.0], 1: [0.5, 4.0]}
        summary = check_all_positive(data)
        assert summary[0]["all_positive"] is True
        assert summary[1]["all_positive"] is True

    def test_contains_zero_is_not_all_positive(self):
        """Zero is not strictly positive; all_positive must be False."""
        data = {0: [0.0, 1.0, 2.0]}
        summary = check_all_positive(data)
        assert summary[0]["all_positive"] is False

    def test_contains_negative_is_not_all_positive(self):
        """A negative value makes all_positive False."""
        data = {0: [-0.01, 1.0]}
        summary = check_all_positive(data)
        assert summary[0]["all_positive"] is False

    def test_min_max_reported_correctly(self):
        """The reported min and max must match np.min / np.max of the input."""
        values = [3.0, 7.0, 1.0, 5.0]
        summary = check_all_positive({0: values})
        assert abs(summary[0]["min"] - 1.0) < _ABS_TOL
        assert abs(summary[0]["max"] - 7.0) < _ABS_TOL

    def test_keys_preserved(self):
        """Every input key must appear in the summary."""
        data = {0: [1.0], 1: [2.0], 2: [3.0]}
        summary = check_all_positive(data)
        assert set(summary.keys()) == {0, 1, 2}


class TestAttachEdgeWeightsToGraph:
    """attach_edge_weights_to_graph copies the graph and sets edge attributes."""

    def test_edge_weights_set(self, complete_graph_k5):
        """Weights must appear as the named edge attribute on every edge."""
        edges = list(complete_graph_k5.edges())
        weights = {e: float(i + 1) for i, e in enumerate(edges)}
        H = attach_edge_weights_to_graph(complete_graph_k5, weights)
        attrs = nx.get_edge_attributes(H, "positive_weight")
        assert len(attrs) == _N_EDGES_K5
        for edge, w in weights.items():
            # networkx may store the edge as (u,v) or (v,u); try both
            stored = attrs.get(edge) or attrs.get((edge[1], edge[0]))
            assert stored is not None
            assert abs(stored - w) < _ABS_TOL

    def test_original_graph_unchanged(self, complete_graph_k5):
        """The function must return a copy; the original must not be mutated."""
        edges = list(complete_graph_k5.edges())
        weights = dict.fromkeys(edges, 1.0)
        attach_edge_weights_to_graph(complete_graph_k5, weights)
        assert not nx.get_edge_attributes(complete_graph_k5, "positive_weight")

    def test_custom_attr_name(self, star_graph_s6):
        """A custom attr_name must be used instead of the default."""
        edges = list(star_graph_s6.edges())
        weights = dict.fromkeys(edges, 0.5)
        H = attach_edge_weights_to_graph(star_graph_s6, weights, attr_name="w_custom")
        assert nx.get_edge_attributes(H, "w_custom")
        assert not nx.get_edge_attributes(H, "positive_weight")

    def test_edge_count_preserved(self, ring_lattice_c10):
        """The copied graph must have the same edge count as the original."""
        edges = list(ring_lattice_c10.edges())
        weights = dict.fromkeys(edges, 1.0)
        H = attach_edge_weights_to_graph(ring_lattice_c10, weights)
        assert H.number_of_edges() == ring_lattice_c10.number_of_edges()
        assert H.number_of_nodes() == ring_lattice_c10.number_of_nodes()


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
