"""Tests for hyphi.configs helpers (the path-anchoring used by the experiment scripts)."""

from pathlib import Path

from hyphi.configs import resolve_loc_paths


def test_resolve_loc_paths_anchors_relative_loc_keys_on_root():
    """Relative ``*_loc`` values become absolute under the given root."""
    config = {"result_loc": "./results/ccorr", "data_loc": "data/adj", "config_id": "ccorr_001"}

    resolved = resolve_loc_paths(config, root="/proj")

    assert resolved["result_loc"] == str(Path("/proj/results/ccorr"))
    assert resolved["data_loc"] == str(Path("/proj/data/adj"))
    # Non-_loc keys are untouched.
    assert resolved["config_id"] == "ccorr_001"


def test_resolve_loc_paths_leaves_absolute_paths_and_is_idempotent():
    """Absolute ``*_loc`` values are left unchanged, and re-applying is a no-op."""
    config = {"result_loc": str(Path("/already/absolute/results")), "pooled_result_loc": "./pooled"}

    once = resolve_loc_paths(dict(config), root="/proj")
    assert once["result_loc"] == str(Path("/already/absolute/results"))
    assert once["pooled_result_loc"] == str(Path("/proj/pooled"))

    twice = resolve_loc_paths(dict(once), root="/proj")
    assert twice == once  # idempotent


def test_resolve_loc_paths_returns_the_same_dict_object():
    """The dict is anchored in place and returned (callers can wrap load_config directly)."""
    config = {"kuramoto_result_loc": "./results/kuramoto"}

    result = resolve_loc_paths(config, root="/proj")

    assert result is config
    assert result["kuramoto_result_loc"] == str(Path("/proj/results/kuramoto"))


def test_resolve_loc_paths_ignores_non_string_loc_values():
    """A non-string ``*_loc`` value (e.g. a list) is left as-is rather than crashing."""
    config = {"extra_loc": ["a", "b"], "result_loc": "./results"}

    resolved = resolve_loc_paths(config, root="/proj")

    assert resolved["extra_loc"] == ["a", "b"]
    assert resolved["result_loc"] == str(Path("/proj/results"))
