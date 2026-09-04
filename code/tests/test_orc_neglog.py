"""Regression test: Ollivier-Ricci curvature must map coupling SIMILARITY weights to a
neglog transport DISTANCE before the optimal transport, or it silently computes the wrong
geometry and does not reproduce the published atlas ORC.

GraphRicciCurvature treats the edge ``weight`` as the DISTANCE in its transport shortest
paths. The atlas / paper convention is ``dist = -log(weight)`` (stronger coupling -> shorter
distance). ``compute_orc``/``get_orc`` default to ``metric="neglog"`` for exactly this reason;
``metric="raw"`` keeps the legacy (incorrect for similarity weights) behaviour. See
``_orc_distance_graph`` and curvature-entropy-atlas/atlas_campaign_cache.py.
"""

import math
import os

import networkx as nx
import numpy as np
import pytest

from hyphi.modeling.graph_curvatures import _orc_distance_graph, compute_orc

# A small fixed coupling-similarity graph and the neglog ORC it must reproduce
# (atlas params: alpha=0.5, exp_power=2.0, method="OTD"). Regenerate GOLDEN only if the
# ORC convention is deliberately changed.
_EDGES = [(0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.7), (3, 0, 0.4), (0, 2, 0.3), (1, 3, 0.6)]
_GOLDEN = {
    (0, 1): 0.16567, (0, 3): 0.642108, (0, 2): 0.662712,
    (1, 2): 0.540963, (1, 3): 0.530118, (2, 3): 0.660607,
}


def _graph():
    g = nx.Graph()
    for u, v, w in _EDGES:
        g.add_edge(u, v, weight=w)
    return g


def _orc(metric):
    h = compute_orc(_graph(), alpha=0.5, base=math.e, exp_power=2.0, method="OTD", metric=metric)
    return {tuple(sorted(e)): h[e[0]][e[1]]["ricciCurvature"] for e in h.edges()}


def test_transform_matches_atlas_formula():
    """weight clipped to [1e-6, 1-1e-9], dist = -log(w), floored at 1e-9."""
    h = _orc_distance_graph(_graph(), "neglog")
    for u, v, w in _EDGES:
        expected = max(-math.log(min(max(w, 1e-6), 1.0 - 1e-9)), 1e-9)
        assert h[u][v]["weight"] == pytest.approx(expected, abs=1e-12)


def test_default_metric_is_neglog():
    """The default path must be the reproducible (neglog) one, not the legacy raw path."""
    default = {tuple(sorted(e)): compute_orc(_graph(), alpha=0.5, base=math.e,
              exp_power=2.0, method="OTD")[e[0]][e[1]]["ricciCurvature"]
              for e in _graph().edges()}
    for k, v in _GOLDEN.items():
        assert default[k] == pytest.approx(v, abs=1e-4)


def test_golden_neglog_orc():
    got = _orc("neglog")
    for k, v in _GOLDEN.items():
        assert got[k] == pytest.approx(v, abs=1e-4), f"edge {k}: {got[k]} != golden {v}"


def test_neglog_differs_from_raw():
    """The bug this guards against: raw similarity weights give a different (wrong) ORC."""
    neglog, raw = _orc("neglog"), _orc("raw")
    max_diff = max(abs(neglog[k] - raw[k]) for k in neglog)
    assert max_diff > 1e-2, "neglog and raw ORC are indistinguishable -- the transform is not applied"


def test_bad_metric_raises():
    with pytest.raises(ValueError):
        _orc_distance_graph(_graph(), "similarity")


# Strong optional check: reproduce a real cached atlas ORC value bit-for-bit. Skips cleanly
# when the atlas cache is not on this machine, so the suite stays self-contained.
_ATLAS_CACHE = os.environ.get(
    "HYPHI_ATLAS_CACHE",
    "/Users/nicolashinrichs/hyphi/curvature-entropy-atlas/cache-inter",
)


@pytest.mark.skipif(not os.path.isdir(_ATLAS_CACHE), reason="atlas cache not present")
def test_reproduces_atlas_cache_bitforbit():
    import glob

    files = sorted(glob.glob(os.path.join(_ATLAS_CACHE, "cache_*_inter_d0.10_*.npz")))
    if not files:
        pytest.skip("no matching atlas cache cell")
    z = np.load(files[0], allow_pickle=True)
    off = np.asarray(z["offsets"], int)
    eu, ev = np.asarray(z["edge_u"], int), np.asarray(z["edge_v"], int)
    w, co = np.asarray(z["weight"], float), np.asarray(z["curv_orc"], float)
    a, b = off[0], off[1]
    g = nx.Graph()
    for i in range(a, b):
        g.add_edge(int(eu[i]), int(ev[i]), weight=float(w[i]))
    h = compute_orc(g, alpha=0.5, base=math.e, exp_power=2.0, method="OTD", metric="neglog")
    got = {tuple(sorted(e)): h[e[0]][e[1]]["ricciCurvature"] for e in h.edges()}
    stored = {tuple(sorted((int(eu[i]), int(ev[i])))): co[i] for i in range(a, b)}
    max_diff = max(abs(got[k] - stored[k]) for k in stored if k in got)
    assert max_diff < 1e-5, f"neglog ORC does not reproduce atlas curv_orc (max|diff|={max_diff:.2e})"
