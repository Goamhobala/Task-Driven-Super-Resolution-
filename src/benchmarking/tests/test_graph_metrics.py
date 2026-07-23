"""Unit tests for the APLS tile metric (benchmarking.graph_metrics).

Torch-free: only numpy/scipy/networkx/skimage, so these run in any env
(including the Stage-0 gate on a CPU login node).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from benchmarking.graph_metrics import apls_tile, mask_to_graph
from benchmarking.tile_metrics import resolve_tile_metrics

H = W = 96
PX = 10.0  # metres/pixel, 10 m GSD
TF = Affine(PX, 0.0, 0.0, 0.0, -PX, 0.0)


def hline(y=48, x0=8, x1=88, thick=3, gap=None):
    """Horizontal road ``thick`` px wide; optional (x_from, x_to) gap."""
    m = np.zeros((H, W), dtype=np.uint8)
    m[y:y + thick, x0:x1] = 1
    if gap is not None:
        m[:, gap[0]:gap[1]] = 0
    return m


def cross(thick=3):
    m = np.zeros((H, W), dtype=np.uint8)
    m[46:46 + thick, 8:88] = 1
    m[8:88, 46:46 + thick] = 1
    return m


def ring(cy=48, cx=48, rad=30, thick=2.0):
    yy, xx = np.mgrid[:H, :W]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (np.abs(d - rad) < thick).astype(np.uint8)


# ------------------------------------------------------------ mask_to_graph

def test_graph_of_line_is_single_edge():
    G = mask_to_graph(hline(), PX)
    assert G.number_of_edges() == 1
    (u, v, data), = G.edges(data=True)
    # ~80 px long road at 10 m/px -> ~800 m (skeletonize shaves the tips a bit)
    assert 600.0 < data["length"] <= 850.0
    assert tuple(int(x) for x in data["pts"][0]) == tuple(u)
    assert tuple(int(x) for x in data["pts"][-1]) == tuple(v)


def test_graph_of_cross_has_junction():
    G = mask_to_graph(cross(), PX)
    degrees = sorted(d for _, d in G.degree())
    assert degrees[-1] >= 3          # a junction exists
    assert degrees.count(1) == 4     # four arms
    assert G.number_of_edges() >= 4


def test_graph_of_ring_is_closed():
    """Pure cycle (no junctions) must be traced as a self-loop, not dropped."""
    G = mask_to_graph(ring(), PX)
    assert G.number_of_edges() >= 1
    total = sum(d["length"] for *_, d in G.edges(data=True))
    circumference = 2 * math.pi * 30 * PX
    assert 0.75 * circumference < total < 1.35 * circumference


def test_graph_empty_mask():
    G = mask_to_graph(np.zeros((H, W), np.uint8), PX)
    assert G.number_of_nodes() == 0 and G.number_of_edges() == 0


def test_spur_pruning():
    m = hline()
    m[30:49, 47:50] = 1  # a stub branching off the road (~19 px = 190 m)
    kept = mask_to_graph(m, PX, min_spur_m=30.0)
    pruned = mask_to_graph(m, PX, min_spur_m=250.0)
    assert kept.number_of_edges() > pruned.number_of_edges()
    assert pruned.number_of_edges() >= 1


# ------------------------------------------------------------------- apls

def test_apls_identity_is_one():
    for mask in (hline(), cross(), ring()):
        out = apls_tile(mask, mask, transform=TF)
        assert out["apls"] == pytest.approx(1.0)
        assert out["apls_gt_to_prop"] == pytest.approx(1.0)
        assert out["apls_prop_to_gt"] == pytest.approx(1.0)


def test_apls_penalizes_gap():
    gt = hline()
    broken = hline(gap=(40, 56))
    out = apls_tile(broken, gt, transform=TF)
    assert 0.0 <= out["apls"] < 0.9
    assert out["apls"] < apls_tile(gt, gt, transform=TF)["apls"]


def test_apls_penalizes_spurious_road():
    gt = hline()
    halluc = hline() | hline(y=20)  # an extra road GT doesn't have
    out = apls_tile(halluc, gt, transform=TF)
    assert out["apls"] < 0.9
    # the spurious road hurts the prop->gt direction specifically
    assert out["apls_prop_to_gt"] < out["apls_gt_to_prop"] + 1e-6


def test_apls_tolerates_small_offset():
    """1-px shift stays well inside the 30 m snap radius -> near-perfect."""
    out = apls_tile(hline(y=49), hline(y=48), transform=TF)
    assert out["apls"] > 0.95


def test_apls_empty_cases():
    empty = np.zeros((H, W), np.uint8)
    assert math.isnan(apls_tile(empty, empty, transform=TF)["apls"])
    assert apls_tile(empty, hline(), transform=TF)["apls"] == 0.0
    assert apls_tile(hline(), empty, transform=TF)["apls"] == 0.0


def test_apls_deterministic():
    gt, pred = cross(), hline(gap=(40, 44))
    a = apls_tile(pred, gt, transform=TF)
    b = apls_tile(pred, gt, transform=TF)
    assert a == b


def test_apls_orders_predictions_by_quality():
    """Bigger gap -> more severed shortest paths -> lower APLS."""
    gt = hline()
    small_gap = apls_tile(hline(gap=(46, 50)), gt, transform=TF)["apls"]
    big_gap = apls_tile(hline(gap=(30, 66)), gt, transform=TF)["apls"]
    assert big_gap < small_gap < 1.0


# ----------------------------------------------------------- cldice metric

def test_cldice_identity_and_gap():
    from benchmarking.skeleton_metrics import cldice_score

    assert cldice_score(hline(), hline()) == pytest.approx(1.0)
    broken = cldice_score(hline(gap=(40, 56)), hline())
    assert 0.0 < broken < 1.0
    # a bigger gap loses more GT-skeleton coverage
    assert cldice_score(hline(gap=(30, 66)), hline()) < broken


def test_cldice_empty_conventions():
    from benchmarking.skeleton_metrics import cldice_score

    empty = np.zeros((H, W), np.uint8)
    assert math.isnan(cldice_score(empty, empty))
    assert cldice_score(empty, hline()) == 0.0
    assert cldice_score(hline(), empty) == 0.0


def test_cldice_plugin_contract():
    (plugin,) = resolve_tile_metrics(["cldice"])
    grid = [(f"t_r{ri}_c{ci}", ri, ci, ri * 48, ci * 48, 48, 48)
            for ri in range(2) for ci in range(2)]
    res = plugin(hline().astype(bool), hline(), transform=TF, tile_id="t", grid=grid)
    assert res.tile["cldice"] == pytest.approx(1.0)
    assert math.isnan(res.chips["t_r0_c0"]["cldice"])       # road-free chip
    assert res.chips["t_r1_c1"]["cldice"] == pytest.approx(1.0)


# ---------------------------------------------------------------- plugin

def _quad_grid(size=48):
    """2x2 footprint grid over the H x W test canvas, runner-style tuples."""
    return [(f"t_r{ri}_c{ci}", ri, ci, ri * size, ci * size, size, size)
            for ri in range(2) for ci in range(2)]


def test_plugin_contract_tile_and_chips():
    (plugin,) = resolve_tile_metrics(["apls"])
    res = plugin(hline().astype(bool), hline(), transform=TF,
                 tile_id="t", grid=_quad_grid())
    # tile row: headline + diagnostics
    for col in ("apls", "apls_gt_to_prop", "apls_prop_to_gt",
                "gt_graph_nodes", "prop_graph_edges"):
        assert col in res.tile
    assert res.tile["apls"] == pytest.approx(1.0)
    # chip rows: one value per grid cell, keyed by chip_id (the pairing unit)
    assert set(res.chips) == {"t_r0_c0", "t_r0_c1", "t_r1_c0", "t_r1_c1"}
    # hline(y=48) lies in the bottom grid row: top chips road-free -> NaN
    assert math.isnan(res.chips["t_r0_c0"]["apls"])
    assert math.isnan(res.chips["t_r0_c1"]["apls"])
    # identity prediction -> perfect within each road-bearing chip
    assert res.chips["t_r1_c0"]["apls"] == pytest.approx(1.0)
    assert res.chips["t_r1_c1"]["apls"] == pytest.approx(1.0)


def test_plugin_chip_localises_gap():
    """A gap confined to one chip must show up in THAT chip's score only."""
    (plugin,) = resolve_tile_metrics(["apls"])
    res = plugin(hline(gap=(60, 70)).astype(bool), hline(), transform=TF,
                 tile_id="t", grid=_quad_grid())
    left, right = res.chips["t_r1_c0"]["apls"], res.chips["t_r1_c1"]["apls"]
    assert left == pytest.approx(1.0)   # intact half
    assert right < 0.9                  # severed half
    # tile-level also degraded (cross-chip paths break too)
    assert res.tile["apls"] < 1.0


def test_plugin_no_grid_gives_tile_only():
    (plugin,) = resolve_tile_metrics(["apls"])
    res = plugin(hline().astype(bool), hline(), transform=TF,
                 tile_id="t", grid=None)
    assert res.chips is None and res.tile["apls"] == pytest.approx(1.0)
