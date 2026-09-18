"""Eligibility filter: the GT decides which units the graph metrics are scored on.

These lock the behaviour the filter exists to fix, so the motivating asymmetry is
regression-tested rather than only described in a docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.graph_metrics import apls_tile, mask_to_graph
from benchmarking.gt_eligibility import (
    apply_eligibility,
    eligible_ids,
    is_graph_metric,
)


class _T:
    a = 2.5  # 2.5 m GSD, the bench's GT resolution


def _line_mask(h=64, w=64, row=32, c0=2, c1=62, thick=1):
    m = np.zeros((h, w), dtype=np.uint8)
    m[row:row + thick, c0:c1] = 1
    return m


# --------------------------------------------------------------------------- #
# the asymmetry this filter closes
# --------------------------------------------------------------------------- #
def test_gt_empty_chip_scores_nan_for_silent_arm_and_zero_for_hallucinating():
    """The motivating bug: on a GT-empty chip the two arms land in different
    populations — NaN leaves the paired stats, 0.0 stays in."""
    gt = np.zeros((64, 64), dtype=np.uint8)
    silent = np.zeros((64, 64), dtype=bool)
    hallucinating = _line_mask().astype(bool)

    assert np.isnan(apls_tile(silent, gt, transform=_T())["apls"])
    assert apls_tile(hallucinating, gt, transform=_T())["apls"] == 0.0


def test_eligible_chip_is_never_nan_for_any_arm():
    """The guarantee the filter buys: with a non-empty GT graph, every arm gets a
    number, so n is identical across arms."""
    gt = _line_mask()
    assert mask_to_graph(gt, 2.5).number_of_edges() > 0
    for pred in (np.zeros((64, 64), dtype=bool),        # silent
                 _line_mask().astype(bool),             # correct
                 _line_mask(row=10).astype(bool)):      # displaced
        assert not np.isnan(apls_tile(pred, gt, transform=_T())["apls"])


def test_road_pixels_are_not_eligibility():
    """A blob whose skeleton is a sub-min_spur_m stub has road pixels but no
    graph — which is why eligibility keys on edges, not on the pixel count."""
    gt = _line_mask(c0=20, c1=28)          # 8 px = 20 m < the 30 m spur floor
    assert gt.sum() > 0
    assert mask_to_graph(gt, 2.5).number_of_edges() == 0
    assert np.isnan(apls_tile(np.zeros_like(gt, dtype=bool), gt, transform=_T())["apls"])


# --------------------------------------------------------------------------- #
# applying the lookup
# --------------------------------------------------------------------------- #
@pytest.fixture
def elig():
    return pd.DataFrame([
        {"unit": "chip", "chip_id": "t_r0_c0", "tile_id": "t", "gt_graph_edges": 4},
        {"unit": "chip", "chip_id": "t_r0_c1", "tile_id": "t", "gt_graph_edges": 0},
        {"unit": "tile", "chip_id": "t", "tile_id": "t", "gt_graph_edges": 4},
    ])


@pytest.fixture
def chips():
    return pd.DataFrame([
        {"model_name": m, "chip_id": c, "apls": v, "f1": v}
        for m, c, v in [("a", "t_r0_c0", 0.8), ("a", "t_r0_c1", float("nan")),
                        ("b", "t_r0_c0", 0.7), ("b", "t_r0_c1", 0.0)]
    ])


def test_filter_drops_ineligible_chips_for_graph_metrics(chips, elig):
    out, note = apply_eligibility(chips, elig, "apls", unit="chip", scope="graph")
    assert set(out["chip_id"]) == {"t_r0_c0"}
    assert "1 of 2 chips dropped" in note
    # and n is now the same for both arms, which was the whole point
    assert out.groupby("model_name")["apls"].count().nunique() == 1


def test_graph_scope_leaves_pixel_metrics_alone(chips, elig):
    out, note = apply_eligibility(chips, elig, "f1", unit="chip", scope="graph")
    assert len(out) == len(chips) and note is None


def test_all_scope_filters_pixel_metrics_too(chips, elig):
    out, note = apply_eligibility(chips, elig, "f1", unit="chip", scope="all")
    assert set(out["chip_id"]) == {"t_r0_c0"}
    assert note is not None


def test_no_lookup_is_a_no_op(chips):
    out, note = apply_eligibility(chips, None, "apls", unit="chip", scope="all")
    assert out is chips and note is None


def test_tile_unit_keys_on_the_renamed_id(elig):
    """The store's tile table arrives with tile_id renamed to chip_id."""
    tiles = pd.DataFrame([{"model_name": "a", "chip_id": "t", "apls": 0.8},
                          {"model_name": "a", "chip_id": "u", "apls": 0.0}])
    out, _ = apply_eligibility(tiles, elig, "apls", unit="tile", scope="graph")
    assert set(out["chip_id"]) == {"t"}          # 'u' is absent from the lookup
    assert eligible_ids(elig, "tile") == {"t"}


def test_unknown_ids_are_dropped_not_kept(elig):
    """An id the GT pass never produced cannot be shown to be eligible."""
    df = pd.DataFrame([{"model_name": "a", "chip_id": "ghost", "apls": 0.9}])
    out, _ = apply_eligibility(df, elig, "apls", unit="chip", scope="graph")
    assert out.empty


def test_graph_metric_names():
    assert is_graph_metric("apls") and is_graph_metric("cldice")
    assert not is_graph_metric("f1") and not is_graph_metric("buffered_f1_r3")


def test_missing_unit_column_is_rejected():
    bad = pd.DataFrame([{"chip_id": "x", "gt_graph_edges": 1}])
    with pytest.raises(ValueError, match="unit"):
        eligible_ids(bad, "chip")
