"""Paired occlusion contrast — the parts of a triptych that have a right answer.

The figure is looked at, not tabulated, so the bookkeeping underneath it has to
be pinned somewhere else: that the generalised pass is still `saliency.py`'s
pass, that the margin readout reads the margin, that a pair resolves to the
seeds it claims, and that the two normalisations are the two things the module
docstring says they are.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sr.probes.occl_contrast import (map_stem, margin_readout, normalise_pair,
                                     occl_field, parse_pair, peak_of,
                                     pixel_readout, resolve_pair, resolve_side)
from sr.probes.occl_context_extract import margin_of
from sr.probes.saliency import saliency_maps


class StubModel:
    def __init__(self, c=4):
        self.band_mean = torch.zeros(1, c, 1, 1)


class StubDecoder:
    """logits(y)[b, i, j] = weighted sum of the bands AT (i, j).

    Purely local, so a sensitivity map must be non-zero exactly where a window
    covering the read-out support sits — an analytic answer for the
    accumulate-and-divide, the same stub `test_saliency` uses.
    """

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        self.m = StubModel(len(weights))
        self.w = torch.tensor(weights).view(1, -1, 1, 1)

    def logits(self, y):
        return (y * self.w).sum(dim=1)


def _y(c=4, h=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(c, h, h, generator=g) + 1.0        # strictly positive


# ------------------------------------------- the generalised pass IS the old one
def test_pixel_readout_reproduces_saliency_maps_exactly():
    """The whole point of generalising rather than copying: one pass, two
    targets. A drift here means the margin figures and the saliency figures
    stopped being the same instrument."""
    dec, y = StubDecoder(), _y()
    pix = [5 * 32 + 7, 20 * 32 + 19]
    ref, l0_ref = saliency_maps(dec, y, pix, 8, 4, 8, torch)
    got, r0 = occl_field(dec, y, pixel_readout(pix, y, torch), 8, 4, 8, torch)
    assert np.allclose(got, ref)
    assert np.allclose(r0, l0_ref)


def test_band_fill_touches_only_that_band():
    dec, y = StubDecoder(weights=(1.0, 0.0, 0.0, 0.0)), _y()
    pix = [16 * 32 + 16]
    r_band, _ = occl_field(dec, y, pixel_readout(pix, y, torch), 8, 4, 8, torch,
                           band=0)
    r_all, _ = occl_field(dec, y, pixel_readout(pix, y, torch), 8, 4, 8, torch)
    # Only band 0 carries weight, so flattening it alone must match flattening
    # everything; a band the model ignores must change nothing.
    assert np.allclose(r_band, r_all)
    r_null, _ = occl_field(dec, y, pixel_readout(pix, y, torch), 8, 4, 8, torch,
                           band=2)
    assert np.allclose(r_null, 0.0)


def test_batching_does_not_change_the_map():
    dec, y = StubDecoder(), _y()
    pix = [10 * 32 + 10]
    a, _ = occl_field(dec, y, pixel_readout(pix, y, torch), 8, 4, 1, torch)
    b, _ = occl_field(dec, y, pixel_readout(pix, y, torch), 8, 4, 7, torch)
    assert np.allclose(a, b)


# --------------------------------------------------------------- the margin target
def test_margin_readout_is_the_suites_margin():
    dec, y = StubDecoder(), _y()
    road = torch.zeros(32, 32, dtype=torch.bool)
    road[4:9, 4:9] = True
    lg = dec.logits(y.unsqueeze(0))
    got = margin_readout(road, torch)(lg)
    assert got.shape == (1, 1)
    assert torch.allclose(got[:, 0], margin_of(lg, road))


def test_margin_map_is_one_map_and_is_local():
    """One map per chip, no pixel to choose — and for a purely local model the
    only locations that can move the margin are the ones a window covers."""
    dec, y = StubDecoder(), _y()
    road = torch.zeros(32, 32, dtype=torch.bool)
    road[4:9, 4:9] = True
    maps, r0 = occl_field(dec, y, margin_readout(road, torch), 8, 4, 8, torch)
    assert maps.shape == (1, 32, 32)
    assert r0.shape == (1,)
    # Every location is covered by at least one window, and the model is
    # non-degenerate, so nothing should be identically zero everywhere.
    assert np.abs(maps).sum() > 0


def test_margin_map_responds_to_the_road_region():
    """Occluding ON the road must cost margin for a model that reads the road
    pixels positively; the sign convention is intact − occluded."""
    dec = StubDecoder(weights=(1.0, 1.0, 1.0, 1.0))
    y = torch.full((4, 32, 32), 1.0)
    road = torch.zeros(32, 32, dtype=torch.bool)
    road[12:20, 12:20] = True
    y[:, 12:20, 12:20] = 5.0                     # the road is bright
    maps, _ = occl_field(dec, y, margin_readout(road, torch), 8, 8, 8, torch)
    assert maps[0][14, 14] > 0                   # on road: removing it costs
    assert maps[0][2, 2] < 0                     # off road: removing it helps


# ------------------------------------------------------------------- pair specs
def test_parse_pair_with_and_without_seeds():
    assert parse_pair("r2a:r1a") == (("r2a", None), ("r1a", None))
    assert parse_pair("r4b.222:r3b.444") == (("r4b", 222), ("r3b", 444))


@pytest.mark.parametrize("bad", ["r2a", "r2a:r1a:r0", ":r1a", "r2a.x:r1a"])
def test_parse_pair_rejects_malformed(bad):
    with pytest.raises(SystemExit):
        parse_pair(bad)


def _runs(tmp_path, names):
    out = []
    for n in names:
        d = tmp_path / n
        d.mkdir()
        out.append(d)
    return out


def test_resolve_pair_prefers_a_shared_seed(tmp_path):
    runs = _runs(tmp_path, [
        "sr_r2a_new_gap_ce_anorm_recalpost_seed1",
        "sr_r2a_new_gap_ce_anorm_recalpost_seed42",
        "sr_r1a_new_gap_ce_anorm_recalpost_seed42",
        "sr_r1a_new_gap_ce_anorm_recalpost_seed666",
    ])
    t, b, matched = resolve_pair(runs, ("r2a", None), ("r1a", None))
    assert matched
    assert t.name.endswith("seed42") and b.name.endswith("seed42")


def test_resolve_pair_reports_when_no_seed_is_shared(tmp_path):
    """r2b/r1b on the real run set. The contrast then carries seed variation
    as well as the treatment, and the caller has to be able to say so."""
    runs = _runs(tmp_path, [
        "sr_r2b_new_nohc_gap_ce_anorm_recalpost_seed1",
        "sr_r1b_new_nohc_gap_ce_anorm_recalpost_seed666",
    ])
    t, b, matched = resolve_pair(runs, ("r2b", None), ("r1b", None))
    assert not matched
    assert t.name.endswith("seed1") and b.name.endswith("seed666")


def test_resolve_pair_honours_a_pinned_seed(tmp_path):
    runs = _runs(tmp_path, [
        "sr_r2a_new_gap_ce_anorm_recalpost_seed1",
        "sr_r2a_new_gap_ce_anorm_recalpost_seed42",
        "sr_r1a_new_gap_ce_anorm_recalpost_seed42",
    ])
    t, _, _ = resolve_pair(runs, ("r2a", 1), ("r1a", 42))
    assert t.name.endswith("seed1")


def test_resolve_side_refuses_an_absent_arm_or_seed(tmp_path):
    runs = _runs(tmp_path, ["sr_r2a_new_gap_ce_anorm_recalpost_seed1"])
    with pytest.raises(SystemExit):
        resolve_side(runs, "r9z", None)
    with pytest.raises(SystemExit):
        resolve_side(runs, "r2a", 777)


# ---------------------------------------------------------------- normalisation
def test_peak_of_ignores_the_negative_half():
    a = np.linspace(-10, 1, 1000).reshape(10, 100)
    assert peak_of(a) >= 0


def test_normalise_none_is_the_identity():
    a, b = np.random.default_rng(0).normal(size=(2, 16, 16))
    na, nb, pa, pb = normalise_pair(a, b, "none")
    assert np.allclose(na, a) and np.allclose(nb, b)
    assert (pa, pb) == (1.0, 1.0)


def test_normalise_peak_puts_each_map_on_its_own_scale():
    """A model that is uniformly louder differences to ~zero under `peak` and
    not under `none` — the separation the flag exists to make."""
    rng = np.random.default_rng(1)
    a = np.abs(rng.normal(size=(32, 32)))
    b = 3.0 * a                                  # same structure, 3x louder
    d_raw = np.subtract(*normalise_pair(a, b, "none")[:2])
    d_peak = np.subtract(*normalise_pair(a, b, "peak")[:2])
    assert np.abs(d_peak).mean() < 1e-9 < np.abs(d_raw).mean()


def test_identical_maps_difference_to_zero_under_both():
    a = np.abs(np.random.default_rng(2).normal(size=(16, 16)))
    for how in ("none", "peak"):
        na, nb, _, _ = normalise_pair(a, a.copy(), how)
        assert np.allclose(na - nb, 0.0)


# ------------------------------------------------------------------- cache names
def test_map_stem_carries_the_geometry():
    """A patch-16 map must not be able to land in a patch-32 figure through
    --reuse-maps."""
    a = map_stem(7, "margin", "all", 16, 8, "r2a", 42)
    b = map_stem(7, "margin", "all", 32, 16, "r2a", 42)
    assert a != b
    assert "p16s8" in a and "p32s16" in b
    assert map_stem(7, "margin", "all", 16, 8, "r1a", 42) != a
