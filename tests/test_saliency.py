"""Per-pixel occlusion saliency — the bookkeeping a figure would hide.

A saliency map is looked at, not tabulated, so a wrong one is easy to accept.
These pin the parts that have a right answer: window coverage, the
accumulate-and-divide, which band a per-band fill actually touches, the pixel
draw, and Eq. 1 on a dense map.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sr.probes import saliency
from sr.probes.saliency import (normalise, saliency_maps, spread_pixels,
                                starts, weighted_context_dense)


class StubModel:
    def __init__(self, c=4):
        self.band_mean = torch.zeros(1, c, 1, 1)


class StubDecoder:
    """logits(y)[b, i, j] = weighted sum of the bands AT (i, j).

    Purely local, so the saliency map for a pixel p must be non-zero exactly
    where a window covering p sits — an analytic answer to check the
    accumulate-and-divide against.
    """

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        self.m = StubModel(len(weights))
        self.w = torch.tensor(weights).view(1, -1, 1, 1)

    def logits(self, y):
        return (y * self.w).sum(dim=1)


def _y(c=4, h=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(c, h, h, generator=g) + 1.0        # strictly positive


# ------------------------------------------------------------------ geometry
def test_starts_cover_the_grid_at_every_stride():
    for size, patch, stride in ((512, 16, 8), (512, 32, 32), (512, 16, 16),
                                (100, 32, 8)):
        cover = np.zeros(size, dtype=int)
        for r in starts(size, patch, stride):
            cover[r:r + patch] += 1
        assert (cover >= 1).all(), (size, patch, stride)


def test_overlapping_windows_actually_overlap():
    """The whole reason this module is not a flag on the suite's pass."""
    assert len(starts(512, 16, 8)) == 63
    assert len(starts(512, 32, 32)) == 16


# ------------------------------------------------------------ the dense pass
def test_map_is_local_for_a_local_model():
    """With a per-pixel model, occluding a window can only move the logit at p
    if the window covers p — so the map is exactly the union of those windows."""
    y = _y(h=32)
    dec = StubDecoder()
    p = 16 * 32 + 16                                  # pixel (16, 16)
    maps, l0 = saliency_maps(dec, y, [p], patch=8, stride=4, batch=4, torch=torch)
    m = maps[0]
    covered = np.zeros((32, 32), dtype=bool)
    for r in starts(32, 8, 4):
        for c in starts(32, 8, 4):
            if r <= 16 < r + 8 and c <= 16 < c + 8:
                covered[r:r + 8, c:c + 8] = True
    assert (np.abs(m[~covered]) < 1e-9).all()
    assert (m[covered] > 0).any()


def test_accumulation_divides_by_cover_count():
    """One window, so the mean over covering windows is the raw Δ: the model is
    the sum of bands at p, the fill is zero, hence Δ = the intact value."""
    y = _y(h=8)
    dec = StubDecoder()
    p = 4 * 8 + 4
    maps, _ = saliency_maps(dec, y, [p], patch=8, stride=8, batch=1, torch=torch)
    assert maps[0][4, 4] == pytest.approx(float(y[:, 4, 4].sum()), abs=1e-5)


def test_one_pass_serves_every_marked_pixel():
    y = _y(h=16)
    dec = StubDecoder()
    pix = [3 * 16 + 3, 12 * 16 + 12]
    both, _ = saliency_maps(dec, y, pix, 8, 4, 4, torch)
    for k, p in enumerate(pix):
        one, _ = saliency_maps(dec, y, [p], 8, 4, 4, torch)
        assert np.allclose(both[k], one[0], atol=1e-6)


def test_batching_does_not_change_the_map():
    y = _y(h=16)
    dec = StubDecoder()
    a, _ = saliency_maps(dec, y, [8 * 16 + 8], 8, 4, 1, torch)
    b, _ = saliency_maps(dec, y, [8 * 16 + 8], 8, 4, 7, torch)
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------- per band
def test_band_fill_touches_only_that_band():
    """A model that ignores band 3 must produce an all-zero NIR-only map, and a
    non-zero one for a band it does read. This is the per-band claim."""
    y = _y(h=16)
    dec = StubDecoder(weights=(1.0, 1.0, 1.0, 0.0))    # NIR ignored
    p = 8 * 16 + 8
    nir, _ = saliency_maps(dec, y, [p], 8, 8, 4, torch, band=3)
    red, _ = saliency_maps(dec, y, [p], 8, 8, 4, torch, band=0)
    assert np.abs(nir).max() < 1e-9
    assert red.max() > 0


def test_all_band_fill_is_the_sum_of_the_per_band_fills_for_a_linear_model():
    """Only true because the stub is linear — but it is the arithmetic the
    accumulator has to get right, and a real model's departure from it is the
    interaction the figure is actually about."""
    y = _y(h=16)
    dec = StubDecoder(weights=(0.5, 1.0, 1.5, 2.0))
    p = 8 * 16 + 8
    allb, _ = saliency_maps(dec, y, [p], 8, 8, 2, torch)
    per = sum(saliency_maps(dec, y, [p], 8, 8, 2, torch, band=b)[0]
              for b in range(4))
    assert np.allclose(allb[0], per, atol=1e-5)


# ------------------------------------------------------------ pixel choice
def test_spread_pixels_are_on_road_away_from_the_border_and_spread():
    mask = np.zeros((256, 256), dtype=bool)
    mask[100, :] = True
    mask[:, 100] = True
    got = spread_pixels(mask, 3, "deadbeef", 7, margin=64)
    ys, xs = got // 256, got % 256
    assert mask.reshape(-1)[got].all()
    assert ((ys >= 64) & (ys < 192) & (xs >= 64) & (xs < 192)).all()
    assert np.array_equal(got, spread_pixels(mask, 3, "deadbeef", 7, margin=64))
    d = np.hypot(ys[:, None] - ys[None, :], xs[:, None] - xs[None, :])
    assert d[np.triu_indices(len(got), 1)].min() > 10


def test_spread_pixels_falls_back_when_all_road_is_marginal():
    """A chip whose only road hugs the border must still yield pixels rather
    than an empty draw."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[2, :] = True
    got = spread_pixels(mask, 2, "abc12345", 0, margin=32)
    assert len(got) == 2 and mask.reshape(-1)[got].all()


# ------------------------------------------------------------------- Eq. 1
def test_dense_weighted_context_matches_the_definition():
    m = np.zeros((8, 8))
    m[7, 7] = 1.0
    g = np.arange(8, dtype="float64")
    d = np.sqrt((g[:, None] - 0) ** 2 + (g[None, :] - 0) ** 2)
    assert weighted_context_dense(m, 0, 0) == pytest.approx(d[7, 7] / d.sum())


def test_dense_weighted_context_grows_with_distance():
    near, far = np.zeros((16, 16)), np.zeros((16, 16))
    near[8, 9] = 1.0
    far[15, 15] = 1.0
    assert weighted_context_dense(far, 8, 8) > weighted_context_dense(near, 8, 8)


def test_normalise_both_readings():
    a = np.array([-2.0, 0.0, 1.0, 3.0])
    mm = normalise(a, "minmax")
    assert mm.min() == 0.0 and mm.max() == 1.0
    rl = normalise(a, "relu")
    assert rl[0] == 0.0 and rl.max() == 1.0
    assert np.array_equal(normalise(np.zeros(4), "minmax"), np.zeros(4))


def test_rgb_stretch_is_per_band():
    """A single stretch across all three keeps the scene's colour cast, which on
    a savanna tile hides the roads the figure is about."""
    y = np.stack([np.full((8, 8), 0.05), np.full((8, 8), 0.5),
                  np.full((8, 8), 0.9), np.zeros((8, 8))])
    y[0, 0, 0], y[1, 0, 0], y[2, 0, 0] = 0.4, 0.9, 1.2
    rgb = saliency.rgb_of(y)
    assert rgb.shape == (8, 8, 3)
    assert (rgb >= 0).all() and (rgb <= 1).all()
    # each band reaches the top of the range on its own maximum
    assert rgb[0, 0, :].min() > 0.9
