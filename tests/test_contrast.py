"""Paired contrast maps — the arithmetic under the picture.

`contrast.py` is three forwards and a subtraction, so what can go wrong is the
subtraction: which band got flattened, which direction the sign runs, and
whether the region summary describes the map it claims to.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sr.probes.contrast import arm_pair, contrasts, region_stats
from sr.probes.extract import BAND_NAMES


class StubModel:
    def __init__(self, c=4):
        self.band_mean = torch.zeros(1, c, 1, 1)


class StubDecoder:
    """logits = per-band weighted sum at each location; no spatial mixing."""

    def __init__(self, weights=(1.0, 2.0, 3.0, 4.0)):
        self.m = StubModel(len(weights))
        self.w = torch.tensor(weights).view(1, -1, 1, 1)

    def logits(self, y):
        return (y * self.w).sum(dim=1)


def test_band_contrast_isolates_the_named_band():
    """Δ for band b must equal w_b * y_b — everything else cancels."""
    y = torch.rand(4, 8, 8, generator=torch.Generator().manual_seed(0)) + 1.0
    dec = StubDecoder()
    maps, intact = contrasts(dec, y, None, list(BAND_NAMES), torch)
    for b, name in enumerate(BAND_NAMES):
        expect = (dec.w[0, b, 0, 0] * y[b]).numpy()
        assert np.allclose(maps[f"band_{name}"], expect, atol=1e-5), name


def test_sign_is_positive_when_the_band_supports_the_logit():
    """Positive means removing it COSTS evidence — the paper's convention and
    the suite's. A sign flip here would invert every map's reading."""
    y = torch.ones(4, 4, 4)
    maps, _ = contrasts(StubDecoder((1.0, 0.0, 0.0, 0.0)), y, None, ["R"], torch)
    assert (maps["band_R"] > 0).all()
    maps, _ = contrasts(StubDecoder((-1.0, 0.0, 0.0, 0.0)), y, None, ["R"], torch)
    assert (maps["band_R"] < 0).all()


def test_a_band_the_model_ignores_yields_a_flat_zero_map():
    y = torch.rand(4, 6, 6, generator=torch.Generator().manual_seed(1))
    maps, _ = contrasts(StubDecoder((1.0, 1.0, 1.0, 0.0)), y, None, ["NIR"], torch)
    assert np.abs(maps["band_NIR"]).max() < 1e-9


def test_model_contrast_is_the_input_swap_on_one_decoder():
    """Both inputs go through the SAME checkpoint's z-score and U-Net: the
    treatment is the image, not the reader."""
    g = torch.Generator().manual_seed(2)
    y, y_ref = torch.rand(4, 6, 6, generator=g), torch.rand(4, 6, 6, generator=g)
    dec = StubDecoder()
    maps, _ = contrasts(dec, y, y_ref, [], torch, ref_name="r0")
    expect = (dec.logits(y[None]) - dec.logits(y_ref[None]))[0].numpy()
    assert np.allclose(maps["model_sr_vs_r0"], expect, atol=1e-6)


def test_model_contrast_is_named_after_its_baseline():
    """r0 makes it 'what the SR added over bicubic'; r1a makes it 'what joint
    training added over the frozen generator'. Same arithmetic, different
    question, so the key and the filename have to distinguish them."""
    g = torch.Generator().manual_seed(3)
    y, y_ref = torch.rand(4, 4, 4, generator=g), torch.rand(4, 4, 4, generator=g)
    maps, _ = contrasts(StubDecoder(), y, y_ref, [], torch, ref_name="r1a")
    assert list(maps) == ["model_sr_vs_r1a"]


def test_model_contrast_is_skipped_without_a_reference():
    y = torch.rand(4, 4, 4)
    maps, _ = contrasts(StubDecoder(), y, None, ["R"], torch)
    assert "model_sr_vs_bicubic" not in maps


def test_region_stats_describe_the_map_they_summarise():
    d = np.zeros((8, 8))
    d[0] = 2.0                                   # one row strongly positive
    d[1] = -1.0
    masks = {"road": np.zeros((8, 8), bool), "veg": np.zeros((8, 8), bool),
             "other": np.zeros((8, 8), bool)}
    masks["road"][0] = True
    masks["veg"][1] = True
    masks["other"][2:] = True
    rows = {r["region"]: r for r in region_stats(d, masks, chip=3)}
    assert rows["road"]["mean"] == pytest.approx(2.0)
    assert rows["road"]["frac_positive"] == 1.0
    assert rows["veg"]["mean"] == pytest.approx(-1.0)
    assert rows["veg"]["frac_positive"] == 0.0
    assert rows["other"]["mean"] == 0.0
    assert all(r["chip"] == 3 for r in rows.values())


def test_region_stats_skip_an_empty_region():
    d = np.ones((4, 4))
    masks = {"road": np.ones((4, 4), bool), "veg": np.zeros((4, 4), bool),
             "other": np.zeros((4, 4), bool)}
    assert [r["region"] for r in region_stats(d, masks)] == ["road"]


def test_frac_positive_survives_a_zero_mean():
    """A band can be neutral on average while mattering in both directions —
    the number a mean alone would hide."""
    d = np.array([[3.0, -3.0], [3.0, -3.0]])
    masks = {"road": np.ones((2, 2), bool), "veg": np.zeros((2, 2), bool),
             "other": np.zeros((2, 2), bool)}
    r = region_stats(d, masks)[0]
    assert r["mean"] == pytest.approx(0.0)
    assert r["mean_abs"] == pytest.approx(3.0)
    assert r["frac_positive"] == 0.5


def test_arm_pair_names_treatment_then_baseline():
    """Every output for a run lands in one directory, so the pair has to be in
    the name or two baselines overwrite each other."""
    assert arm_pair("r2a", "r1a") == "r2a_r1a"
    assert arm_pair("r2a", "r0") == "r2a_r0"
    assert arm_pair("r2a", None) == "r2a"
