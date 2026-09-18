"""Occlusion suite v3 — the acceptance list of docs/occlusion_suite_plan.md §6.

CPU-only and checkpoint-free. What needs a GPU is the model wiring; what is
worth testing here is the bookkeeping that would silently produce a wrong
figure: batching, the window partition, the affine match, the chip set, and
Eq. 1 of O'Sullivan & Dev.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sr.probes import occl_context
from sr.probes.attr_extract import region_masks
from sr.probes.occl_context_extract import (_batched_margins, affine_match,
                                            margin_of, paste, patch_slices,
                                            propose_exemplars,
                                            region_conditions,
                                            sample_road_pixels, sliding,
                                            window_coords)


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(20260903)


class FakeDecoder:
    """A deterministic stand-in for `z-score -> U-Net -> logits`.

    Nonlinear and spatially mixing, so a batching bug cannot hide behind
    linearity, but with no parameters to load.
    """

    def __init__(self):
        self.w = torch.linspace(0.3, 1.7, 4).view(1, 4, 1, 1)

    def logits(self, y):
        z = (y * self.w).sum(dim=1, keepdim=True)
        z = torch.nn.functional.avg_pool2d(z, 3, stride=1, padding=1)
        return torch.tanh(z)[:, 0]


# ---------------------------------------------------------------- §6.1 batching
def test_batched_equals_sequential(rng):
    dec = FakeDecoder()
    road = torch.zeros(16, 16, dtype=torch.bool)
    road[3:6, :] = True
    variants = [torch.rand(4, 16, 16, generator=rng) for _ in range(7)]
    ref = np.array([float(margin_of(dec.logits(v[None]), road)[0]) for v in variants])
    for batch in (1, 2, 4, 16):
        got, _ = _batched_margins(dec, list(variants), road, None, batch, torch)
        assert np.allclose(got, ref, atol=1e-6), batch


def test_batched_pixel_logits_match_sequential(rng):
    dec = FakeDecoder()
    road = torch.zeros(16, 16, dtype=torch.bool)
    road[0, :] = True
    pix = torch.tensor([5, 40, 200])
    variants = [torch.rand(4, 16, 16, generator=rng) for _ in range(5)]
    ref = np.stack([dec.logits(v[None]).reshape(1, -1)[:, pix].numpy()[0]
                    for v in variants])
    _, got = _batched_margins(dec, list(variants), road, pix, 2, torch)
    assert np.allclose(got, ref, atol=1e-6)


# ----------------------------------------------------------------- §6.2 regions
def test_regions_partition_and_road_is_disjoint():
    x = np.zeros((4, 3, 3), dtype="float32")
    x[0], x[3] = 0.1, 0.1
    x[3, 0, 0] = 0.9
    road = np.zeros((12, 12), dtype=bool)
    road[0, :] = True
    m = region_masks(x, road, 4)
    stack = np.stack([m[k] for k in ("road", "veg", "other")])
    assert (stack.sum(0) == 1).all()
    assert not (m["road"] & m["veg"]).any()


# ------------------------------------------------------------ §6.3 affine match
def test_affine_match_is_idempotent_when_stats_agree(rng):
    y = torch.rand(4, 8, 8, generator=rng)
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[2:6, 2:6] = True
    assert torch.allclose(affine_match(y, y, m), y, atol=1e-5)


def test_affine_match_transfers_radiometry_not_structure(rng):
    src = torch.rand(4, 8, 8, generator=rng)
    dst = src * 3.0 + 5.0            # same structure, different radiometry
    m = torch.ones(8, 8, dtype=torch.bool)
    got = affine_match(src, dst, m)
    assert torch.allclose(got, dst, atol=1e-4)


def test_affine_match_is_a_near_noop_when_both_stats_agree(rng):
    """When donor and destination share BOTH moments the match has nothing to
    do. Note what this does NOT license: on a real HC-on arm the shared low
    band fixes the global MEANS (measured 2e-5 relative on r2a/seed42) while
    local std ratios reach 10-22x on a road, because that is exactly what the
    SR's high band adds. The extractor asserts the mean null, not this one."""
    lo = torch.rand(4, 1, 1, generator=rng).expand(4, 16, 16)
    hi_a = 0.01 * torch.randn(4, 16, 16, generator=rng)
    hi_b = 0.01 * torch.randn(4, 16, 16, generator=rng)
    src, dst = lo + hi_a, lo + hi_b
    m = torch.ones(16, 16, dtype=torch.bool)
    shift = float((affine_match(src, dst, m) - src).pow(2).mean().sqrt()
                  / src.pow(2).mean().sqrt())
    assert shift < 0.05, shift


def test_paste_only_touches_the_mask(rng):
    dst = torch.rand(4, 8, 8, generator=rng)
    src = torch.zeros(4, 8, 8)
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[1:3, 1:3] = True
    out = paste(dst, src, m)
    assert (out[:, 1:3, 1:3] == 0).all()
    assert torch.equal(out[:, 5:, 5:], dst[:, 5:, 5:])


# ------------------------------------------------------- §6.5 sliding bookkeeping
def test_windows_partition_the_grid():
    for size, patch in ((512, 32), (512, 64), (512, 16)):
        coords = window_coords(size, patch, patch)
        cover = np.zeros((size, size), dtype=int)
        for r, c in coords:
            cover[r:r + patch, c:c + patch] += 1
        assert (cover == 1).all(), (size, patch)
        assert len(coords) == (size // patch) ** 2


def test_patch_slices_still_cover_a_non_dividing_grid():
    """A patch size that does not divide the grid must still cover it, or the
    robustness re-runs would silently measure a different area than the
    headline."""
    xs = patch_slices(100, 32, 32)
    cover = np.zeros(100, dtype=int)
    for r in xs:
        cover[r:r + 32] += 1
    assert (cover >= 1).all()


def test_sliding_variants_differ_only_inside_their_window(rng):
    y = torch.rand(4, 16, 16, generator=rng)
    fill = torch.zeros(4, 1, 1)
    for (r, c), v in zip(window_coords(16, 8, 8), sliding(y, fill, 8, 8, torch)):
        d = (v != y)
        assert d[:, r:r + 8, c:c + 8].all()
        assert int(d.sum()) == 4 * 8 * 8


def test_broadcast_and_donor_fills_both_cover_the_window(rng):
    """A (C,1,1) constant must broadcast into every window; slicing it — the
    bug this test exists for — leaves all but the first window untouched."""
    y = torch.rand(4, 16, 16, generator=rng)
    for fill in (torch.zeros(4, 1, 1), torch.zeros(4, 16, 16)):
        vs = list(sliding(y, fill, 8, 8, torch))
        assert all(int((v != y).sum()) == 4 * 64 for v in vs), fill.shape


# ----------------------------------------------------------------- conditions
def test_region_conditions_cover_the_planned_grid(rng):
    y = torch.rand(4, 12, 12, generator=rng)
    band_mean = torch.zeros(4)
    masks = {"road": torch.zeros(12, 12, dtype=torch.bool),
             "veg": torch.zeros(12, 12, dtype=torch.bool),
             "other": torch.zeros(12, 12, dtype=torch.bool)}
    masks["road"][0] = True
    masks["veg"][1] = True
    masks["other"][2:] = True
    src = {"bic": torch.rand(4, 12, 12, generator=rng)}
    conds = region_conditions(y, band_mean, masks, src, torch)
    names = [c[0] for c in conds]
    assert all(len(c) == 4 for c in conds)      # name, tensor, shift, std_ratio
    # 4 bands x (3 regions + all) + allbands x 4 + one counterfactual per region
    assert len([n for n in names if n.startswith("band_")]) == 16
    assert len([n for n in names if n.startswith("allbands@")]) == 4
    assert len([n for n in names if n.startswith("cf_bic@")]) == 3
    assert "band_NIR@veg" in names and "band_R@all" in names


def test_counterfactual_records_the_contrast_ratio_it_applied(rng):
    """The std ratio is a REPORTED diagnostic, not an assertion: a donor that is
    flat where the destination has structure yields a large ratio, and that is
    the SR working rather than a wiring fault."""
    y = torch.rand(4, 12, 12, generator=rng) * 2.0
    flat = torch.full((4, 12, 12), 0.5) + 0.001 * torch.rand(4, 12, 12, generator=rng)
    m = {"road": torch.ones(12, 12, dtype=torch.bool)}
    conds = {c[0]: c for c in region_conditions(y, torch.zeros(4), m, {"bic": flat},
                                                torch)}
    assert conds["cf_bic@road"][3] > 5.0


def test_band_condition_touches_one_band_inside_one_region(rng):
    y = torch.rand(4, 12, 12, generator=rng)
    masks = {"veg": torch.zeros(12, 12, dtype=torch.bool)}
    masks["veg"][3:5] = True
    conds = dict((c[0], c[1]) for c in region_conditions(
        y, torch.zeros(4), masks, {}, torch))
    v = conds["band_NIR@veg"]
    assert torch.equal(v[:3], y[:3])                 # other bands untouched
    assert torch.equal(v[3, :3], y[3, :3])           # outside the region
    assert (v[3, 3:5] == 0).all()                    # flattened to band_mean


# ------------------------------------------------------------- §6.4 chip set
def test_sampled_road_pixels_are_deterministic_and_on_road():
    mask = np.zeros((32, 32), dtype=bool)
    mask[4, :] = True
    a = sample_road_pixels(mask, 8, "deadbeef", 3)
    b = sample_road_pixels(mask, 8, "deadbeef", 3)
    c = sample_road_pixels(mask, 8, "deadbeef", 4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.size == 8 and mask.reshape(-1)[a].all()


def test_sampling_returns_every_pixel_of_a_thin_chip():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, :3] = True
    assert np.array_equal(sample_road_pixels(mask, 32, "abc12345", 0),
                          np.array([0, 1, 2]))


def test_propose_exemplars_skips_road_free_chips():
    """A road-free chip produces no margin, hence no map — registering one as an
    exemplar yields a silently empty panel, which is how the first smoke run
    ended up with an empty maps/ directory."""
    chips = [{"stratum": "Urban", "road_px": 0 if i % 2 else 40} for i in range(20)]
    got = propose_exemplars(chips)
    assert chips[got["Urban"]]["road_px"] > 0


def test_propose_exemplars_gives_one_per_stratum():
    chips = [{"stratum": ["Urban", "Rural", "PeriUrban"][i % 3], "road_px": 10}
             for i in range(60)]
    got = propose_exemplars(chips)
    assert set(got) == {"Urban", "Rural", "PeriUrban"}
    assert got == propose_exemplars(chips)
    for s, i in got.items():
        assert chips[i]["stratum"] == s


def test_propose_exemplars_stays_inside_the_subsample():
    """The proposal must come from the chips the pass will actually run, or the
    extractor's own exemplar guard rejects it — which is how this was caught."""
    chips = [{"stratum": ["Urban", "Rural", "PeriUrban"][i % 3], "road_px": 10}
             for i in range(60)]
    subset = list(range(0, 60, 7))
    got = propose_exemplars(chips, subset)
    assert set(got.values()) <= set(subset)


# ------------------------------------------------- weighted context (Eq. 1)
def _ctx_frame(dlogits, size=16, patch=4, pix=0, run="sr_r0_x_seed0"):
    import pandas as pd

    rows = []
    for (r, c), d in zip(window_coords(size, patch, patch), dlogits):
        rows.append(dict(run=run, arm="r0", seed=0, chip=0, pix=pix, pix_k=0,
                         patch=patch, row=r, col=c, dlogit=float(d)))
    return pd.DataFrame(rows)


def test_weighted_context_matches_equation_1_by_hand():
    """All the saliency mass in one window: W is that window's share of the
    total distance from p, which is Eq. 1 evaluated by hand."""
    size, patch = 16, 4
    n = size // patch
    d = np.zeros(n * n)
    d[-1] = 1.0                                  # the far corner window
    w = occl_context.weighted_context(_ctx_frame(d, size, patch), size)
    dist = np.asarray(occl_context.patch_distance_sums(0, 0, size, patch))
    assert float(w["W"].iloc[0]) == pytest.approx(dist[-1] / dist.sum())


def test_weighted_context_grows_with_distance():
    """The metric's whole purpose: mass far from p must score higher than the
    same mass next to p."""
    size, patch = 16, 4
    n = size // patch
    near, far = np.zeros(n * n), np.zeros(n * n)
    near[0] = 1.0
    far[-1] = 1.0
    wn = occl_context.weighted_context(_ctx_frame(near, size, patch), size)
    wf = occl_context.weighted_context(_ctx_frame(far, size, patch), size)
    assert float(wf["W"].iloc[0]) > float(wn["W"].iloc[0])


def test_weighted_context_is_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    d = rng.normal(size=16)
    for norm in ("minmax", "relu"):
        w = occl_context.weighted_context(_ctx_frame(d, 16, 4), 16, norm)
        assert 0.0 <= float(w["W"].iloc[0]) <= 1.0


def test_weighted_context_separates_patch_sizes(tmp_path):
    """The cache holds the §4 sweep alongside the headline, so a pixel appears
    once per patch size. Grouping without `patch` would fold them into one map
    and silently mis-index it."""
    import pandas as pd

    a = _ctx_frame(np.eye(4).ravel(), 16, 4)
    b = _ctx_frame(np.ones(64), 16, 2)
    w = occl_context.weighted_context(pd.concat([a, b], ignore_index=True), 16)
    assert sorted(w["patch"]) == [2, 4]
    assert w["W"].notna().all()


def test_weighted_context_refuses_an_incomplete_map():
    f = _ctx_frame(np.ones(16), 16, 4).iloc[:-1]
    with pytest.raises(SystemExit, match="does not cover the grid"):
        occl_context.weighted_context(f, 16)


# --------------------------------------------------- analysis, end to end
def _write_cache(root, run, dmargins, size=16, patch=4, fixture="fx1", tau=0.3,
                 seed=0, chips=(0, 1)):
    """A synthetic occl2 cache with planted band x region deltas."""
    import pandas as pd

    d = root / run / "occl2"
    d.mkdir(parents=True, exist_ok=True)
    cond, patch_rows, ctx = [], [], []
    for c in chips:
        cond.append(dict(chip=c, condition="intact", region=None, band=None,
                         fill=None, margin=1.0, dmargin=0.0, affine_shift=0.0,
                         road_px=10, stratum="Rural"))
        for band, per_region in dmargins.items():
            for region, v in per_region.items():
                cond.append(dict(chip=c, condition=f"band_{band}@{region}",
                                 region=region, band=band, fill="mean",
                                 margin=1.0 + v, dmargin=v, affine_shift=0.0,
                                 road_px=10, stratum="Rural"))
        cond.append(dict(chip=c, condition="allbands@all", region="all",
                         band=None, fill="mean", margin=0.2, dmargin=-0.8,
                         affine_shift=0.0, road_px=10, stratum="Rural"))
        cond.append(dict(chip=c, condition="cf_bic@road", region="road",
                         band=None, fill="bic", margin=0.7, dmargin=-0.3,
                         affine_shift=0.001, road_px=10, stratum="Rural"))
        for (r, cc) in window_coords(size, patch, patch):
            patch_rows.append(dict(chip=c, patch=patch, fill="mean", row=r,
                                   col=cc, margin=0.95, dmargin=-0.05))
            ctx.append(dict(chip=c, patch=patch, pix=0, pix_k=0, row=r, col=cc,
                            dlogit=0.1 if (r or cc) else 1.0))
    pd.DataFrame(cond).to_parquet(d / "conditions.parquet", index=False)
    pd.DataFrame(patch_rows).to_parquet(d / "patches.parquet", index=False)
    pd.DataFrame(ctx).to_parquet(d / "context.parquet", index=False)
    (d / "meta.json").write_text(json.dumps({
        "run": run, "arm": "?", "seed": seed, "fixture_hash": fixture,
        "fixture_meta": {"crop": size // 4, "upscale": 4}, "ndvi_tau": tau,
        "patch": patch, "k_pixels": 1, "chips": list(chips),
        "chips_evaluated": list(chips), "target": "margin"}))
    return d


@pytest.fixture
def cache(tmp_path):
    _write_cache(tmp_path, "sr_r0_x_seed0",
                 {b: {"road": -v, "veg": -v / 2, "other": -v / 4, "all": -v}
                  for b, v in (("R", 0.1), ("G", 0.3), ("B", 0.4), ("NIR", 0.2))})
    return tmp_path


def test_load_metas_reads_and_orders(cache):
    metas = occl_context.load_metas(cache)
    assert len(metas) == 1 and metas[0]["arm"] == "r0"


def test_load_metas_rejects_a_tau_disagreement(tmp_path):
    _write_cache(tmp_path, "sr_r0_x_seed0", {"R": {"all": -0.1}}, tau=0.3)
    _write_cache(tmp_path, "sr_r2a_x_seed1", {"R": {"all": -0.1}}, tau=0.5)
    with pytest.raises(SystemExit, match="NDVI"):
        occl_context.load_metas(tmp_path)


def test_load_metas_rejects_a_fixture_disagreement(tmp_path):
    _write_cache(tmp_path, "sr_r0_x_seed0", {"R": {"all": -0.1}}, fixture="a")
    _write_cache(tmp_path, "sr_r2a_x_seed1", {"R": {"all": -0.1}}, fixture="b")
    with pytest.raises(SystemExit, match="not on one axis"):
        occl_context.load_metas(tmp_path)


def test_band_region_recovers_the_planted_ordering(cache):
    metas = occl_context.load_metas(cache)
    br = occl_context.band_region(occl_context._read(metas, "conditions.parquet"))
    road = br[br["region"] == "road"].set_index("band")["dmargin"]
    assert road["B"] < road["G"] < road["NIR"] < road["R"] < 0
    veg = br[br["region"] == "veg"].set_index("band")["dmargin"]
    assert veg["NIR"] > road["NIR"]        # a milder region-conditioned effect


def test_main_writes_csvs_and_figures(cache, tmp_path):
    out = tmp_path / "figs"
    assert occl_context.main(["--cache-dir", str(cache), "--out-dir", str(out)]) == 0
    for f in ("occl2_band_region.csv", "occl2_context.csv",
              "occl2_robustness.csv", "occl2_patch_sanity.csv",
              "F6_occl_band_region.pdf", "F7_weighted_context.png",
              "A6_occl_appendix.pdf"):
        assert (out / f).exists(), f
