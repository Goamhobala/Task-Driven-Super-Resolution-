"""Instrument E — the axioms the attribution pass rests on.

Everything here is CPU-only and checkpoint-free: the parts worth testing are
the quadrature, the completeness bookkeeping, the region partition and the
chip subsample, none of which need a trained arm. What needs a GPU is the
wiring, and the completeness guard is what tests that at runtime.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sr.probes import cache
from sr.probes.attr_extract import (choose_chips, completeness,
                                    integrated_gradients, make_target,
                                    mass_rows, region_masks, regions_to_lr)


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(20260902)


# ----------------------------------------------------------------------- IG
def test_ig_is_exact_on_a_linear_function(rng):
    """For f = <w, p>, IG is w * (x - baseline) at any number of steps."""
    w = torch.randn(3, 4, 4, generator=rng)
    x = torch.randn(3, 4, 4, generator=rng)
    b = torch.randn(3, 4, 4, generator=rng)
    f = lambda p: (p * w).flatten(1).sum(1)
    for steps in (1, 3, 32):
        attr = integrated_gradients(f, x, b, steps=steps, chunk=2)
        assert torch.allclose(attr, w * (x - b), atol=1e-6), steps


def test_midpoint_rule_is_exact_on_a_quadratic(rng):
    """The gradient along the path is linear in alpha, and midpoint integrates
    a linear integrand exactly — which is the whole reason plan v2 dropped
    `linspace(0, 1, steps)`. Completeness holds here at ONE step."""
    x = torch.randn(2, 5, 5, generator=rng)
    b = torch.zeros(2, 5, 5)
    f = lambda p: (p ** 2).flatten(1).sum(1)
    for steps in (1, 2, 8):
        attr = integrated_gradients(f, x, b, steps=steps, chunk=3)
        df, resid, rel = completeness(attr, f(x[None])[0], f(b[None])[0])
        assert rel < 1e-6, (steps, rel)


def test_ig_is_chunk_invariant(rng):
    """Chunking is a memory device, not a modelling choice."""
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                              torch.nn.ReLU(),
                              torch.nn.Conv2d(4, 1, 3, padding=1)).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = torch.rand(3, 8, 8, generator=rng)
    b = torch.zeros(3, 8, 8)
    f = lambda p: net(p).flatten(1).mean(1)
    a1 = integrated_gradients(f, x, b, steps=12, chunk=1)
    a5 = integrated_gradients(f, x, b, steps=12, chunk=5)
    a12 = integrated_gradients(f, x, b, steps=12, chunk=12)
    assert torch.allclose(a1, a5, atol=1e-6)
    assert torch.allclose(a1, a12, atol=1e-6)


def test_completeness_holds_through_a_relu_net(rng):
    """A saturating nonlinearity is where a Riemann sum can drift; 64 steps
    must still land inside the pass's own --tol."""
    net = torch.nn.Sequential(torch.nn.Conv2d(2, 6, 3, padding=1),
                              torch.nn.ReLU(),
                              torch.nn.Conv2d(6, 1, 3, padding=1)).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = torch.rand(2, 12, 12, generator=rng) * 3
    b = torch.full((2, 12, 12), 0.5)
    f = lambda p: net(p).flatten(1).mean(1)
    attr = integrated_gradients(f, x, b, steps=64, chunk=8)
    _df, _r, rel = completeness(attr, f(x[None])[0], f(b[None])[0])
    assert rel < 0.05, rel


def test_ig_traverses_the_fft_hard_constraint(rng):
    """The plan's load-bearing assumption: SEN2SR's constraint is differentiable.

    `HardConstraint` splices the LR image's low band into the SR output through
    torch.fft, and the LR path is where an E2 gradient has to survive. The DC
    bin comes entirely from the LR side, so a spatially FLAT target would probe
    only that path — the target here varies spatially, as the road margin does.
    """
    tricks = pytest.importorskip("sen2sr.models.tricks")
    # `ideal_filter`, not `gaussian_filter`: upstream's Gaussian calls
    # `torch.exp` on a python float and raises. The constraint's algebra is the
    # same either way, and the deployed sigma=35 mask is read from the shipped
    # file, never re-derived (instrument D's rule).
    mask = tricks.ideal_filter((16, 16), 4)
    hc = tricks.HardConstraint(low_pass_mask=mask, bands="all")
    gen = torch.nn.Conv2d(2, 2, 3, padding=1).eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    w = torch.randn(2, 16, 16, generator=rng)

    def f(lr):
        up = torch.nn.functional.interpolate(lr, size=(16, 16), mode="bicubic",
                                             antialias=True)
        return (hc(lr, gen(up)) * w).flatten(1).sum(1)

    x = torch.rand(2, 4, 4, generator=rng)
    b = x.mean(dim=(-2, -1), keepdim=True).expand_as(x).contiguous()
    attr = integrated_gradients(f, x, b, steps=16, chunk=4)
    assert float(attr.abs().sum()) > 0, "no gradient survived the FFT splice"
    _df, _r, rel = completeness(attr, f(x[None])[0], f(b[None])[0])
    assert rel < 0.05, rel


def test_a_frozen_generator_severs_the_graph():
    """`freeze_sr=True` wraps the generator in `torch.no_grad()`, so the path
    from the 10 m input to the logits is CUT — autograd raises rather than
    quietly returning zeros. This is why `Arm.__init__` clears the flag: the
    pass wants input gradients, and the flag only ever governed that wrap."""
    net = torch.nn.Conv2d(1, 1, 1).eval()

    def frozen(p):
        with torch.no_grad():
            h = net(p)
        return h.flatten(1).sum(1)

    with pytest.raises(RuntimeError):
        integrated_gradients(frozen, torch.rand(1, 4, 4), torch.zeros(1, 4, 4),
                             steps=2, chunk=2)


# ------------------------------------------------------------------ target
def test_margin_is_road_minus_background():
    road = torch.zeros(4, 4, dtype=torch.bool)
    road[0, :] = True
    logits = torch.zeros(1, 1, 4, 4)
    logits[0, 0, 0, :] = 2.0
    logits[0, 0, 1:, :] = -1.0
    assert float(make_target(road, "margin", torch)(logits)[0]) == pytest.approx(3.0)
    assert float(make_target(road, "road_sum", torch)(logits)[0]) == pytest.approx(8.0)


def test_margin_sees_background_suppression_and_road_sum_does_not():
    """The reason plan v2 changed the target. Two chips share their road logits
    and differ only in how hard the background is pushed down; occlusion's dAP
    would see that difference, `road_sum` cannot."""
    road = torch.zeros(4, 4, dtype=torch.bool)
    road[0, :] = True
    a = torch.zeros(1, 1, 4, 4)
    a[0, 0, 0, :] = 2.0
    b = a.clone()
    b[0, 0, 1:, :] = -3.0
    margin, rsum = make_target(road, "margin", torch), make_target(road, "road_sum", torch)
    assert float(margin(b)[0]) > float(margin(a)[0])
    assert float(rsum(b)[0]) == float(rsum(a)[0])


# ----------------------------------------------------------------- regions
def _chip(up=4):
    x = np.zeros((4, 3, 3), dtype="float32")
    x[0] = 0.1                       # R
    x[3] = 0.1                       # NIR -> NDVI 0 everywhere
    x[3, 0, 0] = 0.9                 # one strongly vegetated LR cell
    road = np.zeros((3 * up, 3 * up), dtype=bool)
    road[0, :] = True                # a road line crossing the veg cell
    return x, road, up


def test_regions_partition_and_road_wins():
    x, road, up = _chip()
    m = region_masks(x, road, up)
    stack = np.stack([m[k] for k in ("road", "veg", "other")])
    assert (stack.sum(0) == 1).all(), "regions must partition the chip"
    assert m["road"].sum() == road.sum()
    assert not (m["veg"] & m["road"]).any()
    assert m["veg"].sum() == up * up - up      # the veg cell minus its road row


def test_regions_do_not_depend_on_the_arm():
    """NDVI comes from the raw 10 m input, so nothing an SR generator does can
    move a boundary. v1 derived it from the SR output, which would have made
    the per-region masses incomparable across arms."""
    x, road, up = _chip()
    a = region_masks(x, road, up)
    b = region_masks(x * 1.0, road, up)        # same input, any arm downstream
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_regions_to_lr_is_a_majority_vote_with_fractions_kept():
    x, road, up = _chip()
    lr, frac = regions_to_lr(region_masks(x, road, up), up)
    stack = np.stack([lr[k] for k in ("road", "veg", "other")])
    assert (stack.sum(0) == 1).all()
    # The road line is 1 of 4 rows in its cells: a minority at 10 m, which is
    # exactly the mixing the fractions are kept to expose.
    assert frac["road"][0, 0] == pytest.approx(0.25)
    assert not lr["road"].any()
    assert lr["veg"][0, 0]


# ------------------------------------------------------------------- table
def test_mass_rows_decompose_the_attribution():
    x, road, up = _chip()
    masks = region_masks(x, road, up)
    attr = np.random.default_rng(0).normal(size=(4, 12, 12))
    rows = mass_rows(attr, masks, ["R", "G", "B", "NIR"], chip=0)
    assert len(rows) == 4 * 3
    assert sum(r["mass"] for r in rows) == pytest.approx(attr.sum())
    assert sum(r["absmass"] for r in rows) == pytest.approx(np.abs(attr).sum())
    assert sum(r["mass_frac"] for r in rows) == pytest.approx(1.0)
    assert sum(r["absmass_frac"] for r in rows) == pytest.approx(1.0)
    assert all(r["chip"] == 0 for r in rows)


# -------------------------------------------------------------- chip subset
def _chips(n=100):
    return [{"stratum": ["urban", "rural", "peri"][i % 3]} for i in range(n)]


def test_choose_chips_is_deterministic_and_exact():
    a = choose_chips(_chips(), 24, None)
    b = choose_chips(_chips(), 24, None)
    assert a == b and len(a) == 24 and a == sorted(set(a))


def test_choose_chips_keeps_the_strata_mix():
    chips = _chips(99)
    got = choose_chips(chips, 30, None)
    counts = {}
    for i in got:
        counts[chips[i]["stratum"]] = counts.get(chips[i]["stratum"], 0) + 1
    assert set(counts) == {"urban", "rural", "peri"}
    assert all(c == 10 for c in counts.values()), counts


def test_choose_chips_explicit_overrides():
    assert choose_chips(_chips(), 5, [9, 3, 3]) == [3, 9]
    assert choose_chips(_chips(10), 50, None) == list(range(10))


# ============================================================ the analysis
# `attribution.py` reads only caches, so it is testable end to end without a
# GPU: a synthetic cache with a KNOWN band ordering must come back out of the
# tables, and the two guards (one fixture across instruments, one chip set)
# must fire on the cases they exist for.
import pandas as pd

from sr.probes import attribution


def _write_attr_cache(root, run, shares, chips=(0, 1), fixture="abc123",
                      seed=None, arm_regions=None):
    """A cache whose per-band mass shares are exactly `shares`."""
    d = root / run
    (d).mkdir(parents=True, exist_ok=True)
    regions = arm_regions or {"road": 0.5, "veg": 0.3, "other": 0.2}
    rows = []
    for c in chips:
        for band, s in shares.items():
            for region, w in regions.items():
                rows.append(dict(chip=c, space="y", baseline="mean", band=band,
                                 region=region, n_px={"road": 100, "veg": 300,
                                                      "other": 600}[region],
                                 mass=s * w, absmass=s * w,
                                 mass_frac=s * w, absmass_frac=s * w))
    pd.DataFrame(rows).to_parquet(d / "attr_mass.parquet", index=False)
    (d / "meta.json").write_text(json.dumps({
        "run": run, "arm": "?", "seed": seed, "fixture_hash": fixture,
        "n_chips": len(chips), "chips": list(chips), "target": "margin",
        "theta": 0.5, "theta_provenance": "sweep", "resid_rel_max": 1e-4,
        "band_names": ["R", "G", "B", "NIR"]}))
    return d


def _write_occlusion_cache(root, run, dap, chips=(0, 1, 2), fixture="abc123"):
    """An occlusion cache where band b's dAP is `dap[b]` — on chips 0,1 ONLY.

    Chip 2 carries the REVERSED band ordering, weighted heavily enough to
    dominate a mean taken over all three, so a comparison that forgets to
    restrict itself to the attributed chips gets a visibly different answer.
    Reversing the ordering is the point: merely flipping the sign would leave
    a rank correlation untouched.
    """
    d = root / run
    d.mkdir(parents=True, exist_ok=True)
    names = list(dap)
    rev = {b: dap[names[len(names) - 1 - i]] * 4 for i, b in enumerate(names)}
    rows = []
    for c in chips:
        base = 0.9
        rows.append(dict(chip=c, condition="none", ap=base, tp=10, fp=1, fn=1,
                         tn=88, road_px=11))
        for b in names:
            v = dap[b] if c < 2 else rev[b]
            rows.append(dict(chip=c, condition=b, ap=base + v, tp=5, fp=1,
                             fn=6, tn=88, road_px=11))
    pd.DataFrame(rows).to_parquet(d / "occlusion.parquet", index=False)
    (d / "meta.json").write_text(json.dumps({
        "run": run, "fixture_hash": fixture, "n_chips": len(chips),
        "theta": 0.5, "theta_provenance": "sweep", "seed": None}))
    return d


@pytest.fixture
def caches(tmp_path):
    a, p = tmp_path / "attr", tmp_path / "probe"
    _write_attr_cache(a, "sr_r0_x_seed0", {"R": 0.1, "G": 0.3, "B": 0.4, "NIR": 0.2})
    _write_occlusion_cache(p, "sr_r0_x_seed0",
                           {"R": -0.05, "G": -0.15, "B": -0.20, "NIR": -0.10})
    return a, p


def test_band_shares_recover_the_planted_ordering(caches):
    a, _ = caches
    metas = cache.load_metas(a, require=("attr_mass.parquet",))
    sh = attribution.band_shares(attribution.load_attr(metas))
    got = sh.set_index("band")["share"]
    assert got["B"] > got["G"] > got["NIR"] > got["R"]
    assert got.sum() == pytest.approx(1.0)


def test_enrichment_divides_by_pixel_share(caches):
    """Road is 10% of the pixels and takes 50% of the mass -> enrichment 5."""
    a, _ = caches
    metas = cache.load_metas(a, require=("attr_mass.parquet",))
    reg = attribution.region_shares(attribution.load_attr(metas))
    road = reg[(reg["band"] == "B") & (reg["region"] == "road")].iloc[0]
    assert road["px_share"] == pytest.approx(0.1)
    assert road["enrichment"] == pytest.approx(5.0)


def test_consistency_uses_only_the_attributed_chips(caches):
    """The planted agreement is perfect on chips 0-1 and inverted on chip 2.

    Forgetting the restriction is not a subtle error here — it flips the sign
    of the correlation the main panel reports."""
    a, p = caches
    metas = cache.load_metas(a, require=("attr_mass.parquet",))
    occ = attribution.check_fixtures(metas, p)
    dap = attribution.occlusion_dap(metas, occ, {m["run"]: m["chips"] for m in metas})
    con = attribution.consistency(
        attribution.band_shares(attribution.load_attr(metas)), dap)
    assert attribution._spearman(con["share"], con["reliance"]) == pytest.approx(1.0)

    all_chips = {m["run"]: [0, 1, 2] for m in metas}
    dap_all = attribution.occlusion_dap(metas, occ, all_chips)
    con_all = attribution.consistency(
        attribution.band_shares(attribution.load_attr(metas)), dap_all)
    assert attribution._spearman(con_all["share"], con_all["reliance"]) < 1.0


def test_fixture_disagreement_across_caches_is_fatal(tmp_path):
    a, p = tmp_path / "attr", tmp_path / "probe"
    _write_attr_cache(a, "sr_r0_x_seed0", {"R": .25, "G": .25, "B": .25, "NIR": .25},
                      fixture="NEWHASH")
    _write_occlusion_cache(p, "sr_r0_x_seed0", {"R": -.1, "G": -.1, "B": -.1,
                                                "NIR": -.1}, fixture="OLDHASH")
    metas = cache.load_metas(a, require=("attr_mass.parquet",))
    with pytest.raises(SystemExit, match="not on one axis"):
        attribution.check_fixtures(metas, p)


def test_main_writes_csvs_and_figures(caches, tmp_path):
    a, p = caches
    out = tmp_path / "figs"
    assert attribution.main(["--attr-dir", str(a), "--cache-dir", str(p),
                             "--out-dir", str(out)]) == 0
    for f in ("attribution_band.csv", "attribution_mass.csv",
              "attribution_consistency.csv", "F9_attribution.pdf",
              "F9_attribution.png", "A9_attribution_regions.pdf"):
        assert (out / f).exists(), f
    con = pd.read_csv(out / "attribution_consistency.csv")
    assert set(con["band"]) == {"R", "G", "B", "NIR"}
    assert con["reliance"].notna().all()


def test_main_survives_a_missing_occlusion_cache(caches, tmp_path):
    """The attribution panel must still draw when only its own cache exists —
    the consistency panel is an addition, not a precondition."""
    a, _ = caches
    out = tmp_path / "figs2"
    assert attribution.main(["--attr-dir", str(a), "--cache-dir",
                             str(tmp_path / "nothing"), "--out-dir", str(out)]) == 0
    assert (out / "F9_attribution.png").exists()
