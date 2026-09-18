"""Unit tests for the probe fixtures and extraction pass.

The probes (docs/lda_cka_band_probes_plan.md) put several arms on ONE axis, and
every claim they support rests on invariants that are cheap to check here and
expensive to notice later:

  * §2  the fixture is a frozen, reproducible sample — same seed, same chips,
        same pixels, and background/road labels that actually match the masks;
  * §4  CKA examples are PAIRED across arms: the spatial positions a stage is
        sampled at must depend on (stage, chip) and on nothing about the arm,
        or a between-arm CKA compares different pixels;
  * §5  occluding band b makes that channel exactly zero AFTER the arm's own
        z-score — the "this band carries no signal" semantics the readout is
        interpreted with — and the condition set is complete and non-redundant.

The models here are throwaway: what is under test is the wiring, not SEN2SR.
Anything needing the real checkpoints (that r0-against-r0 makes `cka_own` and
`cka_common` bit-identical, that θ* matches the run's sweep) is a smoke run of
`sr.probes.extract`, not a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sr.probes.extract import OCCLUSION, Extractor
from sr.probes.make_fixtures import allocate, sample_pixels


# --------------------------------------------------------------- §2 fixtures
def test_allocate_sums_to_total_and_tracks_proportions():
    counts = {"Urban": 59, "Rural": 59, "PeriUrban": 63}
    got = allocate(counts, 400)
    assert sum(got.values()) == 400
    # Largest-remainder rounding never moves a stratum by more than one chip
    # from its exact share, so the sample cannot silently over-weight a class.
    for k, v in counts.items():
        assert abs(got[k] - 400 * v / sum(counts.values())) < 1.0


def test_allocate_handles_a_total_smaller_than_the_stratum_count():
    got = allocate({"a": 10, "b": 10, "c": 10}, 2)
    assert sum(got.values()) == 2


def _toy_masks(rng, n=6, side=16):
    """`n` masks with wildly different road areas, incl. one that is empty."""
    masks = []
    for i in range(n):
        m = np.zeros((side, side), dtype=bool)
        if i:                                     # chip 0 is road-free on purpose
            m.ravel()[rng.choice(side * side, size=8 * i, replace=False)] = True
        masks.append(m)
    return masks


def test_pixel_sample_is_labelled_by_the_mask_it_came_from():
    rng = np.random.default_rng(0)
    masks = _toy_masks(rng)
    road_px = [int(m.sum()) for m in masks]
    ci, pi, is_road = sample_pixels(masks, np.asarray(road_px), 200, 4.0, seed=7)
    for c, p, r in zip(ci, pi, is_road):
        assert bool(masks[c].ravel()[p]) is bool(r)


def test_pixel_sample_respects_the_bg_ratio_and_draws_without_replacement():
    masks = _toy_masks(np.random.default_rng(1))
    road_px = np.asarray([int(m.sum()) for m in masks])
    ci, pi, is_road = sample_pixels(masks, road_px, 200, 4.0, seed=7)
    assert is_road.sum() == 40 and (~is_road).sum() == 160
    # No pixel is drawn twice within a chip: a duplicate would enter the LDA
    # cloud with double weight and be invisible in the figure.
    pairs = list(zip(ci.tolist(), pi.tolist()))
    assert len(set(pairs)) == len(pairs)
    # Sorted by (chip, pixel) so extraction visits each chip once, in order.
    assert np.array_equal(np.lexsort((pi, ci)), np.arange(ci.size))


def test_pixel_sample_is_deterministic_in_its_seed():
    masks = _toy_masks(np.random.default_rng(2))
    road_px = np.asarray([int(m.sum()) for m in masks])
    a = sample_pixels(masks, road_px, 200, 4.0, seed=11)
    b = sample_pixels(masks, road_px, 200, 4.0, seed=11)
    c = sample_pixels(masks, road_px, 200, 4.0, seed=12)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])


def test_a_road_free_chip_still_contributes_background():
    masks = _toy_masks(np.random.default_rng(3))
    road_px = np.asarray([int(m.sum()) for m in masks])
    ci, _, is_road = sample_pixels(masks, road_px, 400, 4.0, seed=5)
    assert (ci == 0).sum() > 0                    # chip 0 has no road at all
    assert not is_road[ci == 0].any()


# ------------------------------------------------------------------- §4 CKA
class _Toy(torch.nn.Module):
    """Enough of a JointSRUNetLightning to exercise the extraction wiring."""

    def __init__(self, c=4, scale=1.0, mean=0.3, std=0.1):
        super().__init__()
        self.model = torch.nn.Conv2d(c, 1, 1)
        self.register_buffer("band_mean", torch.full((1, c, 1, 1), mean))
        self.register_buffer("band_std", torch.full((1, c, 1, 1), std))
        self.hparams = type("H", (), {"reflectance_scale": scale})()


def _ex(**kw):
    return Extractor(_Toy(**kw), "cpu", positions=8, cka_seed=3, stages=[])


def test_stage_positions_depend_on_stage_and_chip_but_not_on_the_arm():
    a, b = _ex(mean=0.3), _ex(mean=0.9)           # two "arms", same fixture
    act = torch.arange(2 * 4 * 6 * 6, dtype=torch.float32).reshape(2, 4, 6, 6)
    a._grab, b._grab = {"s": act}, {"s": act}
    same = (a.sample_stage("s", 0, chip_i=5, stage_i=1),
            b.sample_stage("s", 0, chip_i=5, stage_i=1))
    assert np.array_equal(*same)
    # ... and genuinely move with the chip and the stage, so the sample sweeps
    # the field instead of re-reading one fixed lattice.
    assert not np.array_equal(same[0], a.sample_stage("s", 0, 6, 1))
    assert not np.array_equal(same[0], a.sample_stage("s", 0, 5, 2))


def test_stage_sample_is_positions_by_channels():
    e = _ex()
    e._grab = {"s": torch.zeros(1, 7, 4, 4)}
    assert e.sample_stage("s", 0, 0, 0).shape == (8, 7)   # P=8 positions, C=7


def test_stage_sample_caps_at_the_available_positions():
    e = _ex()                                      # positions=8, map has only 4
    e._grab = {"s": torch.zeros(1, 3, 2, 2)}
    assert e.sample_stage("s", 0, 0, 0).shape == (4, 3)


def test_sampled_sr_pixels_are_reflectance_not_raw_units():
    e = _ex(scale=10000.0)
    y = torch.full((4, 3, 3), 2500.0)              # raw units
    got = e.sample_pixels(y, np.array([0, 4], dtype="int32"))
    assert got.shape == (2, 4)
    assert np.allclose(got, 0.25)


# ------------------------------------------------------------- §5 occlusion
def test_occluding_a_band_zeroes_it_exactly_after_the_z_score():
    e = _ex(mean=0.3, std=0.1)
    y = torch.rand(1, 4, 5, 5) + 0.2
    for name, idxs in OCCLUSION.items():
        yo = y.clone()
        for b in idxs:
            yo[:, b] = e.model.band_mean.reshape(-1)[b]
        x_seg = (yo - e.model.band_mean) / e.model.band_std
        for b in range(4):
            if b in idxs:
                assert torch.equal(x_seg[:, b], torch.zeros_like(x_seg[:, b])), name
            else:
                assert torch.equal(x_seg[:, b], (y[:, b] - 0.3) / 0.1), name


def test_the_condition_set_is_complete_and_non_redundant():
    # Four singles plus RGB. The plan lists a NIR *group* too; for a 4-band
    # model that is the NIR band, so emitting it twice would double-count one
    # condition in the figure.
    assert set(OCCLUSION) == {"R", "G", "B", "NIR", "RGB"}
    assert len({tuple(sorted(v)) for v in OCCLUSION.values()}) == len(OCCLUSION)
    assert sorted(OCCLUSION["RGB"] + OCCLUSION["NIR"]) == [0, 1, 2, 3]


# --------------------------------------------------------------- extraction
def test_hooks_stay_silent_unless_the_forward_asked_to_capture():
    """The five occlusion forwards must not materialise every stage."""
    m = _Toy()
    e = Extractor(m, "cpu", positions=4, cka_seed=1, stages=[("s", m.model)])
    e.attach()
    e.unet(torch.rand(1, 4, 3, 3), capture=False)
    assert e._grab == {}
    e.unet(torch.rand(1, 4, 3, 3), capture=True)
    assert set(e._grab) == {"s"}
    e.detach()


def test_a_hook_point_fired_twice_is_an_error_not_a_silent_average():
    class Twice(_Toy):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.ReLU()
            self.model = torch.nn.Sequential(
                self.shared, torch.nn.Conv2d(4, 1, 1), self.shared)

    m = Twice()
    e = Extractor(m, "cpu", positions=4, cka_seed=1, stages=[("s", m.shared)])
    e.attach()
    with pytest.raises(RuntimeError, match="fired 2 times"):
        e.unet(torch.rand(1, 4, 3, 3), capture=True)
    e.detach()


# ---------------------------------------------------------------- §3 the LDA
def _two_clouds(n=4000, sep=(1.0, 0.0, 0.0, 0.0), seed=0):
    """Road and background Gaussians separated along band 0 only."""
    rng = np.random.default_rng(seed)
    bg = rng.normal(0, 1, size=(4 * n, 4))
    road = rng.normal(0, 1, size=(n, 4)) + np.asarray(sep)
    x = np.concatenate([road, bg]).astype("float32")
    y = np.concatenate([np.ones(n, bool), np.zeros(4 * n, bool)])
    return x, y


def test_lda_axis_finds_the_separating_band():
    from sr.probes.lda import lda_axis

    w = lda_axis(*_two_clouds())
    assert abs(w[0]) > 0.9 and np.allclose(np.linalg.norm(w), 1.0)


def test_lda_axis_beats_every_single_band_it_is_built_from():
    """The whole claim of a *shared linear frame* is that the axis is a better
    projection than any raw band; if it were not, the instrument would be
    reporting band 0 with extra steps."""
    from sr.probes.lda import fisher, lda_axis

    x, y = _two_clouds(sep=(0.8, -0.6, 0.0, 0.3))
    w = lda_axis(x, y)
    best_band = max(fisher(x, y, np.eye(4)[b]) for b in range(4))
    assert fisher(x, y, w) > best_band


def test_fisher_is_invariant_to_axis_norm_and_to_a_global_rescaling():
    """Both invariances are load-bearing: the first lets arms be compared on one
    axis, the second stops a purely global brightness change reading as a gain.
    A PER-BAND affine change must still move it — that is the r2b hypothesis."""
    from sr.probes.lda import fisher, lda_axis

    x, y = _two_clouds(sep=(1.0, 0.4, 0.0, 0.0))
    w = lda_axis(x, y)
    f = fisher(x, y, w)
    assert np.isclose(fisher(x, y, 7.3 * w), f)
    assert np.isclose(fisher(3.5 * x, y, w), f)
    assert not np.isclose(fisher(x * np.array([1.0, 5.0, 1.0, 1.0]), y, w), f)


def test_own_frame_fisher_never_falls_below_the_shared_frame():
    """The own axis maximises exactly this ratio, so only the GAP is readable.
    A cache where own < shared would mean the refit is broken."""
    from sr.probes.lda import fisher, lda_axis

    x0, y = _two_clouds(sep=(1.0, 0.0, 0.0, 0.0), seed=1)
    shared = lda_axis(x0, y)
    x1, y1 = _two_clouds(sep=(0.2, 1.2, 0.0, 0.0), seed=2)   # a different direction
    assert fisher(x1, y1, lda_axis(x1, y1)) >= fisher(x1, y1, shared) - 1e-9


def test_second_plot_axis_is_orthogonal_to_the_discriminant():
    from sr.probes.lda import lda_axis, orthogonal_pc

    x, y = _two_clouds(sep=(1.0, 0.5, 0.0, 0.0))
    w = lda_axis(x, y)
    v = orthogonal_pc(x, y, w)
    assert abs(v @ w) < 1e-6 and np.isclose(np.linalg.norm(v), 1.0)


# --------------------------------------------------------------- §9 the style
def test_every_arm_is_distinguishable_on_lines_and_on_markers_alike():
    """Three orthogonal channels only work if no two arms collide in all of
    them — and the check has to hold for BOTH panel kinds. F2 is a dot plot with
    no lines, so `linestyle` cannot separate r2a from r2b there; marker shape
    stands in for it, and fill is unavailable because the replication cue owns
    it."""
    from sr.probes import style

    on_lines = {(style.color(a), style.linestyle(a)) for a in style.ARMS}
    on_markers = {(style.color(a), style.marker(a)) for a in style.ARMS}
    assert len(on_lines) == len(style.ARMS)
    assert len(on_markers) == len(style.ARMS)


def test_marker_shape_carries_the_same_meaning_as_linestyle():
    from sr.probes import style

    for arm in ("r1a", "r2a", "r4a", "r5a"):
        assert style.marker(arm) == "o" and style.linestyle(arm) == "-"
    for arm in ("r1b", "r2b", "r4b", "r5b"):
        assert style.marker(arm) == "s" and style.linestyle(arm) == "--"
    assert style.marker("r0") == "D"      # the anchor, neither on nor off


def test_hard_constraint_state_is_the_linestyle_and_r0_is_not_dashed():
    from sr.probes import style

    assert style.linestyle("r2a") == "-" and style.linestyle("r2b") == "--"
    assert style.linestyle("r4a") == "-" and style.linestyle("r4b") == "--"
    # r0 has no generator to constrain; dashing it would invite a comparison
    # that does not exist.
    assert style.linestyle("r0") == "-"


def test_adaptation_state_is_the_lightness_within_a_generator_row():
    from sr.probes import style

    assert style.color("r1a") != style.color("r2a")     # frozen vs joint SEN2SR
    assert style.color("r5a") != style.color("r4a")     # frozen vs joint SR4RS
    assert style.color("r1a") == style.color("r1b")     # HC is not the hue
    assert style.color("r4a") == style.color("r4b")


def test_arm_is_parsed_out_of_a_full_run_directory_name():
    from sr.probes import style

    assert style.arm_of("sr_r2b_new_nohc_gap_ce_anorm_recalpost_seed66") == "r2b"
    assert style.arm_of("sr_r0_new_gap_ce_anorm_recalpost_seed66") == "r0"
    assert style.arm_of("r4a") == "r4a"


def test_single_seed_arms_get_a_hollow_marker():
    from sr.probes import style

    assert style.marker_kwargs(1)["markerfacecolor"] == "none"
    assert "markerfacecolor" not in style.marker_kwargs(3)


# --------------------------------------------------------------- §5 readouts
def _occ_frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def test_per_chip_iou_is_nan_on_a_correctly_empty_chip():
    """An empty union means no road and none predicted — a correct outcome with
    no IoU. Scoring it 1.0 would reward an arm for the fixture's 29% empty
    chips; scoring it 0.0 would punish it for the same."""
    from sr.probes.occlusion import _iou_per_chip

    got = _iou_per_chip(_occ_frame([
        {"tp": 0, "fp": 0, "fn": 0},        # correctly empty
        {"tp": 4, "fp": 4, "fn": 2},
    ]))
    assert np.isnan(got.iloc[0]) and np.isclose(got.iloc[1], 0.4)


def _meta(n_chips=1):
    return {"dir": None, "run": "x", "arm": "r0", "seed": 0, "n_chips": n_chips,
            "theta": 0.5, "theta_provenance": "sweep"}


def test_delta_ap_is_paired_on_the_road_bearing_chips_only():
    """AP is NaN on a road-free chip, so both arms of the difference must be
    restricted to the same chips or the delta compares two different means."""
    from sr.probes.occlusion import CONDITIONS, deltas

    road = {0: 100, 1: 0, 2: 50}
    rows = [{"chip": c, "condition": cond,
             "ap": float("nan") if road[c] == 0 else (0.5 if cond == "none" else 0.2),
             "tp": road[c] // 2, "fp": 10, "fn": road[c] // 2, "tn": 1000,
             "road_px": road[c]}
            for c in road for cond in ["none"] + CONDITIONS]
    out = deltas(_meta(len(road)), _occ_frame(rows))
    assert (out["n_ap_chips"] == 2).all()          # the road-free chip is excluded
    assert np.allclose(out["dap"], 0.2 - 0.5)


def test_delta_ap_floor_is_the_chance_level():
    """AP's no-skill value is the chip prevalence, so a collapsed model lands on
    (chance - intact). Several conditions coinciding there is saturation, not a
    coincidence, and the figure draws the floor to say so."""
    from sr.probes.occlusion import CONDITIONS, deltas

    rows = [{"chip": 0, "condition": cond, "ap": 0.4 if cond == "none" else 0.02,
             "tp": 1, "fp": 1, "fn": 1, "tn": 97, "road_px": 2}
            for cond in ["none"] + CONDITIONS]
    out = deltas(_meta(), _occ_frame(rows))
    assert np.allclose(out["ap_chance"], 0.02)     # 2 road px of 100
    assert np.allclose(out["dap_floor"], 0.02 - 0.4)
    assert np.allclose(out["dap"], out["dap_floor"])   # fully collapsed


# ------------------------------------------------------------------- §4 CKA
class _Toy(torch.nn.Module):
    """Enough of a JointSRUNetLightning to exercise the extraction wiring."""

    def __init__(self, c=4, scale=1.0, mean=0.3, std=0.1):
        super().__init__()
        self.model = torch.nn.Conv2d(c, 1, 1)
        self.register_buffer("band_mean", torch.full((1, c, 1, 1), mean))
        self.register_buffer("band_std", torch.full((1, c, 1, 1), std))
        self.hparams = type("H", (), {"reflectance_scale": scale})()


def _ex(**kw):
    return Extractor(_Toy(**kw), "cpu", positions=8, cka_seed=3, stages=[])


def test_stage_positions_depend_on_stage_and_chip_but_not_on_the_arm():
    a, b = _ex(mean=0.3), _ex(mean=0.9)           # two "arms", same fixture
    act = torch.arange(2 * 4 * 6 * 6, dtype=torch.float32).reshape(2, 4, 6, 6)
    a._grab, b._grab = {"s": act}, {"s": act}
    same = (a.sample_stage("s", 0, chip_i=5, stage_i=1),
            b.sample_stage("s", 0, chip_i=5, stage_i=1))
    assert np.array_equal(*same)
    # ... and genuinely move with the chip and the stage, so the sample sweeps
    # the field instead of re-reading one fixed lattice.
    assert not np.array_equal(same[0], a.sample_stage("s", 0, 6, 1))
    assert not np.array_equal(same[0], a.sample_stage("s", 0, 5, 2))


def test_stage_sample_is_positions_by_channels():
    e = _ex()
    e._grab = {"s": torch.zeros(1, 7, 4, 4)}
    assert e.sample_stage("s", 0, 0, 0).shape == (8, 7)   # P=8 positions, C=7


def test_stage_sample_caps_at_the_available_positions():
    e = _ex()                                      # positions=8, map has only 4
    e._grab = {"s": torch.zeros(1, 3, 2, 2)}
    assert e.sample_stage("s", 0, 0, 0).shape == (4, 3)


def test_sampled_sr_pixels_are_reflectance_not_raw_units():
    e = _ex(scale=10000.0)
    y = torch.full((4, 3, 3), 2500.0)              # raw units
    got = e.sample_pixels(y, np.array([0, 4], dtype="int32"))
    assert got.shape == (2, 4)
    assert np.allclose(got, 0.25)


# ------------------------------------------------------------- §5 occlusion
def test_occluding_a_band_zeroes_it_exactly_after_the_z_score():
    e = _ex(mean=0.3, std=0.1)
    y = torch.rand(1, 4, 5, 5) + 0.2
    for name, idxs in OCCLUSION.items():
        yo = y.clone()
        for b in idxs:
            yo[:, b] = e.model.band_mean.reshape(-1)[b]
        x_seg = (yo - e.model.band_mean) / e.model.band_std
        for b in range(4):
            if b in idxs:
                assert torch.equal(x_seg[:, b], torch.zeros_like(x_seg[:, b])), name
            else:
                assert torch.equal(x_seg[:, b], (y[:, b] - 0.3) / 0.1), name


def test_the_condition_set_is_complete_and_non_redundant():
    # Four singles plus RGB. The plan lists a NIR *group* too; for a 4-band
    # model that is the NIR band, so emitting it twice would double-count one
    # condition in the figure.
    assert set(OCCLUSION) == {"R", "G", "B", "NIR", "RGB"}
    assert len({tuple(sorted(v)) for v in OCCLUSION.values()}) == len(OCCLUSION)
    assert sorted(OCCLUSION["RGB"] + OCCLUSION["NIR"]) == [0, 1, 2, 3]


# --------------------------------------------------------------- extraction
def test_hooks_stay_silent_unless_the_forward_asked_to_capture():
    """The five occlusion forwards must not materialise every stage."""
    m = _Toy()
    e = Extractor(m, "cpu", positions=4, cka_seed=1, stages=[("s", m.model)])
    e.attach()
    e.unet(torch.rand(1, 4, 3, 3), capture=False)
    assert e._grab == {}
    e.unet(torch.rand(1, 4, 3, 3), capture=True)
    assert set(e._grab) == {"s"}
    e.detach()


def test_a_hook_point_fired_twice_is_an_error_not_a_silent_average():
    class Twice(_Toy):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.ReLU()
            self.model = torch.nn.Sequential(
                self.shared, torch.nn.Conv2d(4, 1, 1), self.shared)

    m = Twice()
    e = Extractor(m, "cpu", positions=4, cka_seed=1, stages=[("s", m.shared)])
    e.attach()
    with pytest.raises(RuntimeError, match="fired 2 times"):
        e.unet(torch.rand(1, 4, 3, 3), capture=True)
    e.detach()


# ---------------------------------------------------------------- §3 the LDA
def _two_clouds(n=4000, sep=(1.0, 0.0, 0.0, 0.0), seed=0):
    """Road and background Gaussians separated along band 0 only."""
    rng = np.random.default_rng(seed)
    bg = rng.normal(0, 1, size=(4 * n, 4))
    road = rng.normal(0, 1, size=(n, 4)) + np.asarray(sep)
    x = np.concatenate([road, bg]).astype("float32")
    y = np.concatenate([np.ones(n, bool), np.zeros(4 * n, bool)])
    return x, y


def test_lda_axis_finds_the_separating_band():
    from sr.probes.lda import lda_axis

    w = lda_axis(*_two_clouds())
    assert abs(w[0]) > 0.9 and np.allclose(np.linalg.norm(w), 1.0)


def test_lda_axis_beats_every_single_band_it_is_built_from():
    """The whole claim of a *shared linear frame* is that the axis is a better
    projection than any raw band; if it were not, the instrument would be
    reporting band 0 with extra steps."""
    from sr.probes.lda import fisher, lda_axis

    x, y = _two_clouds(sep=(0.8, -0.6, 0.0, 0.3))
    w = lda_axis(x, y)
    best_band = max(fisher(x, y, np.eye(4)[b]) for b in range(4))
    assert fisher(x, y, w) > best_band


def test_fisher_is_invariant_to_axis_norm_and_to_a_global_rescaling():
    """Both invariances are load-bearing: the first lets arms be compared on one
    axis, the second stops a purely global brightness change reading as a gain.
    A PER-BAND affine change must still move it — that is the r2b hypothesis."""
    from sr.probes.lda import fisher, lda_axis

    x, y = _two_clouds(sep=(1.0, 0.4, 0.0, 0.0))
    w = lda_axis(x, y)
    f = fisher(x, y, w)
    assert np.isclose(fisher(x, y, 7.3 * w), f)
    assert np.isclose(fisher(3.5 * x, y, w), f)
    assert not np.isclose(fisher(x * np.array([1.0, 5.0, 1.0, 1.0]), y, w), f)


def test_own_frame_fisher_never_falls_below_the_shared_frame():
    """The own axis maximises exactly this ratio, so only the GAP is readable.
    A cache where own < shared would mean the refit is broken."""
    from sr.probes.lda import fisher, lda_axis

    x0, y = _two_clouds(sep=(1.0, 0.0, 0.0, 0.0), seed=1)
    shared = lda_axis(x0, y)
    x1, y1 = _two_clouds(sep=(0.2, 1.2, 0.0, 0.0), seed=2)   # a different direction
    assert fisher(x1, y1, lda_axis(x1, y1)) >= fisher(x1, y1, shared) - 1e-9


def test_second_plot_axis_is_orthogonal_to_the_discriminant():
    from sr.probes.lda import lda_axis, orthogonal_pc

    x, y = _two_clouds(sep=(1.0, 0.5, 0.0, 0.0))
    w = lda_axis(x, y)
    v = orthogonal_pc(x, y, w)
    assert abs(v @ w) < 1e-6 and np.isclose(np.linalg.norm(v), 1.0)


# --------------------------------------------------------------- §9 the style
def test_every_arm_is_distinguishable_on_lines_and_on_markers_alike():
    """Three orthogonal channels only work if no two arms collide in all of
    them — and the check has to hold for BOTH panel kinds. F2 is a dot plot with
    no lines, so `linestyle` cannot separate r2a from r2b there; marker shape
    stands in for it, and fill is unavailable because the replication cue owns
    it."""
    from sr.probes import style

    on_lines = {(style.color(a), style.linestyle(a)) for a in style.ARMS}
    on_markers = {(style.color(a), style.marker(a)) for a in style.ARMS}
    assert len(on_lines) == len(style.ARMS)
    assert len(on_markers) == len(style.ARMS)


def test_marker_shape_carries_the_same_meaning_as_linestyle():
    from sr.probes import style

    for arm in ("r1a", "r2a", "r4a", "r5a"):
        assert style.marker(arm) == "o" and style.linestyle(arm) == "-"
    for arm in ("r1b", "r2b", "r4b", "r5b"):
        assert style.marker(arm) == "s" and style.linestyle(arm) == "--"
    assert style.marker("r0") == "D"      # the anchor, neither on nor off


def test_hard_constraint_state_is_the_linestyle_and_r0_is_not_dashed():
    from sr.probes import style

    assert style.linestyle("r2a") == "-" and style.linestyle("r2b") == "--"
    assert style.linestyle("r4a") == "-" and style.linestyle("r4b") == "--"
    # r0 has no generator to constrain; dashing it would invite a comparison
    # that does not exist.
    assert style.linestyle("r0") == "-"


def test_adaptation_state_is_the_lightness_within_a_generator_row():
    from sr.probes import style

    assert style.color("r1a") != style.color("r2a")     # frozen vs joint SEN2SR
    assert style.color("r5a") != style.color("r4a")     # frozen vs joint SR4RS
    assert style.color("r1a") == style.color("r1b")     # HC is not the hue
    assert style.color("r4a") == style.color("r4b")


def test_arm_is_parsed_out_of_a_full_run_directory_name():
    from sr.probes import style

    assert style.arm_of("sr_r2b_new_nohc_gap_ce_anorm_recalpost_seed66") == "r2b"
    assert style.arm_of("sr_r0_new_gap_ce_anorm_recalpost_seed66") == "r0"
    assert style.arm_of("r4a") == "r4a"


def test_single_seed_arms_get_a_hollow_marker():
    from sr.probes import style

    assert style.marker_kwargs(1)["markerfacecolor"] == "none"
    assert "markerfacecolor" not in style.marker_kwargs(3)


# --------------------------------------------------------------- §5 readouts
def _occ_frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def test_per_chip_iou_is_nan_on_a_correctly_empty_chip():
    """An empty union means no road and none predicted — a correct outcome with
    no IoU. Scoring it 1.0 would reward an arm for the fixture's 29% empty
    chips; scoring it 0.0 would punish it for the same."""
    from sr.probes.occlusion import _iou_per_chip

    got = _iou_per_chip(_occ_frame([
        {"tp": 0, "fp": 0, "fn": 0},        # correctly empty
        {"tp": 4, "fp": 4, "fn": 2},
    ]))
    assert np.isnan(got.iloc[0]) and np.isclose(got.iloc[1], 0.4)


def _meta(n_chips=1):
    return {"dir": None, "run": "x", "arm": "r0", "seed": 0, "n_chips": n_chips,
            "theta": 0.5, "theta_provenance": "sweep"}


def test_delta_ap_is_paired_on_the_road_bearing_chips_only():
    """AP is NaN on a road-free chip, so both arms of the difference must be
    restricted to the same chips or the delta compares two different means."""
    from sr.probes.occlusion import CONDITIONS, deltas

    road = {0: 100, 1: 0, 2: 50}
    rows = [{"chip": c, "condition": cond,
             "ap": float("nan") if road[c] == 0 else (0.5 if cond == "none" else 0.2),
             "tp": road[c] // 2, "fp": 10, "fn": road[c] // 2, "tn": 1000,
             "road_px": road[c]}
            for c in road for cond in ["none"] + CONDITIONS]
    out = deltas(_meta(len(road)), _occ_frame(rows))
    assert (out["n_ap_chips"] == 2).all()          # the road-free chip is excluded
    assert np.allclose(out["dap"], 0.2 - 0.5)


def test_delta_ap_floor_is_the_chance_level():
    """AP's no-skill value is the chip prevalence, so a collapsed model lands on
    (chance - intact). Several conditions coinciding there is saturation, not a
    coincidence, and the figure draws the floor to say so."""
    import pandas as pd

    from sr.probes.occlusion import deltas

    rows = []
    for cond in ["none", "R", "G", "B", "NIR", "RGB"]:
        rows.append({"chip": 0, "condition": cond,
                     "ap": 0.4 if cond == "none" else 0.02,
                     "tp": 1, "fp": 1, "fn": 1, "tn": 97, "road_px": 2})
    orig = pd.read_parquet
    pd.read_parquet = lambda *_a, **_k: pd.DataFrame(rows)
    try:
        out = deltas({"dir": type("D", (), {"__truediv__": lambda s, o: s})(),
                      "run": "x", "arm": "r0", "seed": 0, "n_chips": 1,
                      "theta": 0.5, "theta_provenance": "sweep"})
    finally:
        pd.read_parquet = orig
    assert np.allclose(out["ap_chance"], 0.02)     # 2 road px of 100
    assert np.allclose(out["dap_floor"], 0.02 - 0.4)
    assert np.allclose(out["dap"], out["dap_floor"])   # fully collapsed


# ------------------------------------------------------------------- §4 CKA
def test_linear_cka_is_one_for_a_rotation_and_scaling_of_the_same_features():
    """The invariances that make CKA the right tool: orthogonal transforms and
    isotropic scaling of a representation must not register as a change."""
    from sr.probes.cka import linear_cka

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert np.isclose(linear_cka(x, x), 1.0)
    assert np.isclose(linear_cka(x, x @ q), 1.0)
    assert np.isclose(linear_cka(x, 4.7 * x), 1.0)
    # ... and translation, which the column-centring removes.
    assert np.isclose(linear_cka(x, x + 3.0), 1.0)


def test_linear_cka_falls_for_unrelated_representations():
    from sr.probes.cka import linear_cka

    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2000, 6)), rng.normal(size=(2000, 6))
    assert linear_cka(a, b) < 0.1


def test_linear_cka_is_symmetric_and_bounded():
    from sr.probes.cka import linear_cka

    rng = np.random.default_rng(2)
    a = rng.normal(size=(400, 5))
    b = a @ rng.normal(size=(5, 7)) + 0.5 * rng.normal(size=(400, 7))
    v = linear_cka(a, b)
    assert np.isclose(v, linear_cka(b, a)) and 0.0 <= v <= 1.0


def test_contrast_sets_resolve_as_documented():
    from sr.probes.cka import S1_CONTRASTS, resolve_contrasts

    arms = {"r0", "r1a", "r2a", "r2b"}
    assert resolve_contrasts(["s1"], arms) is S1_CONTRASTS
    assert resolve_contrasts(["vs-r0"], arms) == [("r1a", "r0"), ("r2a", "r0"),
                                                  ("r2b", "r0")]
    assert resolve_contrasts(["r2a:r1a"], arms) == [("r2a", "r1a")]


def test_vs_r0_without_an_r0_cache_is_an_error_not_an_empty_figure():
    from sr.probes.cka import resolve_contrasts

    with pytest.raises(SystemExit, match="needs an r0 cache"):
        resolve_contrasts(["vs-r0"], {"r2a", "r2b"})


def test_the_noise_floor_only_uses_within_arm_pairs():
    import pandas as pd

    from sr.probes.cka import noise_floor

    pairs = pd.DataFrame([
        {"convention": "own", "stage": "enc_layer1", "arm_a": "r2a",
         "arm_b": "r2a", "cka": 0.9},
        {"convention": "own", "stage": "enc_layer1", "arm_a": "r2a",
         "arm_b": "r2a", "cka": 0.8},
        {"convention": "own", "stage": "enc_layer1", "arm_a": "r2a",
         "arm_b": "r0", "cka": 0.2},     # a between-arm pair must not enter
    ])
    f = noise_floor(pairs)
    assert len(f) == 1
    assert (f["lo"].iloc[0], f["hi"].iloc[0], f["n"].iloc[0]) == (0.8, 0.9, 2)


def test_a_single_seed_cache_set_yields_no_floor_at_all():
    import pandas as pd

    from sr.probes.cka import noise_floor

    pairs = pd.DataFrame([{"convention": "own", "stage": "enc_layer1",
                           "arm_a": "r2a", "arm_b": "r0", "cka": 0.2}])
    assert noise_floor(pairs).empty


def test_own_frame_never_loses_under_class_imbalance():
    """Regression: the scatter matrix and the Fisher denominator have to be the
    SAME objective. With a class-size-weighted S_W and an unweighted (var+var)
    denominator they are not, and on the fixture's 4:1 sample the refitted axis
    scored BELOW the shared one — a negative gap, which §3 says cannot happen."""
    from sr.probes.lda import fisher, lda_axis

    rng = np.random.default_rng(4)
    road = rng.normal(0, 1, size=(2000, 4)) + np.array([0.9, -0.4, 0.2, 0.0])
    bg = rng.normal(0, 1.6, size=(8000, 4))            # 4:1, and heteroscedastic
    x = np.concatenate([road, bg])
    y = np.concatenate([np.ones(2000, bool), np.zeros(8000, bool)])
    own = lda_axis(x, y)
    for other in (np.eye(4)[0], np.eye(4)[1], lda_axis(x[::3], y[::3])):
        assert fisher(x, y, own) >= fisher(x, y, other) - 1e-9


def test_fisher_is_invariant_to_the_fixtures_bg_to_road_ratio():
    """The 4:1 ratio is a sampling-design choice, so no reported number may
    depend on it — otherwise two fixtures are not comparable.

    Tested structurally rather than by resampling: duplicating the background
    leaves its mean and covariance untouched and only its COUNT changes, so the
    unweighted criterion must be bit-stable. A class-size-weighted S_W fails
    this outright, and resampling instead would only measure sampling noise.
    """
    from sr.probes.lda import fisher, lda_axis

    rng = np.random.default_rng(5)
    road = rng.normal(0, 1, size=(1500, 4)) + np.array([1.0, 0.0, -0.5, 0.2])
    bg = rng.normal(0, 1.3, size=(3000, 4))

    def f(background):
        x = np.concatenate([road, background])
        y = np.concatenate([np.ones(len(road), bool),
                            np.zeros(len(background), bool)])
        return fisher(x, y, lda_axis(x, y))

    assert np.isclose(f(bg), f(np.concatenate([bg, bg])), rtol=1e-9)


def test_variance_concentration_flags_a_one_channel_stage():
    from sr.probes.cka import variance_concentration

    rng = np.random.default_rng(6)
    even = rng.normal(size=(500, 20))
    assert variance_concentration(even) > 0.7
    spiked = even.copy()
    spiked[:, 0] *= 300.0
    assert variance_concentration(spiked) < 0.1


def test_bars_carry_the_hard_constraint_state_too():
    """The third channel has to be re-expressed for every mark type: lines get
    linestyle, markers get shape, bars get hatching. Miss one and the two joint
    arms merge into a single bar in F1's loadings and F2's first-conv panel."""
    from sr.probes import style

    on_bars = {(style.color(a), style.hatch(a)) for a in style.ARMS}
    assert len(on_bars) == len(style.ARMS)
    assert style.hatch("r2a") is None and style.hatch("r2b") == "///"
    assert style.hatch("r0") is None
