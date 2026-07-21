"""Unit tests for the ablation loss module (protocol Stage 0).

Run first thing in the training env:  pytest tests/test_losses.py -v
These encode the protocol's fairness rules: §4.4 scale parity, detached
weight maps, §4.5 warmup, and paper-faithful weight-map geometry.
"""
import numpy as np
import pytest
import torch

from unet.losses import (
    ComposedLoss, DiceLoss, FocalTverskyLoss, SkeletonRecallLoss, SoftclDice,
    TverskyLoss, WeightedCE, build_loss, gap_weight_map, make_gap_ce,
    make_gap_tl_ce, make_tl_ce, t2_cells, t2_kernels, t4_cells, t4_kernels,
    tl_weight_map,
)

H = W = 96
HI, LO = 8.0, -8.0  # confident logits


def logits_from(mask: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return torch.where(t > 0.5, torch.tensor(HI), torch.tensor(LO))


def hline_with_gap(y=48, x0=8, x1=88, gap=(40, 56)) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    m[y, x0:x1] = 1
    m[y, gap[0]:gap[1]] = 0
    return m


def ring(cy=48, cx=48, rad=30) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (np.abs(d - rad) < 1.0).astype(np.float32)


def blob_target() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    t = (torch.rand(2, 1, H, W, generator=g) > 0.9).float()
    return torch.nn.functional.max_pool2d(t, 5, 1, 2)  # clumpy ~roads-ish


# ---------------------------------------------------------------- gap loss

def test_gap_no_endpoints_equals_bce():
    """Closed ring skeleton has no endpoints -> W==1 -> gap_ce == BCE exactly."""
    lg = logits_from(ring())
    tgt = torch.from_numpy(hline_with_gap())[None, None]
    Wmap = gap_weight_map(torch.sigmoid(lg))
    assert torch.all(Wmap == 1)
    gap = make_gap_ce()(lg, tgt)
    bce = WeightedCE(None)(lg, tgt)
    assert torch.allclose(gap, bce, atol=1e-6)


def test_gap_weights_concentrate_at_gap():
    mask = hline_with_gap(gap=(40, 56))
    Wmap = gap_weight_map(logits_from(mask).sigmoid(), r=4, K=60.0)[0, 0]
    assert Wmap[48, 40 - 2] > 1 and Wmap[48, 56 + 1] > 1   # around endpoints
    assert Wmap[48, 20] == 1                                # mid-line, far away
    assert Wmap[10, 10] == 1                                # background
    # K scaling: weights are K * endpoint-count
    W10 = gap_weight_map(logits_from(mask).sigmoid(), r=4, K=10.0)[0, 0]
    hot = Wmap > 1
    assert torch.allclose(Wmap[hot] / W10[hot], torch.tensor(6.0))


def test_gap_radius_widens_support():
    mask = hline_with_gap()
    p = logits_from(mask).sigmoid()
    small = (gap_weight_map(p, r=3) > 1).sum()
    large = (gap_weight_map(p, r=9) > 1).sum()
    assert large > small


# ------------------------------------------------------------ topological

def test_tl_weights_at_line_end():
    """Vertical line ending mid-image: high weight near the free end,
    base weight 1 on background far from line/borders (Giannini reset)."""
    m = np.zeros((H, W), np.float32)
    m[10:48, 48] = 1
    Wmap = tl_weight_map(logits_from(m).sigmoid(), ell=5)[0, 0]
    end_zone = Wmap[44:54, 46:51]
    assert end_zone.max() == 10
    assert Wmap[70:90, 5:20].max() == 1  # far background


def test_tl_base_reset_toggle():
    m = np.zeros((H, W), np.float32)
    m[10:48, 48] = 1
    p = logits_from(m).sigmoid()
    w_orig = tl_weight_map(p, ell=5, base_reset=False)[0, 0]
    assert w_orig[80, 10] == 4     # original paper: base weight 4
    w_reset = tl_weight_map(p, ell=5, base_reset=True)[0, 0]
    assert w_reset[80, 10] == 1


@pytest.mark.parametrize("ell", [3, 5, 7])
def test_tl_ell_grid_runs(ell):
    p = logits_from(hline_with_gap()).sigmoid()
    Wmap = tl_weight_map(p, ell=ell)
    assert Wmap.shape == p.shape and torch.isfinite(Wmap).all()


# ------------------------------------------------------- t2/t4 curvature

def test_t2_base_kernel_matches_eq4():
    """Eq. 4, n=5, (i0,j0)=(0,1), t=0..2(n-2): the quarter-circle staircase."""
    assert sorted(t2_cells(5)) == sorted(
        [(0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)])
    assert len(t2_cells(5)) == 2 * (5 - 2) + 1


def test_t4_base_kernel_matches_eq5():
    """Eq. 5, n=5: zig-zag down (Tb=3), middle run L=3, mirrored ascent."""
    assert sorted(t4_cells(5)) == sorted(
        [(0, 0), (1, 0), (1, 1),            # descent
         (2, 1), (2, 2), (2, 3),            # middle row m=2
         (1, 3), (1, 4), (0, 4)])           # ascent
    with pytest.raises(ValueError):
        t4_cells(4)  # n must be odd (protocol grid)


@pytest.mark.parametrize("maker", [t2_kernels, t4_kernels])
@pytest.mark.parametrize("n", [3, 5, 7])
def test_curvature_kernels_are_four_distinct_rotations(maker, n):
    ks = maker(n)
    assert len(ks) == 4 and all(k.shape == (n, n) for k in ks)
    keys = {k.numpy().tobytes() for k in ks}
    assert len(keys) == 4
    # closed under 180-deg rotation: cross-correlation with the SET equals
    # true convolution with the SET, so F.conv2d reproduces the paper's W
    assert all(torch.rot90(k, 2, dims=(0, 1)).numpy().tobytes() in keys for k in ks)


@pytest.mark.parametrize("maker", [t2_kernels, t4_kernels])
def test_curvature_weight_map_conv_xcorr_invariance(maker):
    """W must be identical whether the kernel set is fed as-is or 180-rotated
    (the permutation argument in `_rotations`)."""
    p = logits_from(hline_with_gap()).sigmoid()
    ks = maker(5)
    W1 = tl_weight_map(p, ell=5, extra_kernels=ks)
    W2 = tl_weight_map(p, ell=5, extra_kernels=[torch.rot90(k, 2, dims=(0, 1)) for k in ks])
    assert torch.equal(W1, W2)


@pytest.mark.parametrize("arm", ["t2_ce", "t4_ce"])
def test_t2_t4_background_base_reset(arm):
    """With 8 filters the non-endpoint floor is W==8 -> reset to 1."""
    m = np.zeros((H, W), np.float32)
    m[10:48, 48] = 1
    ks = t2_kernels(5) if arm == "t2_ce" else t4_kernels(5)
    Wmap = tl_weight_map(logits_from(m).sigmoid(), ell=5, extra_kernels=ks)[0, 0]
    assert Wmap[70:90, 5:20].max() == 1      # far background floor reset
    assert Wmap[44:54, 44:53].max() == 10    # free end still weighted up


# ------------------------------------------------- §4.4 scale normalization

@pytest.mark.parametrize("make", [make_gap_ce, make_tl_ce])
def test_scale_parity_with_bce(make):
    """Normalized weighted CE must sit at BCE's scale (weighted mean of the
    same ce map), so 'different loss' is not 'different effective LR'."""
    g = torch.Generator().manual_seed(1)
    lg = torch.randn(2, 1, H, W, generator=g) * 3
    tgt = blob_target()
    bce = WeightedCE(None)(lg, tgt)
    wce = make()(lg, tgt)
    assert 0.3 < (wce / bce).item() < 3.0


def test_gradients_flow_only_through_ce():
    lg = logits_from(hline_with_gap()).requires_grad_(True)
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    loss = make_gap_ce()(lg, tgt)
    loss.backward()
    assert lg.grad is not None and torch.isfinite(lg.grad).all()


# ----------------------------------------------------------------- region

def test_tversky_half_is_dice():
    """Exact identity at smooth=0 (with ε>0 the two smoothings differ by
    O(ε/denom²), so the unsmoothed forms are what's comparable)."""
    g = torch.Generator().manual_seed(2)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    assert torch.allclose(TverskyLoss(alpha=0.5, smooth=0.0)(lg, tgt),
                          DiceLoss(smooth=0.0)(lg, tgt), atol=1e-6)


def test_tversky_alpha_penalizes_fn():
    """alpha>0.5 must punish false negatives (missed road) harder."""
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    miss = logits_from(hline_with_gap(gap=(30, 60)))     # FN-heavy prediction
    lo, hi = TverskyLoss(alpha=0.3)(miss, tgt), TverskyLoss(alpha=0.7)(miss, tgt)
    assert hi > lo


def test_focal_tversky_exponent_one_is_tversky():
    g = torch.Generator().manual_seed(3)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    assert torch.allclose(FocalTverskyLoss(alpha=0.7, exponent=1.0)(lg, tgt),
                          TverskyLoss(alpha=0.7)(lg, tgt), atol=1e-6)


# --------------------------------------------------------------- skeleton

def _thick_hline(y0=47, y1=50, x0=8, x1=88, gap=(0, 0)) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = 1
    m[y0:y1, gap[0]:gap[1]] = 0
    return m


def test_cldice_prefers_connected():
    """clDice must prefer a thin-but-complete prediction over a broken
    full-width one (a gap loses GT-skeleton coverage; thinness doesn't),
    even though Dice prefers the broken one — the reason the skeleton slot
    exists. NB both predictions lie inside the target, so Tprec is ~1 and the
    comparison isolates the skeleton-coverage (Tsens) term."""
    tgt = torch.from_numpy(_thick_hline())[None, None]           # 3 px thick
    broken = logits_from(_thick_hline(gap=(40, 56)))             # thick, 16 px gap
    thin = logits_from(hline_with_gap(y=48, gap=(0, 0)))         # 1 px, connected
    cl = SoftclDice(skel_iters=5)
    assert cl(broken, tgt) > cl(thin, tgt)
    # Dice ranks them the other way round — pixel overlap can't see the break.
    assert DiceLoss()(broken, tgt) < DiceLoss()(thin, tgt)


def test_skelrec_bounds():
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    sr = SkeletonRecallLoss(tube_radius=1)
    full = sr(logits_from(np.ones((H, W), np.float32)), tgt)
    empty = sr(logits_from(np.zeros((H, W), np.float32)), tgt)
    assert full < 0.05 and empty > 0.9


# ------------------------------------------------------- composition/warmup

def test_warmup_schedule():
    loss = build_loss("bce_dice+cldice", warmup_start=30, warmup_ramp=10)
    wt = loss.skeleton_weight
    assert wt(0) == 0 and wt(29) == 0 and wt(30) == 0
    assert abs(wt(35) - 0.15) < 1e-9        # halfway to alpha=0.3
    assert abs(wt(40) - 0.3) < 1e-9 and abs(wt(99) - 0.3) < 1e-9


def test_cldice_convex_mix_weights():
    loss = build_loss("bce_dice+cldice")
    assert abs(loss.w_pix - 0.35) < 1e-9 and abs(loss.w_reg - 0.35) < 1e-9
    assert abs(loss.w_skel - 0.3) < 1e-9


def test_skelrec_additive_weights():
    loss = build_loss("bce_dice+skelrec")
    assert loss.w_pix == 0.5 and loss.w_reg == 0.5 and loss.w_skel == 1.0


def test_wbce_arm_and_bce_purity():
    """wbce consumes pos_weight; the bce arm must IGNORE it (anchor purity) —
    and wbce at w=1 must equal plain BCE exactly."""
    g = torch.Generator().manual_seed(4)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    plain = build_loss("bce")(lg, tgt)
    assert torch.allclose(build_loss("bce", pos_weight=7.0)(lg, tgt), plain)
    assert torch.allclose(build_loss("wbce", pos_weight=1.0)(lg, tgt), plain)
    assert not torch.allclose(build_loss("wbce", pos_weight=5.0)(lg, tgt), plain)
    # wbce+dice via the pstar mechanism (the "fair bce_dice" variant)
    loss = build_loss("pstar_dice", pstar="wbce", pos_weight=5.0)
    assert loss.pixel.pos_weight is not None and loss.w_pix == 0.5


def test_mix_w_reweights_pstar_compounds_only():
    """mix_w drives the pstar_* slot weights ((1-mw)·P* + mw·region); the
    bce_dice anchor must stay frozen at the literature's 0.5/0.5."""
    loss = build_loss("pstar_dice", mix_w=0.7)
    assert abs(loss.w_pix - 0.3) < 1e-9 and abs(loss.w_reg - 0.7) < 1e-9
    anchor = build_loss("bce_dice", mix_w=0.7)   # anchor ignores mix_w
    assert anchor.w_pix == 0.5 and anchor.w_reg == 0.5
    # mix_w=0.5 reproduces the pre-amendment behaviour exactly
    g = torch.Generator().manual_seed(5)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    assert torch.allclose(build_loss("pstar_dice", mix_w=0.5)(lg, tgt),
                          build_loss("bce_dice")(lg, tgt))


def test_gap_tl_blend_is_exact_average():
    """make_gap_tl_ce must equal 0.5·gap_ce + 0.5·tl_ce exactly: mean-1 maps
    summed inside one normalized weighted CE = the average of the two
    normalized losses (the algebra that keeps it a single pixel-slot arm)."""
    lg = logits_from(hline_with_gap()).float()
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    combined = make_gap_tl_ce(r=4, K=60.0, ell=5, theta=0.375)(lg, tgt)
    g = make_gap_ce(r=4, K=60.0)(lg, tgt)
    t = make_tl_ce(ell=5, theta=0.375)(lg, tgt)
    assert torch.allclose(combined, 0.5 * (g + t), atol=1e-5)


@pytest.mark.parametrize("arm", ["bce", "wbce", "gap_ce", "tl_ce", "gap_tl_ce",
                                 "t2_ce", "t4_ce",
                                 "bce_dice", "pstar_dice", "pstar_tversky",
                                 "focal_tversky", "bce_dice+cldice",
                                 "bce_dice+skelrec"])
def test_all_arms_build_and_run(arm):
    loss = build_loss(arm, pstar="gap_ce")
    lg = logits_from(hline_with_gap()).requires_grad_(True)
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    out = loss(lg, tgt, epoch=99)
    out.backward()
    assert torch.isfinite(out) and torch.isfinite(lg.grad).all()


def test_tl_theta_is_plumbed():
    """θ=0.375 must reach the weight map (borderline pixels binarize
    differently). Verified via a probability sitting between the two θs."""
    m = np.zeros((H, W), np.float32)
    m[10:60, 48] = 1
    lg = torch.where(torch.from_numpy(m)[None, None] > 0.5,
                     torch.tensor(-0.3), torch.tensor(LO))  # sigmoid(-0.3)=0.43
    tgt = torch.from_numpy(m)[None, None]
    lo = build_loss("tl_ce", tl_theta=0.375)(lg, tgt)   # 0.43 > 0.375: skeleton
    hi = build_loss("tl_ce", tl_theta=0.5)(lg, tgt)     # 0.43 < 0.5: empty skel
    assert not torch.allclose(lo, hi)
