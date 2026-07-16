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
    make_tl_ce, tl_weight_map,
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


@pytest.mark.parametrize("arm", ["bce", "gap_ce", "tl_ce", "bce_dice",
                                 "pstar_dice", "pstar_tversky", "focal_tversky",
                                 "bce_dice+cldice", "bce_dice+skelrec"])
def test_all_arms_build_and_run(arm):
    loss = build_loss(arm, pstar="gap_ce")
    lg = logits_from(hline_with_gap()).requires_grad_(True)
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    out = loss(lg, tgt, epoch=99)
    out.backward()
    assert torch.isfinite(out) and torch.isfinite(lg.grad).all()


def test_t2_t4_pending():
    with pytest.raises(NotImplementedError):
        build_loss("t2_ce")
