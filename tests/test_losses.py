"""Unit tests for the ablation loss module (protocol Stage 0).

Run first thing in the training env:  pytest tests/test_losses.py -v
These encode the protocol's fairness rules: §4.4 scale parity, detached
weight maps, §4.5 warmup, and paper-faithful weight-map geometry.
"""
import numpy as np
import pytest
import torch

from unet.losses import (
    BalancedCELoss, ComposedLoss, DiceLoss, FocalTverskyLoss, LogCoshDiceLoss,
    SkeletonRecallLoss, SoftclDice, SquaredDiceLoss, TverskyLoss, WeightedCE,
    build_loss, gap_weight_map, make_gap_ce, make_gap_tl_ce, make_tl_ce,
    t2_cells, t2_kernels, t4_cells, t4_kernels, tl_weight_map,
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


def test_skeletonize_batch_parallel_matches_serial():
    """The 2026-08-03 batch-parallel skeletonization must be bit-identical
    to the serial per-image loop (it is a performance change only)."""
    from unet.losses import _sk_skeletonize, _skeletonize_batch

    g = torch.Generator().manual_seed(13)
    b = (torch.rand(4, 1, H, W, generator=g) > 0.85).float()
    out = _skeletonize_batch(b)
    for i in range(4):
        ref = _sk_skeletonize(b[i, 0].numpy() > 0.5)
        assert np.array_equal(out[i, 0].numpy().astype(bool), ref.astype(bool))


def test_tl_floor_with_twelve_kernels():
    """gap_t2t4's TL side runs 4 line + 8 curvature kernels: the background
    floor (= n_kernels = 12) EXCEEDS the cap 10, so the base reset must
    happen before capping or the background saturates at max weight
    (the 2026-08-03 ordering fix). Endpoint weighting must survive."""
    m = np.zeros((H, W), np.float32)
    m[10:48, 48] = 1
    ks = t2_kernels(5) + t4_kernels(5)
    Wmap = tl_weight_map(logits_from(m).sigmoid(), ell=5, extra_kernels=ks)[0, 0]
    assert Wmap[70:90, 5:20].max() == 1     # far background floor reset
    assert Wmap[44:54, 44:53].max() == 10   # free end still weighted up


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


def test_wbce_normalized_scale_parity():
    """Amendment 2026-07-30: pos_weight goes through the §4.4 normalizer as
    W_pos = 1 + (λ−1)·y, so wbce sits at BCE's scale like every other
    weighted CE (previously the kwarg bypassed normalization and inflated
    the wbce arm's loss ~λ-fold on the positive share)."""
    g = torch.Generator().manual_seed(7)
    lg = torch.randn(2, 1, H, W, generator=g) * 3
    tgt = blob_target()
    bce = WeightedCE(None)(lg, tgt)
    wbce = WeightedCE(None, pos_weight=5.0)(lg, tgt)
    assert 0.3 < (wbce / bce).item() < 3.0
    # exact algebra: weighted mean of the plain-CE map under W_pos
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        lg, tgt, reduction="none")
    Wp = 1.0 + 4.0 * tgt
    assert torch.allclose(wbce, (Wp * ce).sum() / Wp.sum(), atol=1e-6)


def test_wbce_unnormalized_reproduces_kwarg_semantics():
    """normalize=False must equal F.bce_with_logits(pos_weight=λ).mean()
    exactly — the map form is the kwarg form, only the normalizer changed."""
    g = torch.Generator().manual_seed(8)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    old = torch.nn.functional.binary_cross_entropy_with_logits(
        lg, tgt, pos_weight=torch.tensor(5.0))
    new = WeightedCE(None, pos_weight=5.0, normalize=False)(lg, tgt)
    assert torch.allclose(new, old, atol=1e-6)


def test_pos_weight_composes_with_spatial_map():
    """λ is slot-orthogonal: with a weight_fn present, W = W_pos · W_spatial
    (one normalized weighted CE), enabling e.g. gap_ce at matched class
    balance in the 2×k pixel-slot design."""
    lg = logits_from(hline_with_gap()).float()
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    fn = lambda p: gap_weight_map(p, r=4, K=60.0)
    combined = WeightedCE(fn, pos_weight=5.0)(lg, tgt)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        lg, tgt, reduction="none")
    Wref = (1.0 + 4.0 * tgt) * gap_weight_map(torch.sigmoid(lg))
    assert torch.allclose(combined, (Wref * ce).sum() / Wref.sum(), atol=1e-6)
    # λ=1 leaves the spatial arm untouched
    assert torch.allclose(WeightedCE(fn, pos_weight=1.0)(lg, tgt),
                          WeightedCE(fn)(lg, tgt), atol=1e-6)


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


def test_sdice_equals_dice_on_binary_predictions():
    """p ∈ {0,1} ⇒ Σp² = Σp, so sDice == Dice exactly (smooth=0); on soft
    predictions the squared denominator makes them differ."""
    lg = logits_from(hline_with_gap())
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    assert torch.allclose(SquaredDiceLoss(smooth=0.0)(lg, tgt),
                          DiceLoss(smooth=0.0)(lg, tgt), atol=1e-6)
    g = torch.Generator().manual_seed(9)
    soft = torch.randn(2, 1, H, W, generator=g)
    assert not torch.allclose(SquaredDiceLoss(smooth=0.0)(soft, blob_target()),
                              DiceLoss(smooth=0.0)(soft, blob_target()))


def test_lcdice_is_logcosh_of_house_dice():
    """lcDice = log(cosh(per-sample house DiceLoss)) — the form the pilot's
    lcdice arm trained with, FROZEN for the study (2026-08-03). Documented
    deviation: Jadon's official pools the batch into one Dice first; the two
    coincide at B=1."""
    g = torch.Generator().manual_seed(10)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    d = DiceLoss()(lg, tgt)
    out = LogCoshDiceLoss()(lg, tgt)
    assert torch.allclose(out, torch.log(torch.cosh(d)), atol=1e-6)
    assert out <= d   # log-cosh ≈ x²/2 near 0: below the raw Dice loss


def test_balance_ce_adaptive_matches_manual():
    """Adaptive BalanCE: β = the batch's negative fraction; normalized
    weighted mean under W = β·y + (1−β)·(1−y)."""
    g = torch.Generator().manual_seed(11)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        lg, tgt, reduction="none")
    b = (1 - tgt).mean()
    Wb = b * tgt + (1 - b) * (1 - tgt)
    assert torch.allclose(BalancedCELoss()(lg, tgt),
                          (Wb * ce).sum() / Wb.sum(), atol=1e-6)
    # all-background batch: clamped β keeps it finite (≈ plain BCE on negs)
    assert torch.isfinite(BalancedCELoss()(lg[:1], torch.zeros(1, 1, H, W)))


def test_balance_ce_fixed_beta_is_wbce():
    """The redundancy identity that justifies the arm design: FIXED-β BalanCE
    ≡ WeightedCE(pos_weight=β/(1−β)) under §4.4 normalization — the (1−β)
    scale cancels. Only the adaptive form earns its own arm."""
    g = torch.Generator().manual_seed(12)
    lg = torch.randn(2, 1, H, W, generator=g)
    tgt = blob_target()
    for beta in (0.25, 0.8):   # 0.25 = Jadon's hard-coded value
        assert torch.allclose(
            BalancedCELoss(beta=beta)(lg, tgt),
            WeightedCE(None, pos_weight=beta / (1 - beta))(lg, tgt), atol=1e-5)


def test_build_loss_threads_pos_weight_into_gap():
    """Amendment 2026-07-30: build_loss('gap_ce', pos_weight=λ) must equal
    WeightedCE(gap map, pos_weight=λ) — matched-λ arms need no shell changes.
    Without the hp key the arm stays paper-faithful λ=1."""
    lg = logits_from(hline_with_gap()).float()
    tgt = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    via_arm = build_loss("gap_ce", pos_weight=5.0)(lg, tgt)
    direct = WeightedCE(lambda p: gap_weight_map(p, r=4, K=60.0),
                        pos_weight=5.0)(lg, tgt)
    assert torch.allclose(via_arm, direct, atol=1e-6)
    plain = build_loss("gap_ce")(lg, tgt)
    assert torch.allclose(plain, make_gap_ce()(lg, tgt), atol=1e-6)
    assert not torch.allclose(via_arm, plain)


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


def test_soft_skeleton_official_semantics():
    """Official clDice recurrence: skeleton stays a valid soft mask in [0,1]
    on soft inputs, and in the binary regime the fuzzy union reduces to set
    union (a clean 1px line's skeleton keeps the line's support)."""
    from unet.losses import SoftSkeletonize

    g = torch.Generator().manual_seed(6)
    soft = torch.nn.functional.max_pool2d(          # blobby soft probabilities
        torch.rand(2, 1, H, W, generator=g), 7, 1, 3) * 0.95
    skel = SoftSkeletonize(num_iter=5)(soft)
    assert float(skel.min()) >= 0.0
    assert float(skel.max()) <= 1.0 + 1e-6
    # binary regime: soft union == set union — a clean 1px line survives intact
    line = torch.from_numpy(hline_with_gap(gap=(0, 0)))[None, None]
    skel_line = SoftSkeletonize(num_iter=5)(line)
    assert float((skel_line * line).sum()) > 0.5 * float(line.sum())


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


@pytest.mark.parametrize("arm", ["bce", "wbce", "balance_ce",
                                 "gap_ce", "tl_ce", "gap_tl_ce",
                                 "t2_ce", "t4_ce",
                                 "gap_t2_ce", "gap_t4_ce", "gap_t2t4_ce",
                                 "bce_dice", "pstar_dice", "pstar_tversky",
                                 "focal_tversky", "sdice", "lcdice",
                                 "pstar_sdice", "pstar_lcdice",
                                 "bce_dice+cldice", "bce_dice+skelrec",
                                 "pstar_sdice+cldice", "pstar_lcdice+skelrec"])
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
