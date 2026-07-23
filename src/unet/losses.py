"""Slot-based losses for the loss-ablation protocol (Phases A-C).

Taxonomy (protocol §3.1):
    L = w_pix · L_pixel + w_reg · L_region + w_skel · L_skeleton

Pixel slot    : plain BCE, or a weighted CE (GapLoss / Topological Loss are
                weighted-CE *replacements* for BCE, not additive extras).
Region slot   : Dice, Tversky, or none.
Skeleton slot : clDice, Skeleton Recall, or none.

Fairness rules implemented here:
  * §4.4 scale normalization — every weighted CE returns sum(W·ce)/sum(W)
    (a weighted mean), so its expected magnitude matches plain BCE and LR
    search spaces / mixing weights are comparable across arms.
  * Weight maps are computed from the *detached* prediction under no_grad;
    gradients flow only through the CE term (as in both source papers).
  * §4.5 skeleton warmup — ComposedLoss holds the skeleton weight at 0 until
    `warmup_start`, then ramps linearly to target over `warmup_ramp` epochs.

Paper provenance (verified against the published texts, 2026-07-15):
  * GapLoss  — Yuan & Xu 2022, Remote Sensing 14(10):2422, Algorithm 1.
    Endpoint = skeleton pixel with exactly one 8-neighbour. Each pixel's
    weight is K·N (N = endpoints in a (2r+1)² window, paper: 9×9 ⇒ r=4),
    else 1. Paper's tuned K=60. Protocol tunes r ∈ {3,5,9}.
  * Topological Loss — Nanni, Brahnam & Loreggia 2024, IEEE Access 12:74218,
    Algorithm 1. Four length-ℓ line filters (paper ℓ=5) detect conv==2
    positions on the skeleton (endpoint vicinity per orientation); a second
    conv paints weight 10 along the same orientation; per-direction maps are
    capped at 10 with zeros→1; W = D+E+F+G, then W≥10→10.
    NOTE the original leaves base weight at 4 (1 from each direction).
    The protocol follows Giannini et al. 2026 and resets W==base → 1
    (`base_reset=True`). Protocol tunes ℓ ∈ {3,5,7} (physical lookahead:
    ℓ px = 10·ℓ m at our GSD).
  * clDice — Shit et al. 2021 (CVPR). Soft skeleton via iterated min/max
    pooling; protocol uses k=5 iterations (covers 1-6 px radii at 10 m).
  * Skeleton Recall — Kirchhoff et al. 2024 (ECCV). Soft recall of the
    prediction over the tubed GT skeleton; GT-side only, so near-free.

Binary-head note: the papers write softmax/2-class; this repo uses a single
sigmoid channel. For binary segmentation the two are equivalent; we binarize
sigmoid(logits) at 0.5 during weight-map construction, as in the papers.

pos_weight: the protocol's arms use *plain* CE (pos_weight=None). The old
RoadSegLoss baseline used a positive-class weight — that is itself a
distribution-slot reweighting, so it is OFF by default everywhere here.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # skimage only needed for the weighted-CE arms (CPU skeletonization)
    from skimage.morphology import skeletonize as _sk_skeletonize
except ImportError:  # pragma: no cover
    _sk_skeletonize = None

EPS = 1.0  # smoothing, protocol §4.3


# --------------------------------------------------------------------------
# weight-map builders (CPU skeletonization + batched torch convolutions)
# --------------------------------------------------------------------------

def _skeletonize_batch(binary: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) bool/float 0-1 -> float skeleton, on CPU (skimage, Zhang)."""
    if _sk_skeletonize is None:
        raise ImportError("scikit-image is required for gap_ce / tl_ce arms")
    b_np = binary.detach().cpu().numpy() > 0.5
    out = np.zeros(b_np.shape, dtype=np.float32)
    for i in range(b_np.shape[0]):
        out[i, 0] = _sk_skeletonize(b_np[i, 0])
    return torch.from_numpy(out)


def gap_weight_map(prob: torch.Tensor, r: int = 4, K: float = 60.0,
                   thresh: float = 0.5) -> torch.Tensor:
    """GapLoss weight map (Yuan & Xu 2022, Algorithm 1).

    prob: detached sigmoid probabilities (B,1,H,W). Returns W on CPU.
    W[p] = K · (#endpoints in (2r+1)×(2r+1) window around p) if any, else 1.
    Paper defaults: 9×9 window (r=4), K=60 (grid-searched on Massachusetts).
    """
    with torch.no_grad():
        skel = _skeletonize_batch(prob > thresh).to(prob.device)  # convs on GPU
        k3 = torch.ones(1, 1, 3, 3, device=prob.device)
        nbrs = F.conv2d(skel, k3, padding=1) - skel          # 8-neighbour count
        endpoints = ((skel > 0.5) & (nbrs.round() == 1)).float()
        win = 2 * r + 1
        cnt = F.conv2d(endpoints, torch.ones(1, 1, win, win, device=prob.device),
                       padding=r).round()
        W = torch.where(cnt > 0, K * cnt, torch.ones_like(cnt))
    return W


def _tl_kernels(ell: int) -> list[torch.Tensor]:
    """The four directional line filters of Nanni et al. 2024 (length ell)."""
    v = torch.ones(ell, 1)                # vertical span
    h = torch.ones(1, ell)                # horizontal span
    d = torch.eye(ell)                    # main diagonal
    a = torch.flip(torch.eye(ell), dims=[1])   # anti-diagonal
    return [v, h, d, a]


def _rotations(base: torch.Tensor) -> list[torch.Tensor]:
    """base + its 90/180/270 deg rotations (Giannini et al.: the other three
    filters are generated by rotating the base filter).

    NB the curvature kernels are NOT 180 deg-symmetric, so conv vs cross-
    correlation matters per kernel — but the 4-rotation SET is closed under
    180 deg rotation, so the summed weight map W is identical either way
    (cross-correlating with k equals convolving with rot180(k), which is also
    in the set and pairs with its own C map). F.conv2d (cross-correlation)
    therefore reproduces the paper's W exactly.
    """
    return [torch.rot90(base, k, dims=(0, 1)) for k in range(4)]


def t2_cells(n: int = 5) -> list[tuple[int, int]]:
    """T2 base-filter cells (Giannini et al. 2026, Eq. 4): a discrete quarter
    circle traced by alternating down/right steps from (0, 1).
    (i_t, j_t) = (i0 + floor((t+1)/2), j0 + floor(t/2)), t = 0..2(n-2)."""
    i0, j0 = 0, 1
    return [(i0 + (t + 1) // 2, j0 + t // 2) for t in range(2 * (n - 2) + 1)]


def t4_cells(n: int = 5) -> list[tuple[int, int]]:
    """T4 base-filter cells (Giannini et al. 2026, Eq. 5): a discrete
    semicircle — zig-zag descent to the middle row, a length-L horizontal
    traversal, and a mirrored zig-zag ascent. n odd (m is unambiguous:
    ceil((n-1)/2) == floor((n-1)/2))."""
    if n % 2 == 0:
        raise ValueError(f"t4_cells expects odd n (protocol grid), got {n}")
    m = (n - 1) // 2          # middle row
    L = 3                     # middle-section length (n odd)
    j0 = m - 1                # column where the middle section starts
    tb = 2 * m - 1            # first middle-section step
    je = j0 + L - 1
    te = n                    # last middle-section step (= tb + L - 1)
    cells = []
    for t in range(n + tb + 1):                     # t = 0..T_max
        if t < tb:
            cells.append(((t + 1) // 2, t // 2))
        elif t <= te:
            cells.append((m, j0 + (t - tb)))
        else:
            cells.append((m - (t - te + 1) // 2, je + (t - te) // 2))
    return cells


def _cells_to_kernel(cells: list[tuple[int, int]], n: int) -> torch.Tensor:
    k = torch.zeros(n, n)
    for i, j in cells:
        k[i, j] = 1.0
    return k


def t2_kernels(n: int = 5) -> list[torch.Tensor]:
    """The four T2 quarter-circle filters (wide, gradual curves)."""
    return _rotations(_cells_to_kernel(t2_cells(n), n))


def t4_kernels(n: int = 5) -> list[torch.Tensor]:
    """The four T4 semicircle filters (tight, abrupt turns)."""
    return _rotations(_cells_to_kernel(t4_cells(n), n))


def tl_weight_map(prob: torch.Tensor, ell: int = 5, thresh: float = 0.375,
                  base_reset: bool = True,
                  extra_kernels: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Topological Loss weight map (Nanni et al. 2024, Algorithm 1).

    Per direction k: C = (conv(skel,k)==2);  D = conv(C, 10·k);
                     D = min(D,10);  D[D==0] = 1.
    W = ΣD;  W[W>=10] = 10;  and (Giannini refinement) W[W==n_dirs] = 1.
    `extra_kernels` is the hook for T2/T4 curvature filters (base becomes 8).
    All kernels here are 180°-symmetric, so conv == cross-correlation.
    """
    kernels = _tl_kernels(ell) + list(extra_kernels or [])
    base = float(len(kernels))
    with torch.no_grad():
        skel = _skeletonize_batch(prob > thresh).to(prob.device)  # convs on GPU
        W = torch.zeros_like(skel)
        for k in kernels:
            k = k.to(prob.device)
            kh, kw = k.shape
            pad = (kh // 2, kw // 2)
            k4 = k.reshape(1, 1, kh, kw)
            C = (F.conv2d(skel, k4, padding=pad).round() == 2).float()
            D = F.conv2d(C, 10.0 * k4, padding=pad)
            D = torch.clamp(D, max=10.0)
            D = torch.where(D == 0, torch.ones_like(D), D)
            W = W + D
        W = torch.where(W >= 10.0, torch.full_like(W, 10.0), W)
        if base_reset:
            W = torch.where(W == base, torch.ones_like(W), W)
    return W


# --------------------------------------------------------------------------
# pixel slot
# --------------------------------------------------------------------------

class WeightedCE(nn.Module):
    """BCE-with-logits, optionally reweighted by a prediction-derived map.

    weight_fn: callable prob(detached, B,1,H,W) -> W. None => plain BCE.
    normalize (§4.4): loss = sum(W·ce)/sum(W), so E[loss] ≈ E[BCE] and the
    weighting only *redistributes* gradient, never inflates it.
    """

    def __init__(self, weight_fn: Callable | None = None, normalize: bool = True,
                 pos_weight: torch.Tensor | float | None = None):
        super().__init__()
        self.weight_fn = weight_fn
        self.normalize = normalize
        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.tensor(float(pos_weight))
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        if self.weight_fn is None:
            return ce.mean()
        with torch.no_grad():
            W = self.weight_fn(torch.sigmoid(logits).detach()).to(ce.device)
        if self.normalize:
            return (W * ce).sum() / W.sum()
        return (W * ce).mean()


def make_gap_ce(r: int = 4, K: float = 60.0, normalize: bool = True) -> WeightedCE:
    return WeightedCE(lambda p: gap_weight_map(p, r=r, K=K), normalize=normalize)


def make_tl_ce(ell: int = 5, normalize: bool = True, theta: float = 0.375,
               extra_kernels: list[torch.Tensor] | None = None) -> WeightedCE:
    """TL / T2 / T4 weighted CE. ``theta`` is the binarization threshold the
    weight map is built at. Default 0.375 = the Giannini/Nanni papers'
    hard-coded value AND the protocol's Appendix-B centre; the protocol grid
    is θ ∈ {0.375, 0.5}. NB the first l3_tl_ce run predates this knob and
    trained at 0.5 — treat it as the θ=0.5 grid point (run names now always
    carry θ, so it cannot be confused with new runs)."""
    return WeightedCE(
        lambda p: tl_weight_map(p, ell=ell, thresh=theta, extra_kernels=extra_kernels),
        normalize=normalize)


def make_gap_tl_ce(r: int = 4, K: float = 60.0, ell: int = 5,
                   theta: float = 0.375, normalize: bool = True,
                   extra_kernels: list[torch.Tensor] | None = None) -> WeightedCE:
    """GL+TL blended pixel slot (Nanni et al. 2024, Table 2: GL+TL and
    GL+TL+DI are their best compounds on 3 of 4 datasets).

    NOT a slot-taxonomy violation: both parents are weighted CEs over the
    SAME ce map, so their average is itself a single weighted CE whose
    attention map blends endpoint buffers (GL) with directional corridors
    (TL). Each map is normalized to mean 1 before summing, which makes this
    EXACTLY 0.5·gap_ce + 0.5·tl_ce under the §4.4 normalization (unit test
    asserts the identity), with equal expected contribution from each parent
    despite their wildly different raw scales (GL up to K·N vs TL's cap 10).
    GL keeps its own 0.5 binarization (Yuan & Xu); θ applies to the TL side.
    The paper's GL+TL+DI is then Phase B's ``pstar_dice`` with this as P*.
    """
    def blend(p: torch.Tensor) -> torch.Tensor:
        wg = gap_weight_map(p, r=r, K=K)
        wt = tl_weight_map(p, ell=ell, thresh=theta, extra_kernels=extra_kernels)
        return wg / wg.mean().clamp_min(1e-8) + wt / wt.mean().clamp_min(1e-8)

    return WeightedCE(blend, normalize=normalize)


# --------------------------------------------------------------------------
# region slot
# --------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """1 − (2Σyp + ε)/(Σy + Σp + ε), per sample then mean (protocol §4.3)."""

    def __init__(self, smooth: float = EPS):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        inter = (p * targets).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (1 - (2 * inter + self.smooth) / (denom + self.smooth)).mean()


class TverskyLoss(nn.Module):
    """Protocol §4.3: 1 − (Σyp+ε)/(Σyp + α·Σy(1−p) + (1−α)·Σ(1−y)p + ε).

    α multiplies the false-negative term: α > 0.5 penalizes FN more ⇒ favors
    recall (the protocol biases the search recall-side given incomplete
    labels and thin structures). α = 0.5 reduces to Dice.
    """

    def __init__(self, alpha: float = 0.7, smooth: float = EPS):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p * targets).sum(dim=(1, 2, 3))
        fn = ((1 - p) * targets).sum(dim=(1, 2, 3))
        fp = (p * (1 - targets)).sum(dim=(1, 2, 3))
        ti = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)
        return (1 - ti).mean()


class FocalTverskyLoss(TverskyLoss):
    """(1 − TI)^exponent (Abraham & Khan 2019; γ=4/3 ⇒ exponent 0.75).
    Phase B literature baseline (arm A1)."""

    def __init__(self, alpha: float = 0.7, exponent: float = 0.75, smooth: float = EPS):
        super().__init__(alpha=alpha, smooth=smooth)
        self.exponent = exponent

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p * targets).sum(dim=(1, 2, 3))
        fn = ((1 - p) * targets).sum(dim=(1, 2, 3))
        fp = (p * (1 - targets)).sum(dim=(1, 2, 3))
        ti = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)
        return ((1 - ti) ** self.exponent).mean()


# --------------------------------------------------------------------------
# skeleton slot
# --------------------------------------------------------------------------

class SoftSkeletonize(nn.Module):
    """Differentiable morphological skeleton — line-faithful to the official
    implementation (Shit et al. 2021):
    https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py

    Per level: delta = relu(img − open(img)) is the structure thinner than the
    current erosion depth; levels merge with the PROBABILISTIC union
    ``skel + relu(delta − skel·delta)`` = 1 − (1−skel)(1−delta): overlapping
    level responses combine as a fuzzy OR (the skel·delta term shrinks the
    overlap share) instead of stacking, and gradients stay DENSE through both
    operands (∂/∂delta = 1−skel, ∂/∂skel = 1−delta) — unlike a max-union
    whose subgradient flows only through the winning branch. NB the naive sum
    is already ≤ img by telescoping (delta_j ≤ img_j − img_{j+1}), so the
    union's value is in its semantics + gradients, not boundedness. The
    depth-0 delta is taken BEFORE the loop, so num_iter=k covers erosion
    depths 0..k.

    (An earlier revision here used torch.max and depths 0..k−1 —
    binary-equivalent but not the paper's soft regime; fixed 2026-07-22.)
    """

    def __init__(self, num_iter: int = 5):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        skel = F.relu(img - self.soft_open(img))
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            delta = F.relu(img - self.soft_open(img))
            skel = skel + F.relu(delta - skel * delta)
        return skel


class SoftclDice(nn.Module):
    """Pure clDice term — VERBATIM port of the official ``soft_cldice``
    forward (jocpae/clDice, cldice.py), adapted only at the boundary for this
    repo's single-sigmoid-channel head:

      * sums are BATCH-GLOBAL (the official ``torch.sum`` has no dim —
        tprec/tsens pool over the whole batch, micro-style), NOT per-sample;
      * ``smooth=1.`` on both ratios; NO extra epsilon in the harmonic mean
        (the official relies on smooth > 0 keeping tprec+tsens positive);
      * probabilities via sigmoid at the entry (official takes y_pred
        post-activation); ``exclude_background`` is meaningless for a
        1-channel binary head and is omitted.

    Protocol Phase C: skel_iters=5 (GSD-adapted; the official class ignores
    its ``iter_`` arg and hard-codes num_iter=10 — here the parameter is
    real). The DECLARED protocol deviation is composition only: the arm pairs
    this term with the BCE+Dice anchor via ComposedLoss (§4.3) instead of the
    official ``soft_dice_cldice`` (1−α)·Dice + α·clDice, so the comparison
    isolates the skeleton term."""

    def __init__(self, skel_iters: int = 5, smooth: float = 1.0):
        super().__init__()
        self.skeletonize = SoftSkeletonize(num_iter=skel_iters)
        self.smooth = smooth

    def forward(self, logits, targets):
        y_pred = torch.sigmoid(logits)
        skel_pred = self.skeletonize(y_pred)
        skel_true = self.skeletonize(targets)
        tprec = (torch.sum(skel_pred * targets) + self.smooth) / \
                (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / \
                (torch.sum(skel_true) + self.smooth)
        return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


class SkeletonRecallLoss(nn.Module):
    """Skeleton Recall (Kirchhoff et al. 2024): soft recall of the prediction
    over the tubed GT skeleton. GT-side skeletons only (cheap). If the
    dataloader provides a precomputed tube pass it via `tube=`; otherwise it
    is computed here on CPU per batch (fine at 256² tiles; cache later if hot).
    """

    def __init__(self, tube_radius: int = 1, smooth: float = EPS):
        super().__init__()
        self.tube_radius = tube_radius
        self.smooth = smooth

    def make_tube(self, targets: torch.Tensor) -> torch.Tensor:
        skel = _skeletonize_batch(targets > 0.5).to(targets.device)
        k = 2 * self.tube_radius + 1
        return F.max_pool2d(skel, k, stride=1, padding=self.tube_radius)

    def forward(self, logits, targets, tube: torch.Tensor | None = None):
        p = torch.sigmoid(logits)
        if tube is None:
            with torch.no_grad():
                tube = self.make_tube(targets)
        rec = ((p * tube).sum(dim=(1, 2, 3)) + self.smooth) / \
              (tube.sum(dim=(1, 2, 3)) + self.smooth)
        return (1 - rec).mean()


# --------------------------------------------------------------------------
# composition + warmup (§3.1, §4.5)
# --------------------------------------------------------------------------

class ComposedLoss(nn.Module):
    """w_pix·pixel + w_reg·region + w_skel(epoch)·skeleton.

    §4.5 warmup: skeleton weight is 0 for epoch < warmup_start, then ramps
    linearly to w_skel over warmup_ramp epochs. From-scratch protocol runs:
    warmup_start=30, warmup_ramp=10 (of 100). Call .set_epoch(e) each epoch
    (or pass epoch= in forward).
    """

    def __init__(self, pixel: nn.Module | None = None, region: nn.Module | None = None,
                 skeleton: nn.Module | None = None, w_pix: float = 1.0,
                 w_reg: float = 0.0, w_skel: float = 0.0,
                 warmup_start: int = 0, warmup_ramp: int = 0):
        super().__init__()
        self.pixel, self.region, self.skeleton = pixel, region, skeleton
        self.w_pix, self.w_reg, self.w_skel = w_pix, w_reg, w_skel
        self.warmup_start, self.warmup_ramp = warmup_start, warmup_ramp
        self._epoch = None

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def skeleton_weight(self, epoch: int | None) -> float:
        if self.skeleton is None or self.w_skel == 0:
            return 0.0
        if epoch is None or self.warmup_ramp == 0:
            return self.w_skel
        if epoch < self.warmup_start:
            return 0.0
        frac = min(1.0, (epoch - self.warmup_start) / self.warmup_ramp)
        return self.w_skel * frac

    def forward(self, logits, targets, epoch: int | None = None):
        epoch = self._epoch if epoch is None else epoch
        loss = logits.sum() * 0.0
        parts = {}
        if self.pixel is not None and self.w_pix:
            parts["pixel"] = self.pixel(logits, targets)
            loss = loss + self.w_pix * parts["pixel"]
        if self.region is not None and self.w_reg:
            parts["region"] = self.region(logits, targets)
            loss = loss + self.w_reg * parts["region"]
        ws = self.skeleton_weight(epoch)
        if ws > 0:
            parts["skeleton"] = self.skeleton(logits, targets)
            loss = loss + ws * parts["skeleton"]
        self.last_parts = {k: v.item() for k, v in parts.items()}
        return loss


# --------------------------------------------------------------------------
# arm factory
# --------------------------------------------------------------------------

def _pixel_slot(name: str, hp: dict) -> nn.Module:
    if name == "bce":
        # STRICTLY plain CE — the distribution-family anchor stays literature-
        # standard even when a pos_weight hp is floating around in `hp`.
        return WeightedCE(None, pos_weight=None)
    if name == "wbce":
        # pos-weighted CE as its OWN pixel-slot candidate (amendment
        # 2026-07-20): a static class-level reweighting, taxonomically the
        # same slot as GapLoss/TL's spatially adaptive ones. The anchor
        # bce_dice stays plain (Giannini Eq. 3 / CoANet / Xu et al.);
        # "bce_dice + tunable pos_weight" = pstar_dice with pstar=wbce.
        return WeightedCE(None, pos_weight=hp.get("pos_weight", 5.0))
    if name == "gap_ce":
        return make_gap_ce(r=hp.get("gap_r", 4), K=hp.get("gap_k", 60.0))
    if name == "tl_ce":
        return make_tl_ce(ell=hp.get("tl_ell", 5), theta=hp.get("tl_theta", 0.375))
    if name == "gap_tl_ce":
        return make_gap_tl_ce(r=hp.get("gap_r", 4), K=hp.get("gap_k", 60.0),
                              ell=hp.get("tl_ell", 5),
                              theta=hp.get("tl_theta", 0.375))
    if name in ("t2_ce", "t4_ce"):
        # Giannini et al. 2026: TL's four line filters + four curvature filters
        # (T2 quarter-circles / T4 semicircles), base weight 8 -> reset to 1.
        # Curvature kernel size follows tl_ell (the paper retains 5x5 for
        # receptive-field consistency with TL; same argument at our GSD).
        n = hp.get("tl_ell", 5)
        extra = t2_kernels(n) if name == "t2_ce" else t4_kernels(n)
        return make_tl_ce(ell=n, theta=hp.get("tl_theta", 0.375), extra_kernels=extra)
    raise ValueError(f"unknown pixel slot {name!r}")


def build_loss(arm: str, **hp) -> ComposedLoss:
    """Protocol arms. Phase A: bce | gap_ce | tl_ce | t2_ce | t4_ce.
    Anchors/compounds: bce_dice | pstar_dice | pstar_tversky | focal_tversky.
    Phase C: append '+cldice' or '+skelrec' to any compound,
    e.g. 'bce_dice+cldice' (α via cl_alpha, template weights rescaled by 1−α).

    hp: gap_r, gap_k, tl_ell, tl_theta, tversky_alpha, cl_alpha, cl_iters,
        sr_w, sr_radius, pstar ('bce'|'wbce'|'gap_ce'|'tl_ce'|'gap_tl_ce'|
        't2_ce'|'t4_ce'), pos_weight, mix_w (pstar_* only; bce_dice anchor
        stays frozen 0.5/0.5), warmup_start, warmup_ramp.
    """
    base, _, skel = arm.partition("+")
    wu = dict(warmup_start=hp.get("warmup_start", 30),
              warmup_ramp=hp.get("warmup_ramp", 10))

    if base in ("bce", "wbce", "gap_ce", "tl_ce", "gap_tl_ce", "t2_ce", "t4_ce"):
        cfg = dict(pixel=_pixel_slot(base, hp), w_pix=1.0)
    elif base == "bce_dice":
        # The ANCHOR: frozen at the literature's 0.5/0.5 (Giannini Eq. 3) —
        # deliberately ignores mix_w, or it stops anchoring anything.
        cfg = dict(pixel=_pixel_slot("bce", hp), region=DiceLoss(),
                   w_pix=0.5, w_reg=0.5)
    elif base == "pstar_dice":
        # Phase B amendment (2026-07-21): the P*<->region mixing ratio is the
        # compound's one genuinely free parameter (no paper default exists), so
        # it is searchable: L = (1-mix_w)·P* + mix_w·region. 0.5 = the old
        # frozen behaviour; unet.tune_loss searches it on short trials.
        mw = hp.get("mix_w", 0.5)
        cfg = dict(pixel=_pixel_slot(hp.get("pstar", "bce"), hp), region=DiceLoss(),
                   w_pix=1.0 - mw, w_reg=mw)
    elif base == "pstar_tversky":
        mw = hp.get("mix_w", 0.5)
        cfg = dict(pixel=_pixel_slot(hp.get("pstar", "bce"), hp),
                   region=TverskyLoss(alpha=hp.get("tversky_alpha", 0.7)),
                   w_pix=1.0 - mw, w_reg=mw)
    elif base == "focal_tversky":
        cfg = dict(pixel=None, w_pix=0.0,
                   region=FocalTverskyLoss(alpha=hp.get("tversky_alpha", 0.7)),
                   w_reg=1.0)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    if skel == "cldice":
        # (1−α)·B* + α·clDice — protocol arm 7 (α=0.3, k=5)
        a = hp.get("cl_alpha", 0.3)
        cfg["w_pix"] = cfg.get("w_pix", 0.0) * (1 - a)
        cfg["w_reg"] = cfg.get("w_reg", 0.0) * (1 - a)
        cfg.update(skeleton=SoftclDice(skel_iters=hp.get("cl_iters", 5)), w_skel=a, **wu)
    elif skel == "skelrec":
        # B* + w·SkelRecall — protocol arm 8 (w=1, tube r=1; additive as in
        # Kirchhoff et al.). NOTE the effective anchor weight differs from the
        # clDice arm's convex mix — flagged in the protocol review.
        cfg.update(skeleton=SkeletonRecallLoss(tube_radius=hp.get("sr_radius", 1)),
                   w_skel=hp.get("sr_w", 1.0), **wu)
    elif skel:
        raise ValueError(f"unknown skeleton slot {skel!r}")

    return ComposedLoss(**cfg)


PHASE_A_ARMS = ["bce", "wbce", "gap_ce", "tl_ce", "gap_tl_ce", "t2_ce", "t4_ce"]
