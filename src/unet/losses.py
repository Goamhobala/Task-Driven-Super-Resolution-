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


def tl_weight_map(prob: torch.Tensor, ell: int = 5, thresh: float = 0.5,
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
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.weight_fn = weight_fn
        self.normalize = normalize
        if pos_weight is not None:
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


def make_tl_ce(ell: int = 5, normalize: bool = True,
               extra_kernels: list[torch.Tensor] | None = None) -> WeightedCE:
    return WeightedCE(
        lambda p: tl_weight_map(p, ell=ell, extra_kernels=extra_kernels),
        normalize=normalize)


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
    """Differentiable morphological skeleton (Shit et al. 2021), min/max pooling."""

    def __init__(self, num_iter: int = 5):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def forward(self, img):
        skel = torch.zeros_like(img)
        for _ in range(self.num_iter):
            opened = self.soft_dilate(self.soft_erode(img))
            skel = torch.max(skel, F.relu(img - opened))
            img = self.soft_erode(img)
        return skel


class SoftclDice(nn.Module):
    """Pure clDice term (Shit et al. 2021) — no Dice mixed in, so slot
    composition stays explicit. Protocol Phase C: k=5 iterations."""

    def __init__(self, skel_iters: int = 5, smooth: float = EPS):
        super().__init__()
        self.skeletonize = SoftSkeletonize(num_iter=skel_iters)
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        skel_p = self.skeletonize(p)
        skel_t = self.skeletonize(targets)
        tprec = ((skel_p * targets).sum(dim=(1, 2, 3)) + self.smooth) / \
                (skel_p.sum(dim=(1, 2, 3)) + self.smooth)
        tsens = ((skel_t * p).sum(dim=(1, 2, 3)) + self.smooth) / \
                (skel_t.sum(dim=(1, 2, 3)) + self.smooth)
        cl = 2 * tprec * tsens / (tprec + tsens + 1e-8)
        return (1 - cl).mean()


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
        return WeightedCE(None, pos_weight=hp.get("pos_weight"))
    if name == "gap_ce":
        return make_gap_ce(r=hp.get("gap_r", 4), K=hp.get("gap_k", 60.0))
    if name == "tl_ce":
        return make_tl_ce(ell=hp.get("tl_ell", 5))
    if name in ("t2_ce", "t4_ce"):
        raise NotImplementedError(
            "T2/T4 need the curvature kernels from Giannini et al. 2026 — "
            "pass them via tl_weight_map(extra_kernels=...) once specified.")
    raise ValueError(f"unknown pixel slot {name!r}")


def build_loss(arm: str, **hp) -> ComposedLoss:
    """Protocol arms. Phase A: bce | gap_ce | tl_ce (t2_ce/t4_ce pending).
    Anchors/compounds: bce_dice | pstar_dice | pstar_tversky | focal_tversky.
    Phase C: append '+cldice' or '+skelrec' to any compound,
    e.g. 'bce_dice+cldice' (α via cl_alpha, template weights rescaled by 1−α).

    hp: gap_r, gap_k, tl_ell, tversky_alpha, cl_alpha, cl_iters, sr_w, sr_radius,
        pstar ('bce'|'gap_ce'|'tl_ce'), pos_weight, warmup_start, warmup_ramp.
    """
    base, _, skel = arm.partition("+")
    wu = dict(warmup_start=hp.get("warmup_start", 30),
              warmup_ramp=hp.get("warmup_ramp", 10))

    if base in ("bce", "gap_ce", "tl_ce", "t2_ce", "t4_ce"):
        cfg = dict(pixel=_pixel_slot(base, hp), w_pix=1.0)
    elif base == "bce_dice":
        cfg = dict(pixel=_pixel_slot("bce", hp), region=DiceLoss(),
                   w_pix=0.5, w_reg=0.5)
    elif base == "pstar_dice":
        cfg = dict(pixel=_pixel_slot(hp.get("pstar", "bce"), hp), region=DiceLoss(),
                   w_pix=0.5, w_reg=0.5)
    elif base == "pstar_tversky":
        cfg = dict(pixel=_pixel_slot(hp.get("pstar", "bce"), hp),
                   region=TverskyLoss(alpha=hp.get("tversky_alpha", 0.7)),
                   w_pix=0.5, w_reg=0.5)
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


PHASE_A_ARMS = ["bce", "gap_ce", "tl_ce"]  # + t2_ce/t4_ce once kernels land
