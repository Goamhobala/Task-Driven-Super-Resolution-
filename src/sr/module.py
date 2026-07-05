"""JointSRSegModule — one LightningModule for the R0/R1/R2 experiments.

Forward pipeline (single differentiable pass):

    10 m reflectance (B, 4, P, P)  [B4, B3, B2, B8]
      -> nan_to_num
      -> upsampler: SEN2SR (trainable | frozen) or bicubic     -> (B, 4, 4P, 4P)
      -> normalisation adapter: reflectance -> the baseline's frozen per-band
         z-score (differentiable affine)
      -> U-Net (baseline.model.build_model, ImageNet-pretrained encoder)
      -> road logits (B, 1, 4P, 4P)
      -> RoadSegLoss(logits, 2.5 m mask)          <- the ONLY loss (task-driven
                                                     SR: no reconstruction term)

The Figure-1 `∇L_seg × α` scaling on SEN2SR is implemented as differential
learning rates: one Adam with two param groups, `lr_sr` (low) on SEN2SR and
`lr_seg` on the U-Net, so effectively α = lr_sr / lr_seg.

Normalisation adapter: SEN2SR outputs surface reflectance, but the baseline
U-Net was built to consume per-band z-scored DN using the frozen Data.npz
stats (see RoadSegDataset). The adapter applies exactly that transform —
`(reflectance * 10000 - mean) / std` with the M0 slice of the same stats — as
a buffer-based affine, so (a) the U-Net input distribution matches R0/R1 and
the M-series baselines, and (b) gradients flow through it into SEN2SR.
(The repo baseline does NOT use ImageNet normalisation; matching R0/R1 means
matching the Data.npz z-score.)

SR warm-up: to protect the pretrained SR weights early in training, the SR
group's LR is held at 0 for the first `freeze_sr_steps` optimiser steps, then
optionally ramped linearly to `lr_sr` over `sr_lr_ramp_steps`. This is done by
scaling the group LR rather than toggling `requires_grad`, so it stays
DDP-safe (DDP fixes its gradient buckets at wrap time).
"""
from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from baseline.model import RoadSegLoss, build_model
from sr.sen2sr_loader import (
    SEN2SR_SCALE,
    BicubicUpsampler,
    TrainableSEN2SR,
    load_trainable_sen2sr,
)
from sr.data import REFLECTANCE_SCALE

# Pluggable criterion registry — selected via the `loss` hparam so the same
# module supports whichever criterion the pilot ablation picked. All entries
# must be callable as criterion(logits, target_mask).
LOSS_REGISTRY = {
    "roadseg": lambda pos_weight, alpha: RoadSegLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32), alpha=alpha
    ),
}


class JointSRSegModule(L.LightningModule):
    """R0 (bicubic) / R1 (frozen SEN2SR) / R2 (joint task-driven SEN2SR + U-Net).

    Args:
        upsampler: "sen2sr" or "bicubic".
        sen2sr_dir: dir holding the mlstac SEN2SR download (model.safetensor,
            hard_constraint.safetensor). Required when upsampler="sen2sr".
        band_mean / band_std: frozen per-band stats (DN units) for the M0
            channels, i.e. Data.npz mean/std[[0,1,2,3]] — the adapter target.
        freeze_sr: R1 — SEN2SR params get requires_grad=False and run under
            no_grad (pure preprocessing). R2 is freeze_sr=False.
        lr_sr / lr_seg: differential LRs; α = lr_sr / lr_seg.
        freeze_sr_steps / sr_lr_ramp_steps: SR warm-up (see module docstring).
        scheduler: "none" | "cosine" (cosine over the whole run, both groups).
        criterion: optional pre-built loss instance; overrides `loss`.
        log_grad_norms_every: if > 0, log per-group grad norms every N steps.
    """

    def __init__(self, upsampler: str = "sen2sr", sen2sr_dir: str | None = None,
                 band_mean: tuple = (), band_std: tuple = (),
                 encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                 loss: str = "roadseg", pos_weight: float = 1.0, alpha: float = 0.3,
                 lr_sr: float = 1e-5, lr_seg: float = 1e-3,
                 freeze_sr: bool = False, freeze_sr_steps: int = 0,
                 sr_lr_ramp_steps: int = 0, scheduler: str = "none",
                 threshold: float = 0.5, log_grad_norms_every: int = 0,
                 criterion: nn.Module | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        if upsampler == "sen2sr":
            if sen2sr_dir is None:
                raise ValueError("upsampler='sen2sr' needs sen2sr_dir")
            self.upsampler = load_trainable_sen2sr(sen2sr_dir)
            # The shipped low_pass_mask fixes the HR size -> LR patches are
            # pinned to mask_size / scale (512/4 = 128). Checked in forward.
            self._required_lr = self.upsampler.hard_constraint.low_pass_mask.shape[-1] // SEN2SR_SCALE
        elif upsampler == "bicubic":
            if freeze_sr:
                raise ValueError("freeze_sr is meaningless with the parameter-free bicubic upsampler")
            self.upsampler = BicubicUpsampler(SEN2SR_SCALE)
            self._required_lr = None
        else:
            raise ValueError(f"unknown upsampler {upsampler!r}")
        self.scale = SEN2SR_SCALE

        if freeze_sr:
            for p in self.upsampler.parameters():
                p.requires_grad = False

        # Segmentation net: the baseline's construction, unchanged — encoder
        # constancy across R0/R1/R2 (and vs the M-series) is the point of the
        # ablation. Only the upsampler treatment varies between experiments.
        self.model = build_model(in_channels=len(band_mean), encoder=encoder,
                                 encoder_weights=encoder_weights)

        # Normalisation adapter buffers (differentiable affine; see docstring).
        mean = torch.as_tensor(band_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.as_tensor(band_std, dtype=torch.float32).view(1, -1, 1, 1)
        if mean.numel() != 4 or std.numel() != 4:
            raise ValueError("band_mean/band_std must hold the 4 M0-band stats")
        self.register_buffer("band_mean", mean)
        self.register_buffer("band_std", torch.where(std == 0, torch.ones_like(std), std))

        # Injectable criterion: explicit instance wins, else built by name.
        self.criterion = criterion if criterion is not None \
            else LOSS_REGISTRY[loss](pos_weight, alpha)

        m = {"task": "binary", "threshold": threshold}
        self.val_iou = torchmetrics.JaccardIndex(**m)
        self.val_f1 = torchmetrics.F1Score(**m)
        self.test_iou = torchmetrics.JaccardIndex(**m)
        self.test_f1 = torchmetrics.F1Score(**m)
        self.test_precision = torchmetrics.Precision(**m)
        self.test_recall = torchmetrics.Recall(**m)

        self._total_steps = None  # resolved in on_train_start (for cosine)

    # ---------------------------------------------------------------- forward
    def train(self, mode: bool = True):
        """Keep a frozen SEN2SR in eval mode even when Lightning flips the
        whole module to train each epoch (harmless today — SPAN has no
        BN/dropout — but guards against upstream changes)."""
        super().train(mode)
        if self.hparams.freeze_sr:
            self.upsampler.eval()
        return self

    def _super_resolve(self, x_lr: torch.Tensor) -> torch.Tensor:
        x_lr = torch.nan_to_num(x_lr, nan=0.0, posinf=0.0, neginf=0.0)
        if self._required_lr is not None and x_lr.shape[-1] != self._required_lr:
            raise ValueError(
                f"SEN2SR's low_pass_mask pins the LR patch to "
                f"{self._required_lr}x{self._required_lr}, got {tuple(x_lr.shape[-2:])}"
            )
        if self.hparams.freeze_sr:
            with torch.no_grad():  # R1: pure preprocessing, save the graph
                return self.upsampler(x_lr)
        return self.upsampler(x_lr)

    def forward(self, x_lr: torch.Tensor) -> torch.Tensor:
        hr_reflectance = self._super_resolve(x_lr)
        # Adapter: reflectance -> the baseline's frozen per-band z-score
        # (differentiable, so segmentation gradients reach SEN2SR).
        x_seg = (hr_reflectance * REFLECTANCE_SCALE - self.band_mean) / self.band_std
        return self.model(x_seg)  # raw logits

    # ------------------------------------------------------------------ steps
    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        yi = y.int()
        self.val_iou.update(probs, yi)
        self.val_f1.update(probs, yi)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=x.size(0))
        self.log_dict({"val_iou": self.val_iou, "val_f1": self.val_f1},
                      on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        probs = torch.sigmoid(self(x))
        yi = y.int()
        self.test_iou.update(probs, yi)
        self.test_f1.update(probs, yi)
        self.test_precision.update(probs, yi)
        self.test_recall.update(probs, yi)
        self.log_dict({
            "test_iou": self.test_iou, "test_f1": self.test_f1,
            "test_precision": self.test_precision, "test_recall": self.test_recall,
        }, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx):
        return torch.sigmoid(self(batch[0]))

    # ------------------------------------------------------------- optimisers
    def configure_optimizers(self):
        """One Adam, two param groups — the differential-LR mechanism for α.

        `requires_grad` filtering drops (a) all SEN2SR params under freeze_sr
        (R1), and (b) SEN2SR's collapsed eval_conv relics, which ship frozen.
        """
        groups = [{
            "params": [p for p in self.model.parameters() if p.requires_grad],
            "lr": self.hparams.lr_seg, "name": "seg", "base_lr": self.hparams.lr_seg,
        }]
        sr_params = [p for p in self.upsampler.parameters() if p.requires_grad]
        if sr_params:
            groups.append({"params": sr_params, "lr": self.hparams.lr_sr,
                           "name": "sr", "base_lr": self.hparams.lr_sr})
        return torch.optim.Adam(groups)

    def _sr_warmup_factor(self, step: int) -> float:
        """0 while frozen, then linear ramp to 1 (see module docstring)."""
        hold, ramp = self.hparams.freeze_sr_steps, self.hparams.sr_lr_ramp_steps
        if step < hold:
            return 0.0
        if ramp > 0 and step < hold + ramp:
            return (step - hold + 1) / ramp
        return 1.0

    def _cosine_factor(self, step: int) -> float:
        if self.hparams.scheduler != "cosine" or not self._total_steps:
            return 1.0
        t = min(step / max(self._total_steps, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def on_train_start(self):
        self._total_steps = int(self.trainer.estimated_stepping_batches)
        alpha = self.hparams.lr_sr / max(self.hparams.lr_seg, 1e-12)
        self.print(f"[sr] effective alpha = lr_sr/lr_seg = {alpha:.3g}; "
                   f"sr warm-up: hold {self.hparams.freeze_sr_steps} steps, "
                   f"ramp {self.hparams.sr_lr_ramp_steps} steps")

    def on_train_batch_start(self, batch, batch_idx):
        # Manual per-step LR control (instead of a torch scheduler object) so
        # the SR warm-up and cosine compose without fighting over group lr.
        step = self.global_step
        cos = self._cosine_factor(step)
        for g in self.trainer.optimizers[0].param_groups:
            factor = cos * (self._sr_warmup_factor(step) if g["name"] == "sr" else 1.0)
            g["lr"] = g["base_lr"] * factor
        if step % 50 == 0:
            self.log_dict({f"lr_{g['name']}": g["lr"]
                           for g in self.trainer.optimizers[0].param_groups})

    # ------------------------------------------------------------ diagnostics
    def grad_norms(self) -> dict[str, float]:
        """L2 grad norm per component — call after backward. The 'sr' entry
        makes the differential-LR/α behaviour visible next to 'seg'."""
        out = {}
        for name, mod in (("sr", self.upsampler), ("seg", self.model)):
            sq = sum((p.grad.norm().item() ** 2) for p in mod.parameters()
                     if p.grad is not None)
            out[name] = sq ** 0.5
        return out

    def on_after_backward(self):
        every = self.hparams.log_grad_norms_every
        if every > 0 and self.global_step % every == 0:
            self.log_dict({f"grad_norm_{k}": v for k, v in self.grad_norms().items()})
