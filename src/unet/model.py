"""UNet model + PyTorch Lightning wrapper for binary road segmentation.

Training and eval are both patch-based: train draws random 256 crops from the 512
tiles; val/test iterate the 4 deterministic non-overlapping quadrants of each tile.
Because the quadrants partition each tile exactly, the torchmetrics IoU/F1 (global
TP/FP/FN, DDP-synced) is a once-per-pixel score, not a per-overlapping-tile average.
"""
from __future__ import annotations

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex


def build_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    """Plain segmentation-models-pytorch UNet."""
    if encoder_weights in (None, "none", "None", ""):
        encoder_weights = None
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


class UNetLightning(pl.LightningModule):
    """UNet + (Dice + weighted BCE); per-crop train/val/test IoU+F1 over the tiles.

    ``loss_arm`` switches the training loss to a slot-composed arm from the
    loss-ablation protocol (``unet.losses.build_loss``: ``bce``, ``gap_ce``,
    ``tl_ce``, ``bce_dice``, ``pstar_*``, ``focal_tversky``, ``<base>+cldice``,
    ``<base>+skelrec``). ``None`` (default) keeps the legacy Dice + pos-weighted
    BCE. Arms use PLAIN CE (protocol: ``pos_weight`` is itself a
    distribution-slot reweighting, so it only applies to the legacy loss). The
    arm + its hyperparameters live in ``hparams``, so a checkpoint records
    exactly which loss trained it and ``benchmarking`` can group runs on it.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        lr: float = 1e-3,
        pos_weight: float = 5.0,
        bands: tuple[int, ...] = (21, 22, 23),
        image_size: int = 256,
        threshold: float = 0.5,
        normalize: bool = True,
        norm_mean: list[float] | None = None,
        norm_std: list[float] | None = None,
        loss_arm: str | None = None,
        pstar: str = "bce",
        gap_r: int = 4,
        gap_k: float = 60.0,
        tl_ell: int = 5,
        tversky_alpha: float = 0.7,
        cl_alpha: float = 0.3,
        cl_iters: int = 5,
        sr_w: float = 1.0,
        sr_radius: int = 1,
        warmup_start: int = 30,
        warmup_ramp: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(encoder_name, encoder_weights, in_channels, classes)
        if loss_arm:
            from unet.losses import build_loss

            self.criterion = build_loss(
                loss_arm, pstar=pstar, gap_r=gap_r, gap_k=gap_k, tl_ell=tl_ell,
                tversky_alpha=tversky_alpha, cl_alpha=cl_alpha, cl_iters=cl_iters,
                sr_w=sr_w, sr_radius=sr_radius,
                warmup_start=warmup_start, warmup_ramp=warmup_ramp,
            )
            self.dice_loss = None
        else:
            self.criterion = None
            self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # Train: per-patch metrics over the random crops (cheap, not stitched).
        self.train_iou = BinaryJaccardIndex()
        self.train_f1 = BinaryF1Score()
        # Per-crop val/test metrics (global TP/FP/FN over the quadrants; auto DDP-sync).
        self.val_iou = BinaryJaccardIndex()
        self.val_f1 = BinaryF1Score()
        self.test_iou = BinaryJaccardIndex()
        self.test_f1 = BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, masks):
        if self.criterion is not None:
            return self.criterion(logits, masks)
        # Legacy loss: Dice handles overlap; weighted BCE pushes the sparse road
        # class so the model can't minimise loss by predicting all-background.
        dice = self.dice_loss(logits, masks)
        pos_weight = torch.tensor(self.hparams.pos_weight, device=logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, masks, pos_weight=pos_weight
        )
        return dice + bce

    def on_train_epoch_start(self):
        # §4.5 skeleton warmup: the composed loss ramps its skeleton weight on
        # the (0-based) epoch counter.
        if self.criterion is not None:
            self.criterion.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        images, masks, _ = batch
        logits = self(images)
        loss = self._loss(logits, masks)
        if self.criterion is not None:
            for name, val in getattr(self.criterion, "last_parts", {}).items():
                self.log(f"loss_{name}", val, on_step=False, on_epoch=True,
                         batch_size=images.size(0), sync_dist=True)
            self.log("skel_weight",
                     self.criterion.skeleton_weight(self.current_epoch),
                     on_step=False, on_epoch=True, sync_dist=True)
        # Per-patch IoU/F1 (torchmetrics auto-accumulates the epoch + DDP-syncs).
        preds = torch.sigmoid(logits) > self.hparams.threshold
        target = (masks > 0.5).long()
        self.train_iou.update(preds, target)
        self.train_f1.update(preds, target)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
            batch_size=images.size(0), sync_dist=True,
        )
        self.log("train_iou", self.train_iou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # -- per-crop val/test -------------------------------------------------
    def _eval_step(self, batch, iou_metric, f1_metric):
        images, masks, _ = batch
        logits = self(images)
        loss = self._loss(logits, masks)
        preds = torch.sigmoid(logits) > self.hparams.threshold
        target = (masks > 0.5).long()
        iou_metric.update(preds, target)
        f1_metric.update(preds, target)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, self.val_iou, self.val_f1)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=batch[0].size(0), sync_dist=True)
        self.log("val_iou", self.val_iou, prog_bar=True, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, self.test_iou, self.test_f1)
        self.log("test_iou", self.test_iou, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
