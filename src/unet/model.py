"""UNet model + PyTorch Lightning wrapper for binary road segmentation.

Training is patch-based (random native crops). Validation/test are the
**authoritative metric**: each whole zone is predicted with a native-CRS,
cosine-blended sliding window (:func:`sentinel2data.dataset.predict_zone`) and scored
**once per ground pixel** via torchmetrics (global TP/FP/FN, DDP-synced) -- never
a per-overlapping-tile IoU/F1 average.
"""
from __future__ import annotations

import lightning.pytorch as pl
import rasterio
import segmentation_models_pytorch as smp
import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from sentinel2data.dataset import predict_zone


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
    """UNet + (Dice + weighted BCE); train_loss per epoch, stitched val/test IoU+F1."""

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
        val_overlap: int = 128,
        threshold: float = 0.5,
        normalize: bool = True,
        norm_mean: list[float] | None = None,
        norm_std: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(encoder_name, encoder_weights, in_channels, classes)
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # Stitched, once-per-pixel metrics (accumulate global TP/FP/FN; auto DDP-sync).
        self.val_iou = BinaryJaccardIndex()
        self.val_f1 = BinaryF1Score()
        self.test_iou = BinaryJaccardIndex()
        self.test_f1 = BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, masks):
        # Dice handles overlap; weighted BCE pushes the sparse road class so the
        # model can't minimise loss by predicting all-background.
        dice = self.dice_loss(logits, masks)
        pos_weight = torch.tensor(self.hparams.pos_weight, device=logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, masks, pos_weight=pos_weight
        )
        return dice + bce

    def training_step(self, batch, batch_idx):
        images, masks, _ = batch
        loss = self._loss(self(images), masks)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
            batch_size=images.size(0), sync_dist=True,
        )
        return loss

    # -- stitched val/test -------------------------------------------------
    def _stitched_eval(self, batch, iou_metric, f1_metric):
        image_path, mask_path, _zone = batch
        prob, _ = predict_zone(
            self, image_path, list(self.hparams.bands),
            size=self.hparams.image_size, overlap=self.hparams.val_overlap,
            normalize=self.hparams.normalize,
            mean=self.hparams.norm_mean, std=self.hparams.norm_std,
        )
        pred = torch.from_numpy((prob > self.hparams.threshold)).to(self.device)
        with rasterio.open(mask_path) as m:
            gt = torch.from_numpy((m.read(1) > 0)).to(self.device)
        iou_metric.update(pred, gt)
        f1_metric.update(pred, gt)

    def validation_step(self, batch, batch_idx):
        self._stitched_eval(batch, self.val_iou, self.val_f1)
        self.log("val_iou", self.val_iou, prog_bar=True, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        self._stitched_eval(batch, self.test_iou, self.test_f1)
        self.log("test_iou", self.test_iou, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

    # -- inference: one stitched probability raster per zone ---------------
    def predict_step(self, batch, batch_idx):
        # `unet.writer.ZonePredictionWriter` consumes this to write COG/PNG + metrics.
        # Pair with `--return_predictions false` so rasters aren't also kept in RAM.
        image_path, mask_path, zone = batch
        prob, profile = predict_zone(
            self, image_path, list(self.hparams.bands),
            size=self.hparams.image_size, overlap=self.hparams.val_overlap,
            normalize=self.hparams.normalize,
            mean=self.hparams.norm_mean, std=self.hparams.norm_std,
        )
        return {
            "zone": zone, "image_path": image_path, "mask_path": mask_path,
            "prob": prob, "profile": profile,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
