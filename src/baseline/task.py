"""Lightning module for the baseline.

`BaselineSegTask` wraps the same smp UNet++ (`baseline.model.build_model`) and the
same `RoadSegLoss` used by the plain `train.py`, but as a `lightning.LightningModule`
so it trains under a `lightning.Trainer` (checkpointing, DDP, loggers, etc.).

Batches are the `(image, mask)` 2-tuples produced by `RoadSegDataset`; the
benchmarking `(x, y, chip_id, tile_id)` tuples also work — only the first two
elements are read.

This is the framework-agnostic Lightning path. For TerraTorch's own task
(config-driven, its loss/metric registry), register the architecture via
`baseline.terratorch_register` and use `SemanticSegmentationTask` instead.
"""
from __future__ import annotations

import lightning as L
import torch
import torchmetrics

from baseline.model import RoadSegLoss, build_model


class BaselineSegTask(L.LightningModule):
    def __init__(self, in_channels: int = 3, encoder: str = "resnet50",
                 encoder_weights: str | None = "imagenet", lr: float = 1e-3,
                 pos_weight: float = 1.0, alpha: float = 0.3, threshold: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(in_channels=in_channels, encoder=encoder,
                                 encoder_weights=encoder_weights)
        self.criterion = RoadSegLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32), alpha=alpha
        )
        self.threshold = threshold

        m = {"task": "binary", "threshold": threshold}
        self.val_iou = torchmetrics.JaccardIndex(**m)
        self.val_f1 = torchmetrics.F1Score(**m)
        self.test_iou = torchmetrics.JaccardIndex(**m)
        self.test_f1 = torchmetrics.F1Score(**m)
        self.test_precision = torchmetrics.Precision(**m)
        self.test_recall = torchmetrics.Recall(**m)

    def forward(self, x):
        return self.model(x)  # raw logits

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        yi = y.int()
        self.val_iou.update(probs, yi)
        self.val_f1.update(probs, yi)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=x.size(0))
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
        x = batch[0]
        return torch.sigmoid(self(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
