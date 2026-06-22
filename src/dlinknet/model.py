import torch
import lightning as L  # pip install lightning  (formerly pytorch-lightning)
import segmentation_models_pytorch as smp
import torchmetrics  # pip install torchmetrics (ships with lightning, but pin if needed)

from dlinknet.networks.dinknet import DinkNet34


def build_model(in_channels=3):
    # designed to receive 1024x1024 images, outputs preds after sigmoid
    return DinkNet34(num_classes=1, num_channels=in_channels)


class LightningWrapper(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3, threshold=0.5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
        self.save_hyperparameters(ignore=["model"])

        metric_kwargs = {"task": "binary", "threshold": threshold}
        self.test_iou = torchmetrics.JaccardIndex(**metric_kwargs)
        self.test_f1 = torchmetrics.F1Score(**metric_kwargs)
        self.test_precision = torchmetrics.Precision(**metric_kwargs)
        self.test_recall = torchmetrics.Recall(**metric_kwargs)
        self.test_accuracy = torchmetrics.Accuracy(**metric_kwargs)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        images, masks, _ = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0))
        return loss

    # --- replaces evaluate_metrics() ---
    def test_step(self, batch, batch_idx):
        images, masks, _ = batch
        preds = self.model(images)          # already sigmoided
        masks_int = masks.int()
        self.test_iou.update(preds, masks_int)
        self.test_f1.update(preds, masks_int)
        self.test_precision.update(preds, masks_int)
        self.test_recall.update(preds, masks_int)
        self.test_accuracy.update(preds, masks_int)

    def on_test_epoch_end(self):
        self.log_dict({
            "test_iou": self.test_iou.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute(),
            "test_accuracy": self.test_accuracy.compute(),
        })

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.model(images)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)