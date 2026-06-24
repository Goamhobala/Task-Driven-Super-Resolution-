import os 
import torch
import lightning as L  # pip install lightning  (formerly pytorch-lightning)
import segmentation_models_pytorch as smp
import torchmetrics  # pip install torchmetrics (ships with lightning, but pin if needed)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class PredictionSaverCallback(L.Callback):
    """Writes comparison plots (or raw masks) per batch instead of buffering them."""
    def __init__(self, output_dir, save_comparison=True, threshold=0.5):
        super().__init__()
        self.output_dir = output_dir
        self.save_comparison = save_comparison
        self.threshold = threshold
        os.makedirs(output_dir, exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images, masks, filenames = batch
        preds = (outputs > self.threshold).float().cpu().numpy()
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(len(filenames)):
            pred_mask = preds[i].squeeze()

            if self.save_comparison:
                img = images_np[i].transpose(1, 2, 0)
                true_mask = masks_np[i].squeeze()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img); axes[0].set_title("Original Image"); axes[0].axis("off")
                axes[1].imshow(true_mask, cmap="gray"); axes[1].set_title("True Road Label"); axes[1].axis("off")
                axes[2].imshow(pred_mask, cmap="gray"); axes[2].set_title("Predicted Road"); axes[2].axis("off")
                plt.tight_layout()
                fig.savefig(os.path.join(self.output_dir, f"comp_{filenames[i]}"), bbox_inches="tight")
                plt.close(fig)
            else:
                pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_mask_uint8).save(os.path.join(self.output_dir, filenames[i]))

class LightningWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate=1e-3, threshold=0.5):
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
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch[0].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch[0].size(0))
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