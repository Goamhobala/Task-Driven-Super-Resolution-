"""UNet model + PyTorch Lightning wrapper for binary road segmentation."""

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch


def build_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    """Plain segmentation-models-pytorch UNet (was UnetPlusPlus)."""
    if encoder_weights in (None, "none", "None", ""):
        encoder_weights = None
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def _binary_metrics(logits, masks, eps=1e-6):
    """IoU and F1 (Dice) for the road class from logits vs. binary masks."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    tp = torch.sum(preds * masks)
    fp = torch.sum(preds * (1 - masks))
    fn = torch.sum((1 - preds) * masks)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return iou, f1


class UNetLightning(pl.LightningModule):
    """UNet + (Dice + weighted BCE), logging loss/IoU/F1 per train/val/test epoch."""

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        lr=1e-3,
        pos_weight=5.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(encoder_name, encoder_weights, in_channels, classes)
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, masks):
        # Dice handles overlap; weighted BCE pushes the sparse road class so the
        # model can't minimise loss by predicting all-background (roads ~13%).
        dice = self.dice_loss(logits, masks)
        pos_weight = torch.tensor(self.hparams.pos_weight, device=logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, masks, pos_weight=pos_weight
        )
        return dice + bce

    def _shared_step(self, batch, stage):
        images, masks, _ = batch
        logits = self(images)
        loss = self._loss(logits, masks)

        with torch.no_grad():
            iou, f1 = _binary_metrics(logits, masks)

        bs = images.size(0)
        log = dict(on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}_loss", loss, prog_bar=True, **log)
        self.log(f"{stage}_iou", iou, prog_bar=True, **log)
        self.log(f"{stage}_f1", f1, **log)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
