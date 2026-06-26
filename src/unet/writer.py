"""Lightning prediction writer for the UNet ``predict`` subcommand.

The model's :meth:`UNetLightning.predict_step` stitches one probability raster per
zone in its native CRS (:func:`sentinel2data.dataset.predict_zone`). This callback
turns each into outputs + metrics, mirroring the old ``inference.py``:

  * a 2-band COG (band1 binary mask, band2 probability),
  * a 3-panel comparison PNG (RGB | ground truth | prediction),
  * global once-per-pixel TP/FP/FN/TN -> IoU/F1/precision/recall/accuracy,
    printed at the end (single-device; not DDP-synced).

Wire it under ``trainer.callbacks`` in the YAML config. It only acts on the
``predict`` hooks, so it is inert during ``fit``/``validate``/``test``.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from lightning.pytorch.callbacks import BasePredictionWriter


class ZonePredictionWriter(BasePredictionWriter):
    """Write per-zone COG + comparison PNG and print global stitched metrics."""

    def __init__(
        self,
        output_dir: str = "predictions",
        threshold: float = 0.5,
        write_cog: bool = True,
        write_png: bool = True,
        compute_metrics: bool = True,
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.threshold = threshold
        self.write_cog = write_cog
        self.write_png = write_png
        self.compute_metrics = compute_metrics
        self._reset()

    def _reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def on_predict_start(self, trainer, pl_module):
        self._reset()
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx,
        dataloader_idx=0,
    ):
        prob = prediction["prob"]
        image_path, mask_path = prediction["image_path"], prediction["mask_path"]
        name = Path(image_path).stem

        if self.compute_metrics:
            with rasterio.open(mask_path) as m:
                gt = m.read(1) > 0
            pred = prob > self.threshold
            self.tp += float(np.sum(pred & gt)); self.fp += float(np.sum(pred & ~gt))
            self.fn += float(np.sum(~pred & gt)); self.tn += float(np.sum(~pred & ~gt))

        if self.write_cog:
            cog_out = os.path.join(self.output_dir, f"{name}_pred.tif")
            write_prediction_cog(prob, prediction["profile"], cog_out, threshold=self.threshold)
            print(f"    -> {cog_out}")
        if self.write_png:
            comp_out = os.path.join(self.output_dir, f"{name}_comparison.png")
            save_cog_comparison(image_path, mask_path, prob, comp_out, threshold=self.threshold)
            print(f"    -> {comp_out}")

    def on_predict_epoch_end(self, trainer, pl_module):
        if not self.compute_metrics:
            return
        eps = 1e-6
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        metrics = {
            "IoU": tp / (tp + fp + fn + eps),
            "F1 Score": 2 * tp / (2 * tp + fp + fn + eps),
            "Precision": tp / (tp + fp + eps),
            "Recall": tp / (tp + fn + eps),
            "Accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        }
        print("\nStitched metrics (scored once per ground pixel):")
        for k, v in metrics.items():
            print(f"  - {k}: {round(float(v), 4)}")


# -- writers (moved verbatim from the old inference.py) --------------------
def write_prediction_cog(prob, profile, out_path, threshold=0.5):
    """Write the stitched prediction as a 2-band COG (band1 binary, band2 prob)."""
    cog_profile = profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "interleave", "predictor"):
        cog_profile.pop(key, None)

    prob = prob.astype(np.float32)
    binary = (prob > threshold).astype(np.float32)
    data = np.stack([binary, prob])  # (2, H, W)
    cog_profile.update(driver="COG", dtype="float32", count=2, nodata=None, compress="DEFLATE")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **cog_profile) as dst:
        dst.write(data)
        dst.set_band_description(1, "road_mask_binary")
        dst.set_band_description(2, "road_probability")


def _stretch_rgb(img, percentile_range=(2, 98)):
    """Percentile contrast-stretch a (3, H, W) float array to a (H, W, 3) uint8 image."""
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        band = img[i]
        valid = band[band > 0]
        if valid.size:
            lo, hi = np.percentile(valid, list(percentile_range))
            if hi > lo:
                out[i] = np.clip((np.clip(band, lo, hi) - lo) / (hi - lo), 0, 1)
    return (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8)


def save_cog_comparison(tile_path, mask_path, prob, out_path, threshold=0.5):
    """Whole-zone three-panel figure: satellite RGB | ground-truth | predicted road."""
    with rasterio.open(tile_path) as src:
        rgb = _stretch_rgb(src.read([1, 2, 3]).astype(np.float32))
    with rasterio.open(mask_path) as msrc:
        true_mask = (msrc.read(1) > 0).astype(np.float32)
    pred_mask = (prob > threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(rgb)
    axes[0].set_title("Satellite (RGB)")
    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth Road")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted Road")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
