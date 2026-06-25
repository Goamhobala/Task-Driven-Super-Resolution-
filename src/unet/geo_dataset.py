"""torchgeo-backed S2-ROSA-V2 loading for the UNet baseline.

The S2-ROSA-V2 producer (``sentinel2data.generator.make_v2rosa_pipeline``, Kaggle
dataset ``kelvinwei/s2rosa-v2``) writes split-segregated imagery -- raw 512x512
GTiff train tiles + whole-zone val/test COGs -- each with 20 source bands plus 3
appended CLAHE+gamma enhanced-RGB bands (23 total), a raster mask, and a per-image
road-graph parquet. This module wraps the imagery+mask with
``sentinel2data.torchgeo_dataset`` and adapts the torchgeo sample dict to the
``(image, mask, filename)`` tuple ``unet.model.UNetLightning`` consumes:

  * train -> ``RandomGeoSampler``; val/test -> ``GridGeoSampler`` (dense).
  * image is per-channel standardised per patch.
  * mask ``(B, H, W)`` long -> ``(B, 1, H, W)`` float {0, 1}.

``bands`` are 1-based indices into the 23-band imagery; the default is the enhanced
RGB triplet (:data:`ENHANCED_RGB`, bands 21-23). Pass :data:`RAW_RGB` ((1, 2, 3))
for the unprocessed source RGB.
"""
from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

from sentinel2data.torchgeo_dataset import (
    S2_V2_BANDS,
    S2RosaV2Image,
    WGS84,
    build_dataset,
)

# 1-based band indices into the 23-band S2-ROSA-V2 imagery.
RAW_RGB = (1, 2, 3)           # source B4, B3, B2
ENHANCED_RGB = (21, 22, 23)   # appended CLAHE+gamma B4, B3, B2
DEFAULT_BANDS = ENHANCED_RGB  # feed the enhanced RGB to the ImageNet encoder


def _band_names(bands):
    """Map 1-based band indices to the 23-band V2 torchgeo band names."""
    return tuple(S2_V2_BANDS[i - 1] for i in bands)


def _make_transform(normalize):
    """Per-patch image cleanup (+ optional per-channel standardisation).

    Runs on the merged ``{'image','mask',...}`` sample inside the torchgeo
    IntersectionDataset, i.e. once per patch, so standardisation is per-image.
    """

    def _t(sample):
        img = torch.nan_to_num(
            sample["image"].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        if normalize:
            mean = img.mean(dim=(1, 2), keepdim=True)
            std = img.std(dim=(1, 2), keepdim=True) + 1e-6
            img = (img - mean) / std
        sample["image"] = img
        return sample

    return _t


def _collate(samples):
    """torchgeo stack -> the model's ``(image, mask, filename)`` tuple."""
    batch = stack_samples(samples)
    image = batch["image"]                          # (B, C, H, W) float
    mask = (batch["mask"] > 0).unsqueeze(1).float()  # (B, 1, H, W) {0, 1}
    names = [f"patch_{i}.png" for i in range(image.size(0))]
    return image, mask, names


class ROSAGeoDataModule(pl.LightningDataModule):
    """Serves train/val/test loaders of patches sampled from the tiled COGs."""

    def __init__(
        self,
        dataset_dir,
        batch_size=16,
        num_workers=2,
        bands=DEFAULT_BANDS,
        image_size=256,
        length=None,
        normalize=True,
        crs=WGS84,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.band_names = _band_names(bands)
        self.image_size = image_size
        self.length = length            # train patches/epoch (None -> 100*n_tiles)
        self.transform = _make_transform(normalize)
        self.crs = crs

    def _dataset(self, split):
        ds = build_dataset(
            self.dataset_dir, split=split, bands=self.band_names, crs=self.crs,
            image_cls=S2RosaV2Image,
        )
        ds.transforms = self.transform  # applied per merged sample in __getitem__
        return ds

    def _loader(self, split):
        ds = self._dataset(split)
        n_tiles = len(ds.datasets[0].index)
        if split == "train":
            length = self.length or 10 * n_tiles
            sampler = RandomGeoSampler(ds, size=self.image_size, length=length)
        else:
            sampler = GridGeoSampler(
                ds, size=self.image_size, stride=self.image_size
            )
        return DataLoader(
            ds,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
