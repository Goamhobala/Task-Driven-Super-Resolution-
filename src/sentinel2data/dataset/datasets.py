"""Custom Dataset Loader"""
import random
from pathlib import Path
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
from sentinel2data.dataset.bands import DEFAULT_BANDS
from sentinel2data.dataset.reading import apply_norm, read_window


def _read_split_csv(dataset_dir, split):
    csv = Path(dataset_dir) / "splits" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv}")
    return pd.read_csv(csv)


def _remap_mask_paths(df, dataset_dir, mask_dirname):
    """Point ``mask_path`` at an alternative label set living beside the
    pipeline's ``masks_raster/`` (e.g. ``mask_osm_10`` written by
    OpenStreetMapTest/dataset_hr_masks.py --scale 1). ``None`` keeps the CSV's
    masks unchanged. The alternative masks are rasterised on each tile's own
    grid, so dims/CRS match; every remapped file must exist — missing labels
    are an error, not a silent filter, so label-source comparisons stay on
    identical tile sets."""
    if mask_dirname is None:
        return df
    df = df.copy()
    df["mask_path"] = df["mask_path"].map(
        lambda rel: str(Path(rel).parent.parent / mask_dirname / Path(rel).name))
    missing = [rel for rel in df["mask_path"]
               if not (Path(dataset_dir) / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(df)} tiles have no mask under "
            f"<split>/{mask_dirname}/ (first: {missing[0]}). Generate them "
            f"with OpenStreetMapTest/dataset_hr_masks.py --scale 1 "
            f"--out-dirname {mask_dirname}."
        )
    return df


class RoadTileDataset(Dataset):
    """Use this dataset for training. Random pixel crops from the 512x512 train tiles.

    Length by default is 10 * len(train_tiles)

    ``transform`` is an optional Albumentations pipeline (see
    ``sentinel2data.dataset.augment.build_transform``) applied per patch AFTER
    normalisation — the photometric magnitudes are tuned for z-scored input.
    Image and mask go through the same call, so geometry stays aligned.
    """

    def __init__(self, dataset_dir, bands=DEFAULT_BANDS, image_size=256,
                 length=None, normalize=True, norm_mean=None, norm_std=None,
                 mask_dirname=None, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.df = _remap_mask_paths(
            _read_split_csv(dataset_dir, "train"), dataset_dir, mask_dirname
        ).reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = image_size
        self.normalize = normalize
        self.norm_mean = norm_mean  # frozen train stats (or None -> per-image)
        self.norm_std = norm_std
        self.length = length if length is not None else 10 * len(self.df)
        self.transform = transform

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Random crop and normalise"""
        row = self.df.iloc[random.randrange(len(self.df))]
        s = self.image_size
        with rasterio.open(self.dataset_dir / row["image_path"]) as src, \
                rasterio.open(self.dataset_dir / row["mask_path"]) as msrc:
            H, W = src.height, src.width
            top = random.randint(0, max(0, H - s))
            left = random.randint(0, max(0, W - s))
            win = Window(left, top, min(s, W - left), min(s, H - top))
            img = read_window(src, self.bands, win)                 # (C, h, w)
            mask = (msrc.read(1, window=win) > 0).astype("float32")  # (h, w)

        c, h, w = img.shape
        if (h, w) != (s, s):  # short edge tile -> zero-pad (train tiles are 512, rare)
            pad_i = np.zeros((c, s, s), dtype="float32"); pad_i[:, :h, :w] = img
            pad_m = np.zeros((s, s), dtype="float32"); pad_m[:h, :w] = mask
            img, mask = pad_i, pad_m

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        if self.transform is not None:
            aug = self.transform(image=img.transpose(1, 2, 0), mask=mask)
            img = aug["image"].transpose(2, 0, 1)
            mask = aug["mask"]
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_{top}_{left}.png"


class TileCropDataset(Dataset):
    """
    Use this dataset for validation/testing. 

    Deterministic 2x2 crops from the 512x512 tiles - sliding window with no overlaps
    Item ``idx`` -> tile ``idx // 4``, quadrant ``idx % 4`` (row-major)
    """

    def __init__(self, dataset_dir, split, bands=DEFAULT_BANDS,
                 normalize=True, norm_mean=None, norm_std=None,
                 mask_dirname=None):
        self.dataset_dir = Path(dataset_dir)
        self.df = _remap_mask_paths(
            _read_split_csv(dataset_dir, split), dataset_dir, mask_dirname
        ).reset_index(drop=True)
        self.bands = list(bands)
        self.image_size = 256
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.per_tile = 4  # Assume 512x512 tiles, hence 4 patches.

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def __len__(self):
        return len(self.df) * self.per_tile

    def __getitem__(self, idx):
        """Sliding Window and normalise"""
        row = self.df.iloc[idx // self.per_tile]
        quad = idx % self.per_tile
        s = self.image_size
        top = (quad // 2) * s
        left = (quad % 2) * s
        with rasterio.open(self.dataset_dir / row["image_path"]) as src, \
                rasterio.open(self.dataset_dir / row["mask_path"]) as msrc:
            win = Window(left, top, s, s)
            img = read_window(src, self.bands, win)                  # (C, s, s)
            mask = (msrc.read(1, window=win) > 0).astype("float32")  # (s, s)

        if self.normalize:
            img = apply_norm(img, self.bands, self.norm_mean, self.norm_std)
        image = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, mask, f"{row['zone_name']}_q{quad}.png"


class RoadDataModule(pl.LightningDataModule):
    """Train - random native crops; val/test - deterministic 2x2 quadrant crops.

    ``mask_dirname`` switches the label source: ``None`` (default) uses the
    pipeline masks the split CSVs point at (``masks_raster/``, e.g. CDNGI);
    a dir name (e.g. ``mask_osm_10``) uses the alternative masks stored
    beside them, applied to train/val/test alike. To cross-evaluate (train on
    one label set, test on the other), pass a different ``--data.mask_dirname``
    to ``unet.cli test``.

    ``aug_*`` toggles build the Albumentations pipeline of
    ``sentinel2data.dataset.augment`` for the TRAIN split only (val/test are
    never augmented). All off by default; ``aug_flip`` (lossless D4
    flips/rotations) is the safe one to start with. The photometric toggles
    each fire with probability ``aug_p``.
    """

    def __init__(self, dataset_dir: str, bands: tuple[int, ...] = DEFAULT_BANDS,
                 batch_size: int = 16, num_workers: int = 2, image_size: int = 256,
                 length: int | None = None, normalize: bool = True,
                 norm_mean: list[float] | None = None, norm_std: list[float] | None = None,
                 mask_dirname: str | None = None,
                 aug_flip: bool = False, aug_sharpen: bool = False,
                 aug_noise: bool = False, aug_blur: bool = False,
                 aug_colour: bool = False, aug_p: float = 0.5):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.bands = tuple(bands)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.length = length
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mask_dirname = mask_dirname
        self.aug_flags = dict(flip=aug_flip, sharpen=aug_sharpen, noise=aug_noise,
                              blur=aug_blur, colour=aug_colour)
        self.aug_p = aug_p

        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No frozen train stats given")

    def _train_transform(self):
        if not any(self.aug_flags.values()):
            return None
        from sentinel2data.dataset.augment import build_transform

        return build_transform(**self.aug_flags, p=self.aug_p)

    def train_dataloader(self):
        ds = RoadTileDataset(
            self.dataset_dir, self.bands, self.image_size, self.length, self.normalize,
            self.norm_mean, self.norm_std, self.mask_dirname,
            transform=self._train_transform(),
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,  # randomness is in __getitem__; DDP adds DistributedSampler
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def _eval_loader(self, split):
        ds = TileCropDataset(
            self.dataset_dir, split, self.bands, self.normalize,
            self.norm_mean, self.norm_std, self.mask_dirname,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self):
        return self._eval_loader("val")

    def test_dataloader(self):
        return self._eval_loader("test")
