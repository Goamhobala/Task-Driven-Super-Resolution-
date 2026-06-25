"""Train the UNet road-segmentation baseline on S2-ROSA-V2 (native loaders).

Native-CRS, no-warp loaders (``unet.patch_dataset``): train = random 256 crops;
val/test = whole-zone stitched IoU/F1 (in ``unet.model``). Checkpoints on the
stitched ``val_iou``.
"""

import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from unet.model import UNetLightning
from unet.patch_dataset import DEFAULT_BANDS, RoadDataModule

# Dataset symlink created on Kaggle (kelvinwei/s2rosa-v2; see prep/kaggle_dependencies.py).
KAGGLE_DATASET_DIR = "/kaggle/working/InstaRoadPrototype/dataset/s2rosa"


def _bands(value):
    return tuple(int(x) for x in value.split(","))


def parse_args():
    p = argparse.ArgumentParser(description="Train UNet on S2-ROSA-V2")
    p.add_argument(
        "dataset_dir", nargs="?", default=KAGGLE_DATASET_DIR,
        help="S2-ROSA-V2 root (contains splits/ + metadata.parquet).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--bands", type=_bands, default=DEFAULT_BANDS,
        help="1-based indices into the 23-band imagery (default 21,22,23 = enhanced "
        "RGB; 1,2,3,4 = raw RGB+NIR).",
    )
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument(
        "--length", type=int, default=None,
        help="Train patches per epoch (default 10 * n_train_tiles).",
    )
    p.add_argument(
        "--val-overlap", type=int, default=128,
        help="Val/test sliding-window overlap (px); 0 = non-overlapping.",
    )
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=5.0, help="BCE weight on road class.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-normalize", action="store_true", help="Disable per-image standardisation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--fast-dev-run", action="store_true", help="Single-batch smoke run.")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    normalize = not args.no_normalize

    datamodule = RoadDataModule(
        dataset_dir=args.dataset_dir,
        bands=args.bands,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        length=args.length,
        normalize=normalize,
    )

    model = UNetLightning(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=len(args.bands),
        classes=1,
        lr=args.lr,
        pos_weight=args.pos_weight,
        bands=args.bands,
        image_size=args.image_size,
        val_overlap=args.val_overlap,
        threshold=args.threshold,
        normalize=normalize,
    )

    logger = WandbLogger(project="unet_s2rosa_baseline") if args.wandb else False
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="unet_s2rosa_best",
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, datamodule=datamodule)

    if not args.fast_dev_run:
        print("Best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()
