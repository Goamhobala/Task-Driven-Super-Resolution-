"""Train the UNet road-segmentation baseline on the S2-ROSA dataset.
"""

import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from unet.dataset import DEFAULT_BANDS, ROSADataModule
from unet.model import UNetLightning

# Default location of the dataset symlink created on Kaggle.
KAGGLE_DATASET_DIR = "/kaggle/working/InstaRoadPrototype/dataset/s2rosa"


def _bands(value):
    return tuple(int(x) for x in value.split(","))


def parse_args():
    p = argparse.ArgumentParser(description="Train UNet on S2-ROSA")
    p.add_argument(
        "dataset_dir",
        nargs="?",
        default=KAGGLE_DATASET_DIR,
        help="Path to the S2-ROSA dataset root (contains metadata.parquet).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bands", type=_bands, default=DEFAULT_BANDS, help="e.g. 1,2,3")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=5.0, help="BCE weight on road class.")
    p.add_argument("--no-normalize", action="store_true", help="Disable per-image standardization.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument(
        "--keep-edge-blocks",
        action="store_true",
        help="Keep partial edge block windows (default: drop them).",
    )
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--fast-dev-run", action="store_true", help="Single-batch smoke run.")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    datamodule = ROSADataModule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bands=args.bands,
        image_size=args.image_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        drop_edge_blocks=not args.keep_edge_blocks,
        normalize=not args.no_normalize,
    )

    model = UNetLightning(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=len(args.bands),
        classes=1,
        lr=args.lr,
        pos_weight=args.pos_weight,
    )

    logger = WandbLogger(project="unet_s2rosa_baseline") if args.wandb else False
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="unet_s2rosa_best",
        monitor="val_loss",
        mode="min",
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
