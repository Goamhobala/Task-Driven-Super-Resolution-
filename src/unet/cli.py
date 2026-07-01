"""LightningCLI entry point -- one YAML drives train + eval.

The UNet baseline is now config-driven via Lightning subcommands (no more
``train.py`` / ``inference.py``):

    # train (checkpoints on val_iou)
    python -m unet.cli fit  --config src/unet/configs/unet.yaml

    # per-crop IoU/F1 over the test tiles (model.test_step)
    python -m unet.cli test --config src/unet/configs/unet.yaml \
        --ckpt_path checkpoints/unet_s2rosa_best.ckpt

Override any value on the CLI, e.g. ``--data.batch_size 64 --trainer.max_epochs 200``.

``bands``, ``image_size`` and ``normalize`` are declared ONCE under ``data:`` and
linked into the model; ``in_channels`` is derived as ``len(bands)``. Never set those
four under ``model:`` -- jsonargparse rejects setting a linked argument.
"""
from lightning.pytorch.cli import LightningCLI

from sentinel2data.dataset.datasets import RoadDataModule
from unet.model import UNetLightning


class UNetCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Single source of truth in `data:`; the model reads the same values.
        parser.link_arguments("data.bands", "model.bands")
        parser.link_arguments("data.image_size", "model.image_size")
        parser.link_arguments("data.normalize", "model.normalize")
        parser.link_arguments("data.norm_mean", "model.norm_mean")
        parser.link_arguments("data.norm_std", "model.norm_std")
        parser.link_arguments("data.bands", "model.in_channels", compute_fn=len)


def cli_main():
    UNetCLI(
        UNetLightning,
        RoadDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
