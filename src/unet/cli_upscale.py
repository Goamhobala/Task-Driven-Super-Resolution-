"""LightningCLI for the super-resolution dataloader (:class:`UpscaleRoadDataModule`).

Mirrors :mod:`unet.cli` (reuses its ``UNetCLI`` argument links + ``UNetLightning``)
but feeds bicubic-upsampled crops + graph-rasterised masks over a NATIVE V2 dataset:

    python -m unet.cli_upscale fit  --config src/unet/configs/unet_upscale.yaml
    python -m unet.cli_upscale test --config src/unet/configs/unet_upscale.yaml \
        --ckpt_path checkpoints/unet_s2rosa_upscale_best.ckpt

The plain ``unet.cli`` + ``RoadDataModule`` native path is untouched.
"""
from sentinel2data.dataset.upscale_dataset import UpscaleRoadDataModule
from unet.cli import UNetCLI
from unet.model import UNetLightning


def cli_main():
    UNetCLI(
        UNetLightning,
        UpscaleRoadDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
