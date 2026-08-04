"""LightningCLI for joint SR + UNet training (:class:`sr.model.JointSRUNetLightning`).

Mirrors :mod:`unet.cli` / :mod:`unet.cli_upscale` (same ``UNetCLI`` argument
links, same subcommands) but drives the joint-SR stack: native 10 m crops
(``JointSRDataModule``) -> SEN2SR (or bicubic) -> UNet, with the SR net and the
UNet on separate learning rates (``model.lr_sr`` / ``model.lr``):

    python -m sr.cli fit  --config src/sr/configs/joint_sr.yaml \
                          --config src/unet/configs/norm_stats.yaml
    python -m sr.cli test --config ... --ckpt_path checkpoints/unet_s2rosa_jointsr_best.ckpt

The plain ``unet.cli`` native path and ``unet.cli_upscale`` bicubic path are
untouched; all three share the split CSVs, metrics and checkpoint convention.
"""
import torch

from sentinel2data.dataset.joint_sr_dataset import JointSRDataModule
from sr.model import JointSRUNetLightning
from unet.cli import UNetCLI

# Keep the refit's numerical/throughput regime identical to the search's
# (sr.tune sets the same two flags; see the comment there).
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class JointSRCLI(UNetCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        # data.upscale defines both the mask grid and the SR factor.
        parser.link_arguments("data.upscale", "model.upscale")


def cli_main():
    JointSRCLI(
        JointSRUNetLightning,
        JointSRDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
