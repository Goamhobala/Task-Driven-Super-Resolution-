"""Register the baseline architecture with TerraTorch.

Importing this module registers a `ModelFactory` named ``"SmpBaselineFactory"``
in TerraTorch's ``MODEL_FACTORY_REGISTRY``. Once registered, the baseline trains
through TerraTorch's own Lightning task (config- or code-driven), e.g.::

    import baseline.terratorch_register          # noqa: F401  (registers the factory)
    from terratorch.tasks import SemanticSegmentationTask

    task = SemanticSegmentationTask(
        model_factory="SmpBaselineFactory",
        model_args={"config": "M3", "encoder": "resnet50", "num_classes": 2},
        loss="dice",          # TerraTorch's loss registry; or "ce", "focal", ...
        lr=1e-3,
    )
    # task is a LightningModule -> trainer.fit(task, datamodule=...)

Or from a TerraTorch YAML config:

    model:
      class_path: terratorch.tasks.SemanticSegmentationTask
      init_args:
        model_factory: SmpBaselineFactory
        model_args: {config: M3, encoder: resnet50, num_classes: 2}
        loss: dice

Notes:
  * TerraTorch's task owns the loss/optimiser/metrics, so the custom RoadSegLoss
    (BCE+Dice+clDice) is NOT used on this path — use `baseline.task.BaselineSegTask`
    if you want clDice. The factory here only contributes the architecture.
  * `num_classes` follows TerraTorch's convention (2 = background/road, softmax).
    `config` (M0-M3) or an explicit `in_channels` sets the input band count.

Requires the `terra` extra (terratorch). The TerraTorch imports are done lazily
so the rest of `baseline` stays importable without it.
"""
from __future__ import annotations

import torch.nn as nn

from baseline.data import CHANNEL_GROUPS
from baseline.model import build_model

try:
    from terratorch.models.model import Model, ModelFactory, ModelOutput
    from terratorch.registry import MODEL_FACTORY_REGISTRY
except ImportError as e:  # pragma: no cover - only hit without the terra extra
    raise ImportError(
        "baseline.terratorch_register needs TerraTorch. Install the extra: "
        'uv pip install -e ".[terra]"'
    ) from e


class SmpBaselineModel(Model, nn.Module):
    """smp UNet++ adapted to TerraTorch's Model contract (forward -> ModelOutput).

    Holds the smp model so `freeze_encoder` / `freeze_decoder` can toggle the
    encoder and decoder+head independently, as the task may request.
    """

    def __init__(self, smp_model: nn.Module):
        super().__init__()
        self.smp_model = smp_model

    def forward(self, x, **kwargs) -> ModelOutput:
        return ModelOutput(output=self.smp_model(x))

    def freeze_encoder(self):
        for p in self.smp_model.encoder.parameters():
            p.requires_grad = False

    def freeze_decoder(self):
        for p in self.smp_model.decoder.parameters():
            p.requires_grad = False
        for p in self.smp_model.segmentation_head.parameters():
            p.requires_grad = False


@MODEL_FACTORY_REGISTRY.register
class SmpBaselineFactory(ModelFactory):
    """Builds the baseline UNet++ for TerraTorch's SemanticSegmentationTask."""

    def build_model(
        self,
        task: str = "segmentation",
        config: str | None = "M3",
        in_channels: int | None = None,
        num_classes: int = 2,
        encoder: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        **kwargs,
    ) -> Model:
        if in_channels is None:
            if config not in CHANNEL_GROUPS:
                raise ValueError(f"config must be one of {list(CHANNEL_GROUPS)} or pass in_channels")
            in_channels = len(CHANNEL_GROUPS[config])

        smp_model = build_model(
            in_channels=in_channels, encoder=encoder,
            encoder_weights=encoder_weights, classes=num_classes,
        )
        return SmpBaselineModel(smp_model)
