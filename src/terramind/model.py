import torch
import wandb
from terratorch.models import EncoderDecoderFactory


def build_model(ckpt_path: str | None = None):
    """
    Builds a TerraMind-based binary segmentation model using TerraTorch.

    Backbone : terramind_v1_base  (ViT-Base pre-trained on EO data)
    Modality : RGB  — supported natively; patch embedding was pre-trained on
               Sentinel-2 RGB inputs in [0, 255].
    Neck     : ReshapeTokensToImage — converts final-layer ViT tokens
               [B, T, D] → [B, 768, 16, 16] (patch_size=16, 256px input).
    Decoder  : FCNDecoder → 1-channel logit map (road / no-road), bilinearly
               upsampled to 256×256 by the factory's built-in rescale.

    Args:
        ckpt_path: Path to a locally downloaded TerraMind backbone checkpoint
                   (e.g. TerraMind_v1_base.pt).  When None the backbone is
                   initialised with random weights — intended for inference
                   where a full fine-tuned state dict will be loaded afterwards
                   via model.load_state_dict().
    """
    factory = EncoderDecoderFactory()

    backbone_extra = {}
    if ckpt_path is not None:
        backbone_extra["backbone_ckpt_path"] = ckpt_path

    model = factory.build_model(
        task="segmentation",
        num_classes=1,                      # top-level: binary road vs. background
        # --- Backbone ---
        backbone="terramind_v1_base",
        backbone_pretrained=False,          # weights come from backbone_ckpt_path
        backbone_modalities=["RGB"],        # 3-channel RGB input
        **backbone_extra,
        # --- Neck ---
        # ViT outputs token sequences [B, T, D]. ReshapeTokensToImage converts
        # the final-layer tokens to [B, D, H', W'] (16×16 for 256px input with
        # patch_size=16). The factory's rescale=True then bilinearly upsamples
        # the decoder output back to 256×256.
        # UperNetDecoder is avoided here: its Pyramid Pooling Module requires
        # spatial dims ≥ its pool_scales, which breaks on the 8×8 / 4×4 / 2×2
        # maps that LearnedInterpolateToPyramidal would produce from a 16×16 grid.
        necks=[
            {"name": "ReshapeTokensToImage"},   # removes CLS token, gives [B, 768, 16, 16]
        ],
        # --- Decoder ---
        decoder="FCNDecoder",
    )

    return model


def _extract_logits(output):
    """
    Normalise model output to a plain tensor regardless of whether
    TerraTorch returned a ModelOutput dataclass, a dict, a list/tuple,
    or a raw tensor.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        return output.get("output", next(iter(output.values())))
    if isinstance(output, (list, tuple)):
        return output[0]
    if hasattr(output, "output"):
        return output.output
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"Unexpected model output type: {type(output)}")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int = 5,
    save_path: str = "best_terramind.pth",
):
    print("Starting TerraMind fine-tuning...")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            # TerraMind expects a modality dict; "RGB" maps to 3-channel input
            raw = model({"RGB": images})
            outputs = _extract_logits(raw)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks, _ in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)

                raw = model({"RGB": val_images})
                outputs = _extract_logits(raw)
                loss = criterion(outputs, val_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # --- LOGGING TO W&B ---
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # --- SAVE BEST MODEL ---
        if avg_val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.4f} "
                f"to {avg_val_loss:.4f}. Saving model..."
            )
            best_val_loss = avg_val_loss
            # DataParallel wraps the real model under .module
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)

    print("Fine-tuning complete.")
    return model
