import torch
import wandb
from terratorch.models import EncoderDecoderFactory


def build_model(pretrained: bool = True):
    """
    Builds a TerraMind-based binary segmentation model using TerraTorch.

    Backbone: terramind_v1_base (ViT-Base pre-trained on EO data)
    Decoder:  FCNDecoder -> 1-channel logit map (road / no-road)

    The dataset images are RGB PNGs (3 channels, 256x256).  TerraMind
    was pre-trained on multi-spectral data but can be fine-tuned on RGB
    by setting in_chans=3, which replaces the patch-embedding projection.
    """
    factory = EncoderDecoderFactory()

    model = factory.build_model(
        task="segmentation",
        backbone="terramind_v1_base",
        backbone_kwargs={
            "pretrained": pretrained,
            "in_chans": 3,         # RGB input (Sentinel-2 enhanced PNGs)
        },
        decoder="FCNDecoder",
        decoder_kwargs={
            "num_classes": 1,      # Binary: road vs. background
            "channels": 256,
        },
        head_kwargs={
            "dropout": 0.1,
        },
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
    # terratorch ModelOutput / dataclass
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
            raw = model(images)
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

                raw = model(val_images)
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
            torch.save(model.state_dict(), save_path)

    print("Fine-tuning complete.")
    return model
