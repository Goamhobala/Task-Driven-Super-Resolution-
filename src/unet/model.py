import torch
import segmentation_models_pytorch as smp

def build_model(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1):
    """
    Builds and returns the UnetPlusPlus model for road segmentation.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    return model

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    """
    Executes the training loop for the given model.
    """
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    print("Fine-tuning complete.")

    return model