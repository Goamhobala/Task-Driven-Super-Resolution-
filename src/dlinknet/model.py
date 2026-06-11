import torch
import wandb
# Import directly from the cloned GitHub repository files
from networks.dlinknet import dlinknet34 

def build_dlinknet():
    # Initializes the DLinkNet34 model from the GitHub source
    return dlinknet34(num_classes=1)

# --- Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, save_path='best_model.pth'):
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
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

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks, _ in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)

                outputs = model(val_images)
                loss = criterion(outputs, val_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- LOGGING TO W&B ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # --- SAVE BEST MODEL ---
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

    return model