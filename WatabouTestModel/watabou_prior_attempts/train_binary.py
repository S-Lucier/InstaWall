import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Model Imports"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from unet_model import UNet
from dataset import SegmentationDataset, get_train_transforms, get_val_transforms
from utils import save_checkpoint, check_accuracy, save_predictions


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Reduced for 512×512 to fit in 8GB VRAM
NUM_EPOCHS = 1000
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
CHECKPOINT_PATH = "saved_models/checkpoint_epoch_20.pth.tar"
SAVE_EVERY_N_EPOCHS = 50  # Save checkpoints every N epochs

# Paths
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    """Train for one epoch."""
    loop = tqdm(loader, desc="Training")
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.unsqueeze(1).to(device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        loop.set_postfix(loss=loss.item())


def main():
    # Create directories for saved models and predictions
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_predictions", exist_ok=True)

    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Load datasets
    train_dataset = SegmentationDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=get_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    val_dataset = SegmentationDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=get_val_transforms(IMAGE_HEIGHT, IMAGE_WIDTH),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # Load checkpoint if specified
    start_epoch = 0
    if LOAD_MODEL:
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    print(f"Training on device: {DEVICE}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} (Training epoch {epoch - start_epoch + 1}/{NUM_EPOCHS - start_epoch})")

        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, DEVICE)

        # Check accuracy
        accuracy, dice = check_accuracy(val_loader, model, device=DEVICE)
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Validation Dice Score: {dice:.4f}")

        # Save checkpoint every N epochs or on the last epoch
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            save_checkpoint(checkpoint, filename=f"saved_models/checkpoint_epoch_{epoch + 1}.pth.tar")
            print(f"Checkpoint saved at epoch {epoch + 1}")

            # Save predictions for visualization
            save_predictions(
                val_loader, model, folder="saved_predictions", device=DEVICE
            )

    print("\nTraining complete!")
    print("Hibernating computer in 10 seconds...")

    # Hibernate the computer after training
    import time
    import subprocess
    time.sleep(10)
    subprocess.run(["shutdown", "/h"])


if __name__ == "__main__":
    main()
