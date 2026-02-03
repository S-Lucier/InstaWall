import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "model_imports"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from unet_model import UNet
from dataset import SegmentationDataset, get_train_transforms, get_val_transforms
from utils import save_checkpoint

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
CHECKPOINT_PATH = "saved_models/checkpoint_epoch_800.pth.tar"
SAVE_EVERY_N_EPOCHS = 50  # Save checkpoints every N epochs

# Paths
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"


class ThreeClassSegmentationDataset(SegmentationDataset):
    """3-class segmentation dataset that returns integer class labels.

    Maps pixel values to class indices:
    - 0 (black) -> 0 (walls)
    - 127 (gray) -> 1 (doors)
    - 255 (white) -> 2 (floors)
    """

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Load image and mask
        from PIL import Image
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        # Map pixel values to class indices
        # 0 -> 0 (walls), 127 -> 1 (doors), 255 -> 2 (floors)
        mask_classes = np.zeros_like(mask, dtype=np.int64)
        mask_classes[mask == 0] = 0    # walls
        mask_classes[mask == 127] = 1  # doors
        mask_classes[mask == 255] = 2  # floors

        # Apply transforms (image only gets augmented, mask gets same geometric transforms)
        if self.transform is not None:
            # Convert mask back to uint8 for albumentations compatibility
            mask_for_transform = mask_classes.astype(np.uint8)
            augmented = self.transform(image=image, mask=mask_for_transform)
            image = augmented["image"]
            mask_classes = augmented["mask"].long()  # Convert to long tensor for CrossEntropyLoss

        return image, mask_classes


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    """Train for one epoch with 3-class segmentation."""
    loop = tqdm(loader, desc="Training")
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)  # Shape: (N, H, W) - no unsqueeze needed

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            predictions = model(data)  # Shape: (N, 3, H, W)
            loss = loss_fn(predictions, targets)  # CrossEntropyLoss expects (N, C, H, W) and (N, H, W)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        loop.set_postfix(loss=loss.item())


def check_accuracy_3class(loader, model, device="cuda"):
    """Calculate accuracy and Dice score for 3-class segmentation."""
    num_correct = 0
    num_pixels = 0
    dice_scores = [0.0, 0.0, 0.0]  # For each class
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device)
            y = y.to(device)

            # Get predictions
            preds = model(x)  # Shape: (N, 3, H, W)
            preds = torch.argmax(preds, dim=1)  # Shape: (N, H, W) with class indices

            # Overall pixel accuracy
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # Dice score per class
            for class_idx in range(3):
                pred_class = (preds == class_idx).float()
                target_class = (y == class_idx).float()

                intersection = (pred_class * target_class).sum()
                dice = (2.0 * intersection) / (pred_class.sum() + target_class.sum() + 1e-8)
                dice_scores[class_idx] += dice.item()

    accuracy = (num_correct / num_pixels) * 100
    dice_scores = [d / len(loader) for d in dice_scores]
    avg_dice = sum(dice_scores) / 3

    model.train()
    return accuracy.item(), dice_scores, avg_dice


def main():
    # Create directories for saved models and predictions
    os.makedirs("saved_models_3class", exist_ok=True)
    os.makedirs("saved_predictions_3class", exist_ok=True)

    # Initialize model for 3-class segmentation
    model = UNet(in_channels=3, out_channels=3).to(DEVICE)

    # CrossEntropyLoss for multi-class segmentation
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Load datasets
    train_dataset = ThreeClassSegmentationDataset(
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

    val_dataset = ThreeClassSegmentationDataset(
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
    if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    print(f"Training 3-CLASS segmentation model")
    print(f"Classes: 0=walls, 1=doors, 2=floors")
    print(f"Training on device: {DEVICE}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, DEVICE)

        # Check accuracy
        accuracy, dice_per_class, avg_dice = check_accuracy_3class(val_loader, model, device=DEVICE)
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Dice Score - Walls: {dice_per_class[0]:.4f}, Doors: {dice_per_class[1]:.4f}, Floors: {dice_per_class[2]:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")

        # Save checkpoint every N epochs or on the last epoch
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            save_checkpoint(checkpoint, filename=f"saved_models_3class/checkpoint_epoch_{epoch + 1}.pth.tar")
            print(f"Checkpoint saved at epoch {epoch + 1}")

    print("\nTraining complete!")
    print("Hibernating computer in 10 seconds...")

    # Hibernate the computer after training
    import time
    import subprocess
    time.sleep(10)
    subprocess.run(["shutdown", "/h"])


if __name__ == "__main__":
    main()
