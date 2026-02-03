import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import numpy as np
import subprocess
import time
import logging
from datetime import datetime

from unet_attention_aspp import AttentionASPPUNet
from dataset import SegmentationDataset, get_train_transforms, get_val_transforms
from utils import save_checkpoint, load_checkpoint

# Setup logging to both file and console
LOG_FILE = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # Reduced from 4 due to higher memory usage of ASPP
NUM_EPOCHS = 2000
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
CHECKPOINT_PATH = "WatabouTestModel/saved_models_3class/checkpoint_epoch_90.pth.tar"
SAVE_EVERY_N_EPOCHS = 50  # Save checkpoints every N epochs

# Early stopping parameters
EARLY_STOP_WINDOW = 300  # epochs to check for improvement
MIN_IMPROVEMENT = 0.002  # minimum dice improvement required in window

# Paths
TRAIN_IMG_DIR = "../data/train_images"
TRAIN_MASK_DIR = "../data/train_masks"
VAL_IMG_DIR = "../data/val_images"
VAL_MASK_DIR = "../data/val_masks"


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


def check_early_stopping(dice_history, window=300, min_improvement=0.002):
    """Check if training should stop due to lack of improvement.

    Returns True if training should stop (no improvement >= min_improvement in last `window` epochs).
    """
    if len(dice_history) < window:
        return False

    # Get the best dice score from the start of the window to check against
    window_start_best = max(dice_history[:-window]) if len(dice_history) > window else dice_history[0]
    current_best = max(dice_history[-window:])

    improvement = current_best - window_start_best

    if improvement < min_improvement:
        logger.info("Early stopping triggered!")
        logger.info(f"Best dice before window: {window_start_best:.4f}")
        logger.info(f"Best dice in last {window} epochs: {current_best:.4f}")
        logger.info(f"Improvement: {improvement:.4f} (required: {min_improvement})")
        return True

    return False


def hibernate_after_delay(delay_minutes=2):
    """Hibernate the computer after a delay."""
    logger.info(f"Training complete. Hibernating in {delay_minutes} minutes...")
    time.sleep(delay_minutes * 60)
    logger.info("Initiating hibernate...")
    # Windows hibernate command
    subprocess.run(["shutdown", "/h"], shell=True)


def main():
    # Create directories for saved models and predictions
    os.makedirs("../saved_models_3class", exist_ok=True)
    os.makedirs("../saved_predictions_3class", exist_ok=True)

    logger.info(f"Training log file: {LOG_FILE}")
    logger.info("=" * 60)

    # Initialize ENHANCED model with Attention + Multi-Scale features
    logger.info("Initializing AttentionASPPUNet (Attention Gates + Multi-Scale ASPP)...")
    model = AttentionASPPUNet(in_channels=3, out_channels=3).to(DEVICE)

    # CrossEntropyLoss with class weights (inverse frequency)
    # Distribution: Walls 74%, Doors 1%, Floors 25%
    # Weights calculated from 20-sample analysis
    class_weights = torch.tensor([0.45, 32.0, 1.36]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

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
        drop_last=True,  # Skip incomplete last batch to avoid BatchNorm errors
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
    dice_history = []  # Track average dice scores for early stopping

    if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 89) + 1  # Default to 90 if not found

        # Initialize dice history with checkpoint's avg_dice if available
        if "avg_dice" in checkpoint:
            dice_history.append(checkpoint["avg_dice"])
            logger.info(f"Loaded avg_dice from checkpoint: {checkpoint['avg_dice']:.4f}")

        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        logger.warning(f"Checkpoint not found at {CHECKPOINT_PATH}")
        logger.info("Starting from scratch...")

    logger.info(f"TRAINING ENHANCED 3-CLASS MODEL (Extended Run)")
    logger.info(f"=" * 60)
    logger.info(f"Architecture: Attention U-Net + Multi-Scale ASPP")
    logger.info(f"Classes: 0=walls, 1=doors, 2=floors")
    logger.info(f"Class weights: Walls={class_weights[0]:.2f}, Doors={class_weights[1]:.2f}, Floors={class_weights[2]:.2f}")
    logger.info(f"Training on device: {DEVICE}")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Starting epoch: {start_epoch}")
    logger.info(f"Target epochs: {NUM_EPOCHS}")
    logger.info(f"Early stopping: No improvement >= {MIN_IMPROVEMENT} in {EARLY_STOP_WINDOW} epochs")
    logger.info(f"=" * 60)

    # Training loop
    best_avg_dice = 0.0
    best_epoch = 0
    early_stopped = False

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

            train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, DEVICE)

            # Check accuracy
            accuracy, dice_per_class, avg_dice = check_accuracy_3class(val_loader, model, device=DEVICE)
            logger.info(f"Validation Accuracy: {accuracy:.2f}%")
            logger.info(f"Dice Score - Walls: {dice_per_class[0]:.4f}, Doors: {dice_per_class[1]:.4f}, Floors: {dice_per_class[2]:.4f}")
            logger.info(f"Average Dice Score: {avg_dice:.4f}")

            # Track dice history for early stopping
            dice_history.append(avg_dice)

            # Track best average dice
            if avg_dice > best_avg_dice:
                best_avg_dice = avg_dice
                best_epoch = epoch + 1
                logger.info(f"*** NEW BEST Average Dice: {best_avg_dice:.4f} at epoch {best_epoch} ***")

            # Save checkpoint every N epochs or on the last epoch
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "dice_scores": dice_per_class,
                    "avg_dice": avg_dice,
                    "best_avg_dice": best_avg_dice,
                }
                save_checkpoint(checkpoint, filename=f"WatabouTestModel/saved_models_3class/checkpoint_epoch_{epoch + 1}.pth.tar")
                logger.info(f"=> Checkpoint saved at epoch {epoch + 1}")

            # Check for early stopping
            if check_early_stopping(dice_history, window=EARLY_STOP_WINDOW, min_improvement=MIN_IMPROVEMENT):
                early_stopped = True
                # Save final checkpoint before stopping
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "dice_scores": dice_per_class,
                    "avg_dice": avg_dice,
                    "best_avg_dice": best_avg_dice,
                    "early_stopped": True,
                }
                save_checkpoint(checkpoint, filename=f"WatabouTestModel/saved_models_3class/checkpoint_epoch_{epoch + 1}_early_stop.pth.tar")
                logger.info(f"=> Early stop checkpoint saved at epoch {epoch + 1}")
                break

        logger.info("=" * 60)
        logger.info("Training complete!")
        if early_stopped:
            logger.info("Training was stopped early due to lack of improvement.")
        logger.info(f"Best Average Dice: {best_avg_dice:.4f} at epoch {best_epoch}")
        logger.info(f"Final checkpoint saved to saved_models_3class/")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user!")
        logger.info(f"Best Average Dice so far: {best_avg_dice:.4f} at epoch {best_epoch}")
        # Save checkpoint on interrupt
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch if 'epoch' in dir() else start_epoch,
            "interrupted": True,
        }
        save_checkpoint(checkpoint, filename="WatabouTestModel/saved_models_3class/checkpoint_interrupted.pth.tar")
        logger.info("=> Interrupted checkpoint saved")
        return  # Don't hibernate on manual interrupt

    # Hibernate after training (only on successful completion or early stop)
    hibernate_after_delay(delay_minutes=2)


if __name__ == "__main__":
    main()
