"""
Training script for wall segmentation model.

Usage:
    python -m primary_model_training.train
    python -m primary_model_training.train --epochs 50 --batch-size 4
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .config import Config
from .model import WallSegmentationUNet, create_model
from .dataset import WallSegmentationDataset, ValidationDataset, create_dataloader


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: WallSegmentationDataset, num_classes: int) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency in the dataset.

    Samples a subset of tiles to estimate class distribution.
    """
    print("Computing class weights from dataset sample...")
    counts = np.zeros(num_classes)

    # Sample up to 100 tiles
    num_samples = min(100, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        sample = dataset[idx]
        mask = sample['mask'].numpy()
        for c in range(num_classes):
            counts[c] += np.sum(mask == c)

    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (num_classes * counts + 1e-6)

    # Normalize
    weights = weights / weights.sum() * num_classes

    print(f"  Class distribution: {counts / total * 100}")
    print(f"  Class weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    Compute segmentation metrics.

    Args:
        pred: Predicted class indices (B, H, W)
        target: Ground truth class indices (B, H, W)
        num_classes: Number of classes

    Returns:
        Dict with metrics
    """
    pred = pred.view(-1)
    target = target.view(-1)

    # Overall accuracy
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total

    # Per-class IoU
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        if union > 0:
            ious.append(intersection / union)

    mean_iou = np.mean(ious) if ious else 0.0

    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
    }


class Trainer:
    """Training manager."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Set seed
        set_seed(config.seed)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(config), f, indent=2)

        # Create model
        self.model = create_model(config).to(self.device)
        print(f"Model parameters: {self.model.get_num_parameters():,}")

        # Create datasets
        self.train_dataset = WallSegmentationDataset(
            image_dir=config.image_dir,
            mask_dir=config.mask_dir,
            tile_grid_cells=config.tile_grid_cells,
            tile_size=config.tile_size,
            tiles_per_image=4,
            mask_scale=config.mask_scale,
            augment=config.augment,
            merge_terrain=config.merge_terrain,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Compute or use provided class weights
        if config.class_weights:
            class_weights = torch.tensor(config.class_weights, dtype=torch.float32)
        else:
            class_weights = compute_class_weights(self.train_dataset, config.num_classes)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        if config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=config.learning_rate / 100,
            )
        elif config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            self.scheduler = None

        # Mixed precision training
        self.scaler = GradScaler() if self.device.type == 'cuda' else None

        # Tracking
        self.best_loss = float('inf')
        self.best_iou = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'lr': [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        all_ious = []

        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()

            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                metrics = compute_metrics(preds, masks, self.config.num_classes)
                total_correct += metrics['accuracy'] * masks.numel()
                total_pixels += masks.numel()
                all_ious.append(metrics['mean_iou'])

            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = total_correct / total_pixels
                avg_iou = np.mean(all_ious)
                lr = self.optimizer.param_groups[0]['lr']

                print(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                      f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} IoU: {avg_iou:.4f} LR: {lr:.6f}")

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_pixels,
            'mean_iou': np.mean(all_ious),
        }

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        all_ious = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.train_loader:  # TODO: Use separate validation set
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                metrics = compute_metrics(preds, masks, self.config.num_classes)
                all_ious.append(metrics['mean_iou'])

                num_batches += 1

                # Limit validation batches
                if num_batches >= 50:
                    break

        return {
            'loss': total_loss / max(num_batches, 1),
            'mean_iou': np.mean(all_ious) if all_ious else 0.0,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_iou': self.best_iou,
            'config': vars(self.config),
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')

        # Save periodic
        if (epoch + 1) % self.config.save_interval == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch{epoch+1}.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            train_metrics = self.train_epoch(epoch + 1)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Validate
            val_metrics = self.validate()

            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_iou'].append(train_metrics['mean_iou'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_iou'].append(val_metrics['mean_iou'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Check for best model
            is_best = val_metrics['mean_iou'] > self.best_iou
            if is_best:
                self.best_iou = val_metrics['mean_iou']
                self.best_loss = val_metrics['loss']

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['mean_iou']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Time: {epoch_time:.1f}s" + (" [BEST]" if is_best else ""))

            # Save history
            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training complete in {total_time / 60:.1f} minutes")
        print(f"Best IoU: {self.best_iou:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train wall segmentation model')

    # Data paths
    parser.add_argument('--image-dir', default='data/foundry_to_mask',
                        help='Directory with battlemap images')
    parser.add_argument('--mask-dir', default='data/foundry_to_mask/line_masks',
                        help='Directory with mask files')
    parser.add_argument('--output-dir', default='outputs/wall_segmentation',
                        help='Output directory for checkpoints')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loading workers')

    # Tiling
    parser.add_argument('--tile-grid-cells', type=int, default=8,
                        help='Grid cells per tile')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Model input size')

    # Model
    parser.add_argument('--no-aspp', action='store_true',
                        help='Disable ASPP in bottleneck')
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable attention gates')

    # Classes
    parser.add_argument('--merge-terrain', action='store_true',
                        help='Merge terrain class into wall class (4 -> 3 classes)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable augmentations')

    args = parser.parse_args()

    # Create config
    config = Config(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        tile_grid_cells=args.tile_grid_cells,
        tile_size=args.tile_size,
        use_aspp=not args.no_aspp,
        use_attention=not args.no_attention,
        seed=args.seed,
        augment=not args.no_augment,
        merge_terrain=args.merge_terrain,
    )

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
