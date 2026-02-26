"""
Training script for wall segmentation model.

Usage:
    python -m primary_model_training.train
    python -m primary_model_training.train --epochs 50 --batch-size 4
"""

import argparse
import ctypes
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


def _show_power_limit_reminder():
    """Show a popup reminding the user to run the GPU power limit bat file."""
    bat_path = Path(__file__).resolve().parent.parent / 'gpu_power_training.bat'
    ctypes.windll.user32.MessageBoxW(
        0,
        f"Run this as admin before training:\n\n{bat_path}",
        "GPU Power Limit Reminder",
        0x40,  # MB_ICONINFORMATION
    )

from .config import Config
from .model import create_model
from .dataset import WallSegmentationDataset, ValidationDataset, create_dataloader
from .losses import FocalLoss


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


class EMAModel:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of the full model state_dict (parameters + buffers)
    updated as: ema_param = decay * ema_param + (1 - decay) * model_param
    Buffers (e.g. batch norm running stats) are copied directly.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.param_names = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_names.add(name)
        for name, tensor in model.state_dict().items():
            self.shadow[name] = tensor.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, tensor in model.state_dict().items():
            if name in self.param_names:
                # EMA for trainable parameters
                self.shadow[name].mul_(self.decay).add_(
                    tensor, alpha=1.0 - self.decay
                )
            else:
                # Copy buffers directly (batch norm running stats, etc.)
                self.shadow[name].copy_(tensor)

    def apply_shadow(self, model: nn.Module):
        """Swap model state with EMA state (for validation/inference)."""
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        """Restore original model state after validation."""
        model.load_state_dict(self.backup)
        self.backup = {}

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))


class Trainer:
    """Training manager."""

    def __init__(self, config: Config, resume_path: Optional[str] = None):
        self.config = config
        self.resume_path = resume_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Set seed
        set_seed(config.seed)

        # Create output directory (reuse parent dir if resuming)
        if resume_path:
            self.output_dir = Path(resume_path).parent
        else:
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
            metadata_file=config.metadata_file,
            tile_grid_cells=config.tile_grid_cells,
            tile_size=config.tile_size,
            tiles_per_image=config.tiles_per_image,
            mask_scale=config.mask_scale,
            augment=config.augment,
            merge_terrain=config.merge_terrain,
            watabou_dir=config.watabou_dir,
            watabou_include_prob=config.watabou_include_prob,
            use_imagenet_norm=config.use_imagenet_norm,
            global_image_size=config.global_image_size if config.use_global_context else 0,
            mask_dilation=config.mask_dilation,
            grayscale_aug=config.grayscale_aug,
            tile_context_cells=config.tile_context_cells,
            sample_texture_crops=(config.model_type == "segformer_texture"),
            texture_crop_size=config.texture_crop_size,
            texture_crops_per_class=config.texture_crops_per_class,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if config.num_workers > 0 else None,
        )

        # Compute or use provided class weights
        if config.class_weights:
            class_weights = torch.tensor(config.class_weights, dtype=torch.float32)
        else:
            class_weights = compute_class_weights(self.train_dataset, config.num_classes)

        # Always create CE criterion; create Focal if needed by either flag.
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        needs_focal = config.use_focal_loss or config.focal_loss_schedule
        self.focal_criterion = (
            FocalLoss(weight=class_weights.to(self.device), gamma=config.focal_gamma)
            if needs_focal else None
        )

        if config.use_focal_loss and not config.focal_loss_schedule:
            self.criterion = self.focal_criterion
            print(f"Using Focal Loss (gamma={config.focal_gamma})")
        elif config.focal_loss_schedule:
            self.criterion = self.ce_criterion
            end_str = str(config.focal_schedule_end) if config.focal_schedule_end > 0 else "never"
            print(f"Focal schedule: CE until epoch {config.focal_schedule_start}, "
                  f"then Focal (gamma={config.focal_gamma}), "
                  f"then CE at epoch {end_str}")
        else:
            self.criterion = self.ce_criterion

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

        # EMA
        self.ema = None
        if config.ema_decay > 0:
            self.ema = EMAModel(self.model, decay=config.ema_decay)
            print(f"EMA enabled with decay={config.ema_decay}")

        # Tracking
        self.best_loss = float('inf')
        self.best_iou = 0.0
        self.best_state = None  # Best model weights kept in CPU RAM
        self.best_epoch = 0
        self.epochs_since_best = 0
        self.start_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'lr': [],
        }

        # Resume from checkpoint
        if resume_path:
            self._load_checkpoint(resume_path)

    def _load_checkpoint(self, path: str):
        """Load model, optimizer, scheduler, and tracking state from checkpoint."""
        print(f"Resuming from: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_iou = checkpoint.get('best_iou', 0.0)

        # Restore EMA state if available
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        # Reload history if available
        history_path = Path(path).parent / 'history.json'
        if history_path.exists():
            with open(history_path) as f:
                self.history = json.load(f)

        print(f"  Resumed at epoch {self.start_epoch}, best IoU: {self.best_iou:.4f}")

    def _model_kwargs(self, batch: Dict) -> Dict:
        """Build extra keyword arguments for model forward pass."""
        kwargs = {}
        if self.config.use_global_context and 'global_image' in batch:
            kwargs['global_image'] = batch['global_image'].to(self.device)
        if self.config.model_type == 'segformer_texture':
            if 'wall_crops' in batch:
                kwargs['wall_crops'] = batch['wall_crops'].to(self.device)
            if 'door_crops' in batch:
                kwargs['door_crops'] = batch['door_crops'].to(self.device)
        return kwargs

    def _update_criterion(self, epoch: int):
        """Switch loss function according to the focal loss phase schedule."""
        if not self.config.focal_loss_schedule:
            return

        start = self.config.focal_schedule_start
        end = self.config.focal_schedule_end
        in_focal_phase = epoch >= start and (end == 0 or epoch < end)

        if in_focal_phase and self.criterion is not self.focal_criterion:
            self.criterion = self.focal_criterion
            print(f"  [Focal schedule] Epoch {epoch}: switching to Focal Loss "
                  f"(gamma={self.config.focal_gamma})")
        elif not in_focal_phase and self.criterion is not self.ce_criterion:
            self.criterion = self.ce_criterion
            print(f"  [Focal schedule] Epoch {epoch}: switching to Cross-Entropy Loss")

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
            extra_kwargs = self._model_kwargs(batch)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images, **extra_kwargs)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, **extra_kwargs)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

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
        """Run validation (uses EMA weights if available)."""
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()

        total_loss = 0.0
        all_ious = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.train_loader:  # TODO: Use separate validation set
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                extra_kwargs = self._model_kwargs(batch)

                outputs = self.model(images, **extra_kwargs)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                metrics = compute_metrics(preds, masks, self.config.num_classes)
                all_ious.append(metrics['mean_iou'])

                num_batches += 1

                # Limit validation batches
                if num_batches >= 50:
                    break

        if self.ema is not None:
            self.ema.restore(self.model)

        return {
            'loss': total_loss / max(num_batches, 1),
            'mean_iou': np.mean(all_ious) if all_ious else 0.0,
        }

    def _snapshot_best(self):
        """Copy current best weights to CPU RAM (EMA weights if available)."""
        if self.ema is not None:
            self.best_state = self.ema.state_dict()  # already returns CPU tensors
        else:
            import copy
            self.best_state = copy.deepcopy(self.model.state_dict())
            for k in self.best_state:
                self.best_state[k] = self.best_state[k].cpu()
        self.best_epoch = self.start_epoch  # updated by caller

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint at save intervals."""
        # Snapshot best weights to CPU RAM whenever we get a new best
        if is_best:
            self._snapshot_best()
            self.best_epoch = epoch

        # Only write to disk at save intervals
        if (epoch + 1) % self.config.save_interval == 0:
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
            if self.ema is not None:
                checkpoint['ema_state_dict'] = self.ema.state_dict()

            torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch{epoch+1}.pt')

            # Write best from RAM snapshot
            if self.best_state is not None:
                best_checkpoint = {
                    'epoch': self.best_epoch,
                    'model_state_dict': self.best_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'best_iou': self.best_iou,
                    'config': vars(self.config),
                }
                if self.scheduler is not None:
                    best_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                if self.ema is not None:
                    best_checkpoint['ema_state_dict'] = self.ema.state_dict()
                torch.save(best_checkpoint, self.output_dir / 'checkpoint_best.pt')

    def _save_final_checkpoint(self, epoch: int):
        """Save latest and best checkpoint on exit."""
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
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')

        if self.best_state is not None:
            best_checkpoint = {
                'epoch': self.best_epoch,
                'model_state_dict': self.best_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'best_iou': self.best_iou,
                'config': vars(self.config),
            }
            if self.scheduler is not None:
                best_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.ema is not None:
                best_checkpoint['ema_state_dict'] = self.ema.state_dict()
            torch.save(best_checkpoint, self.output_dir / 'checkpoint_best.pt')

    def _write_stop_bat(self):
        """Write a stop_training.bat to the output dir for graceful early exit."""
        stop_file = self.output_dir / 'STOP'
        bat_path = self.output_dir / 'stop_training.bat'
        bat_path.write_text(
            f'@echo off\n'
            f'echo Requesting graceful stop after current epoch...\n'
            f'echo. > "{stop_file}"\n'
            f'echo Done. Training will save and exit at end of current epoch.\n'
            f'pause\n',
            encoding='utf-8',
        )
        return bat_path

    def _check_stop_requested(self) -> bool:
        """Return True (and remove sentinel) if a graceful stop was requested."""
        stop_file = self.output_dir / 'STOP'
        if stop_file.exists():
            stop_file.unlink()
            return True
        return False

    def train(self):
        """Main training loop."""
        # Remind user to set GPU power limit
        if self.device.type == 'cuda' and os.name == 'nt':
            _show_power_limit_reminder()

        # Write stop helper
        bat_path = self._write_stop_bat()

        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"To stop early: run  {bat_path}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = time.time()

            # Resample data sources for this epoch (subsamples Watabou, randomizes variants)
            self.train_dataset.resample_for_epoch()

            # Switch loss function if focal schedule is active
            self._update_criterion(epoch + 1)

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
                self.epochs_since_best = 0
            else:
                self.epochs_since_best += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['mean_iou']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Time: {epoch_time:.1f}s" + (" [BEST]" if is_best else
                  f" (no improvement for {self.epochs_since_best} epochs)"))

            # Early stopping
            patience = self.config.early_stopping_patience
            if patience > 0 and self.epochs_since_best >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
                break

            # Graceful stop via sentinel file
            if self._check_stop_requested():
                print("\nStop requested — saving and exiting gracefully.")
                break

            # Save history every epoch (small file, lets the watcher detect progress)
            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

        # Final saves (history + pending best/latest checkpoint)
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        self._save_final_checkpoint(epoch)

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training complete in {total_time / 60:.1f} minutes")
        print(f"Best IoU: {self.best_iou:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train wall segmentation model')

    # Data paths
    parser.add_argument('--image-dir', default='data/foundry_to_mask/Map_Images',
                        help='Directory with battlemap images')
    parser.add_argument('--mask-dir', default='data/foundry_to_mask/line_masks',
                        help='Directory with mask files')
    parser.add_argument('--metadata-file', default=None,
                        help='JSON file with per-map grid sizes')
    parser.add_argument('--output-dir', default='outputs/wall_segmentation',
                        help='Output directory for checkpoints')

    # Training
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--early-stopping', type=int, default=50,
                        help='Stop after N epochs without improvement (0 = disabled)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loading workers')
    parser.add_argument('--tiles-per-image', type=int, default=4,
                        help='Random tiles extracted per image per epoch (default: 4)')

    # Tiling
    parser.add_argument('--tile-grid-cells', type=int, default=8,
                        help='Grid cells per tile')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Model input size')

    # Model
    parser.add_argument('--model', default='unet',
                        choices=['unet', 'segformer', 'segformer_gc', 'segformer_texture'],
                        help='Model architecture (default: unet)')
    parser.add_argument('--segformer-variant', default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer variant (default: b0)')
    parser.add_argument('--no-aspp', action='store_true',
                        help='Disable ASPP in bottleneck (UNet only)')
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable attention gates (UNet only)')

    # Classes
    parser.add_argument('--merge-terrain', action='store_true',
                        help='Merge terrain class into wall class (4 -> 3 classes)')

    # Watabou data source
    parser.add_argument('--watabou-dir', default=None,
                        help='Watabou data directory (contains watabou_images/, watabou_edge_mask/, watabou_recessed_mask/)')
    parser.add_argument('--watabou-prob', type=float, default=0.36,
                        help='Per-epoch inclusion probability for Watabou maps (default: 0.36)')

    # Loss function
    parser.add_argument('--focal-loss', action='store_true',
                        help='Use focal loss instead of cross-entropy')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma (default: 2.0)')
    parser.add_argument('--focal-schedule', action='store_true',
                        help='CE->Focal->CE curriculum (default: off). '
                             'Use with --focal-start-epoch and --focal-end-epoch')
    parser.add_argument('--focal-start-epoch', type=int, default=15,
                        help='Epoch to switch from CE to Focal loss (default: 15)')
    parser.add_argument('--focal-end-epoch', type=int, default=30,
                        help='Epoch to switch back from Focal to CE (0 = never, default: 30)')

    # Mask dilation
    parser.add_argument('--mask-dilation', type=int, default=0,
                        help='Dilate training masks by N pixels to strengthen wall signal (default: 0)')

    # Augmentation
    parser.add_argument('--no-grayscale-aug', action='store_true',
                        help='Disable random greyscale augmentation (enabled by default, p=0.3)')

    # Tiling
    parser.add_argument('--tile-context-cells', type=int, default=1,
                        help='Grid cells of context border included around each tile (default: 1). '
                             '0 = disabled (old overlap-based stitching)')
    parser.add_argument('--no-tile-context', action='store_true',
                        help='Disable context border tiling (equivalent to --tile-context-cells 0)')

    # Texture prototype (segformer_texture only)
    parser.add_argument('--texture-crop-size', type=int, default=64,
                        help='Texture crop side length in 512px tile space (default: 64)')
    parser.add_argument('--texture-crops-per-class', type=int, default=2,
                        help='Number of texture crops to sample per class per tile (default: 2)')

    # Global context
    parser.add_argument('--global-context', action='store_true',
                        help='Enable global context for segformer_texture (auto-enabled for segformer_gc)')
    parser.add_argument('--no-global-context', action='store_true',
                        help='Disable 256x256 whole-map global context for segformer_gc')

    # Checkpointing
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50)')

    # EMA
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate (0 = disabled, default: 0.999)')

    # Resume
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable augmentations')

    args = parser.parse_args()

    tile_context = 0 if args.no_tile_context else args.tile_context_cells

    # Create config
    config = Config(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        model_type=args.model,
        segformer_variant=args.segformer_variant,
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
        watabou_dir=args.watabou_dir,
        watabou_include_prob=args.watabou_prob,
        early_stopping_patience=args.early_stopping,
        save_interval=args.save_interval,
        ema_decay=args.ema_decay,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        focal_loss_schedule=args.focal_schedule,
        focal_schedule_start=args.focal_start_epoch,
        focal_schedule_end=args.focal_end_epoch,
        mask_dilation=args.mask_dilation,
        grayscale_aug=not args.no_grayscale_aug,
        tile_context_cells=tile_context,
        texture_crop_size=args.texture_crop_size,
        texture_crops_per_class=args.texture_crops_per_class,
        tiles_per_image=args.tiles_per_image,
    )

    # --global-context / --no-global-context override Config.__post_init__ defaults
    if args.global_context:
        config.use_global_context = True
    if args.no_global_context:
        config.use_global_context = False

    # Train
    trainer = Trainer(config, resume_path=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
