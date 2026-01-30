"""
Self-supervised pretraining script.

Trains the shared encoder on multiple pretext tasks simultaneously:
- Edge prediction (learns boundary detection)
- Colorization (learns material differences)
- Masked autoencoding (learns spatial structure)
- Jigsaw puzzle (learns spatial relationships)

After pretraining, the encoder can be transferred to the segmentation model.

Usage:
    python train_pretrain.py --data_dir /path/to/battlemaps --epochs 100
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import MultiTaskPretrainer, MultiTaskLoss
from data.dataset import BattlemapDataset, PretrainingDataset, create_dataloader
from pretext_tasks import EdgePredictionTask, ColorizationTask, MAETask, JigsawTask


def parse_args():
    parser = argparse.ArgumentParser(description='Self-supervised pretraining for battlemap segmentation')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to unlabeled battlemap images (or comma-separated list)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Training image size')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')

    # Loss weights
    parser.add_argument('--edge_weight', type=float, default=0.3,
                       help='Weight for edge prediction loss')
    parser.add_argument('--color_weight', type=float, default=0.2,
                       help='Weight for colorization loss')
    parser.add_argument('--mae_weight', type=float, default=0.4,
                       help='Weight for MAE loss')
    parser.add_argument('--jigsaw_weight', type=float, default=0.1,
                       help='Weight for jigsaw loss')

    # Tasks to enable
    parser.add_argument('--tasks', type=str, default='edge,color,mae,jigsaw',
                       help='Comma-separated list of tasks to train')

    # Model
    parser.add_argument('--features', type=str, default='64,128,256,512',
                       help='Encoder feature sizes')

    # Output
    parser.add_argument('--output_dir', type=str, default='./pretrained_models',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    return parser.parse_args()


class PretrainTrainer:
    """Trainer for multi-task self-supervised pretraining."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Parse feature sizes
        self.features = [int(x) for x in args.features.split(',')]

        # Parse enabled tasks
        self.enabled_tasks = [t.strip() for t in args.tasks.split(',')]
        print(f"Enabled tasks: {self.enabled_tasks}")

        # Setup model
        self.model = self._build_model()

        # Setup loss
        self.loss_fn = MultiTaskLoss(weights={
            'edge': args.edge_weight,
            'color': args.color_weight,
            'mae': args.mae_weight,
            'jigsaw': args.jigsaw_weight
        })

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )

        # Setup AMP
        self.scaler = GradScaler() if args.amp else None

        # Setup data
        self.train_loader = self._build_dataloader()

        # Setup output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'task_losses': {}}

        # Resume if specified
        if args.resume:
            self._load_checkpoint(args.resume)

    def _build_model(self):
        """Build the multi-task pretraining model."""
        model = MultiTaskPretrainer(
            in_channels=3,
            features=self.features,
            num_permutations=100,
            image_size=self.args.image_size
        )
        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Encoder parameters: {encoder_params:,}")

        return model

    def _build_dataloader(self):
        """Build training dataloader."""
        # Parse data directories
        data_dirs = [d.strip() for d in self.args.data_dir.split(',')]

        # Create task objects
        tasks = {}
        if 'edge' in self.enabled_tasks:
            tasks['edge'] = EdgePredictionTask(method='canny', random_thresholds=True)
        if 'color' in self.enabled_tasks:
            tasks['color'] = ColorizationTask(add_noise=True)
        if 'mae' in self.enabled_tasks:
            tasks['mae'] = MAETask(patch_size=32, mask_ratio=0.75, variable_ratio=True)
        if 'jigsaw' in self.enabled_tasks:
            tasks['jigsaw'] = JigsawTask(grid_size=3, num_permutations=100)

        # Create datasets
        base_dataset = BattlemapDataset(
            image_dirs=data_dirs,
            image_size=self.args.image_size,
            augment=True
        )

        dataset = PretrainingDataset(base_dataset, tasks=tasks)

        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        print(f"Dataset size: {len(dataset)}")
        print(f"Batches per epoch: {len(loader)}")

        return loader

    def _load_checkpoint(self, path):
        """Load checkpoint to resume training."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)

        print(f"Resumed from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.model.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.model.config
        }

        # Regular checkpoint
        path = self.output_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # Best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Latest model (for easy resuming)
        latest_path = self.output_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {task: 0.0 for task in self.enabled_tasks}
        epoch_losses['total'] = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    losses = self._compute_losses(batch)
                    total_loss = losses['total']

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self._compute_losses(batch)
                total_loss = losses['total']
                total_loss.backward()
                self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            num_batches += 1

            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_loss = epoch_losses['total'] / num_batches
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (len(self.train_loader) - batch_idx - 1)
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"ETA: {eta:.0f}s")

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    def _compute_losses(self, batch):
        """Compute losses for all enabled tasks."""
        predictions = {}
        targets = {}

        # Edge prediction
        if 'edge' in self.enabled_tasks:
            edge_pred = self.model(batch['image'], task='edge')
            predictions['edge'] = edge_pred
            targets['edge'] = batch['edge_target']

        # Colorization
        if 'color' in self.enabled_tasks:
            color_pred = self.model(batch['gray_input'], task='color')
            predictions['color'] = color_pred
            targets['color'] = batch['color_target']

        # MAE
        if 'mae' in self.enabled_tasks:
            mae_pred = self.model(batch['mae_input'], task='mae')
            predictions['mae'] = mae_pred
            targets['mae_image'] = batch['mae_target']
            targets['mae_mask'] = batch['mae_mask']

        # Jigsaw
        if 'jigsaw' in self.enabled_tasks:
            jigsaw_pred = self.model(batch['jigsaw_input'], task='jigsaw')
            predictions['jigsaw'] = jigsaw_pred
            targets['jigsaw'] = batch['jigsaw_target']

        # Compute combined loss
        losses = self.loss_fn(predictions, targets)

        return losses

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)

        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.2e}")

            # Train
            epoch_losses = self.train_epoch(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log losses
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Total Loss: {epoch_losses['total']:.4f}")
            for task in self.enabled_tasks:
                if task in epoch_losses:
                    print(f"  {task.capitalize()} Loss: {epoch_losses[task]:.4f}")

            # Update history
            self.history['train_loss'].append(epoch_losses['total'])
            for task in self.enabled_tasks:
                if task not in self.history['task_losses']:
                    self.history['task_losses'][task] = []
                if task in epoch_losses:
                    self.history['task_losses'][task].append(epoch_losses[task])

            # Check for best model
            is_best = epoch_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = epoch_losses['total']
                print(f"  New best loss: {self.best_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0 or is_best:
                self._save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")

        # Save final model
        self._save_checkpoint(self.args.epochs - 1, False)

        # Save encoder only (for transfer)
        encoder_path = self.output_dir / 'encoder_only.pth'
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'config': {
                'features': self.features,
                'use_aspp': True
            }
        }, encoder_path)
        print(f"Saved encoder weights to {encoder_path}")


def main():
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("Self-Supervised Pretraining")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Tasks: {args.tasks}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Create trainer and run
    trainer = PretrainTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
