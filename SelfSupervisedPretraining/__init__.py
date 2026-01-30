"""
Self-Supervised Pretraining for Battlemap Segmentation

This package implements multi-task self-supervised pretraining to learn
general visual features from unlabeled battlemap images.

Pretext Tasks:
- Edge prediction: Learn boundary detection
- Colorization: Learn material differences
- Masked autoencoding: Learn spatial structure
- Jigsaw puzzle: Learn spatial relationships

Usage:
    # Pretraining
    python train_pretrain.py --data_dir ./battlemaps --epochs 100

    # Fine-tuning
    python train_finetune.py --pretrained ./pretrained/encoder.pth --train_images ./data
"""

from .models import MultiTaskPretrainer, MultiTaskLoss, SharedEncoder
from .pretext_tasks import EdgePredictionTask, ColorizationTask, MAETask, JigsawTask
from .data import BattlemapDataset, PretrainingDataset, create_dataloader

__version__ = '0.1.0'

__all__ = [
    'MultiTaskPretrainer',
    'MultiTaskLoss',
    'SharedEncoder',
    'EdgePredictionTask',
    'ColorizationTask',
    'MAETask',
    'JigsawTask',
    'BattlemapDataset',
    'PretrainingDataset',
    'create_dataloader',
]
