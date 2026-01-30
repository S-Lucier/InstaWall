"""Pretext tasks for self-supervised pretraining."""

from .edge_prediction import EdgePredictionTask, generate_edge_labels
from .colorization import ColorizationTask, rgb_to_grayscale
from .masked_autoencoder import MAETask, generate_mae_input
from .jigsaw import JigsawTask, PERMUTATIONS_3x3, PERMUTATIONS_2x2

__all__ = [
    'EdgePredictionTask',
    'ColorizationTask',
    'MAETask',
    'JigsawTask',
    'generate_edge_labels',
    'rgb_to_grayscale',
    'generate_mae_input',
    'PERMUTATIONS_3x3',
    'PERMUTATIONS_2x2',
]
