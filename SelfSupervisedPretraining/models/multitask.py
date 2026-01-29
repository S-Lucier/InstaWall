"""
Multi-task self-supervised pretraining model.

Combines shared encoder with multiple task-specific decoders for:
- Edge prediction (learns boundary detection)
- Colorization (learns material/texture differences)
- Masked Autoencoding (learns spatial structure)
- Jigsaw puzzle (learns spatial relationships)

After pretraining, the encoder weights are transferred to the segmentation model.
"""

import torch
import torch.nn as nn

from .encoder import SharedEncoder, GrayscaleAdapter
from .decoders import EdgeDecoder, ColorizationDecoder, MAEDecoder, JigsawHead


class MultiTaskPretrainer(nn.Module):
    """Multi-task self-supervised pretraining model.

    Architecture:
        [Input] -> [Shared Encoder] -> [Task-Specific Decoder] -> [Output]

    Tasks:
        - 'edge': Predict Canny edges from RGB image
        - 'color': Predict RGB from grayscale
        - 'mae': Reconstruct masked image
        - 'jigsaw': Classify patch permutation

    Args:
        in_channels: Input channels (3 for RGB)
        features: Encoder feature sizes [64, 128, 256, 512]
        num_permutations: Number of jigsaw permutations
        image_size: Expected input image size (for MAE decoder)
    """

    def __init__(self, in_channels=3, features=[64, 128, 256, 512],
                 num_permutations=100, image_size=512):
        super().__init__()

        # Shared encoder (this is what we transfer after pretraining)
        self.encoder = SharedEncoder(in_channels=in_channels, features=features, use_aspp=True)
        bottleneck_ch = self.encoder.get_output_channels()

        # Grayscale adapter for colorization task
        self.grayscale_encoder = SharedEncoder(in_channels=1, features=features, use_aspp=True)

        # Task-specific decoders (discarded after pretraining)
        reversed_features = list(reversed(features))

        self.edge_decoder = EdgeDecoder(bottleneck_ch, reversed_features)
        self.color_decoder = ColorizationDecoder(bottleneck_ch, reversed_features)
        self.mae_decoder = MAEDecoder(bottleneck_ch, target_size=image_size)
        self.jigsaw_head = JigsawHead(bottleneck_ch, num_permutations)

        # Store config for saving/loading
        self.config = {
            'in_channels': in_channels,
            'features': features,
            'num_permutations': num_permutations,
            'image_size': image_size
        }

    def forward(self, x, task):
        """Forward pass for a specific task.

        Args:
            x: Input tensor
            task: One of 'edge', 'color', 'mae', 'jigsaw'

        Returns:
            Task-specific output
        """
        if task == 'color':
            # Colorization uses grayscale input
            bottleneck, skips = self.grayscale_encoder(x)
            skips = skips[::-1]  # Reverse for decoder
            return self.color_decoder(bottleneck, skips)

        # All other tasks use RGB encoder
        bottleneck, skips = self.encoder(x)
        skips = skips[::-1]  # Reverse for decoder

        if task == 'edge':
            return self.edge_decoder(bottleneck, skips)
        elif task == 'mae':
            return self.mae_decoder(bottleneck, skips)
        elif task == 'jigsaw':
            return self.jigsaw_head(bottleneck, skips)
        else:
            raise ValueError(f"Unknown task: {task}. Expected one of: edge, color, mae, jigsaw")

    def encode(self, x):
        """Get encoder features only (for downstream tasks)."""
        return self.encoder(x)

    def get_encoder_state_dict(self):
        """Get only the encoder weights for transfer."""
        return self.encoder.state_dict()

    def save_pretrained(self, path):
        """Save the full model and config."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'config': self.config
        }, path)
        print(f"Saved pretrained model to {path}")

    @classmethod
    def load_pretrained(cls, path, device='cuda'):
        """Load a pretrained model."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"Loaded pretrained model from {path}")
        return model

    @staticmethod
    def load_encoder_weights(path, device='cuda'):
        """Load only the encoder weights (for transfer to segmentation model)."""
        checkpoint = torch.load(path, map_location=device)
        return checkpoint['encoder_state_dict']


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task pretraining.

    Computes weighted sum of task-specific losses.
    """

    def __init__(self, weights=None):
        super().__init__()

        # Default weights (tune based on task importance and loss scales)
        self.weights = weights or {
            'edge': 0.3,
            'color': 0.2,
            'mae': 0.4,
            'jigsaw': 0.1
        }

        # Task-specific loss functions
        self.edge_loss = nn.BCEWithLogitsLoss()
        self.color_loss = nn.L1Loss()
        self.mae_loss = nn.MSELoss(reduction='none')  # Per-pixel for masking
        self.jigsaw_loss = nn.CrossEntropyLoss()

    def compute_edge_loss(self, pred, target):
        """Binary cross-entropy for edge prediction."""
        return self.edge_loss(pred, target)

    def compute_color_loss(self, pred, target):
        """L1 loss for colorization."""
        return self.color_loss(pred, target)

    def compute_mae_loss(self, pred, target, mask):
        """MSE loss only on masked regions.

        Args:
            pred: Reconstructed image
            target: Original image
            mask: Binary mask where 1 = masked (compute loss), 0 = visible (ignore)
        """
        loss = self.mae_loss(pred, target)
        # Expand mask to match loss dimensions if needed
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        if mask.shape[2:] != loss.shape[2:]:
            mask = nn.functional.interpolate(mask.float(), size=loss.shape[2:],
                                            mode='nearest')
        # Apply mask and compute mean
        masked_loss = loss * mask
        return masked_loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)

    def compute_jigsaw_loss(self, pred, target):
        """Cross-entropy for permutation classification."""
        return self.jigsaw_loss(pred, target)

    def forward(self, predictions, targets):
        """Compute total weighted loss.

        Args:
            predictions: Dict with keys 'edge', 'color', 'mae', 'jigsaw'
            targets: Dict with corresponding targets
                    (mae_target should include 'image' and 'mask')
        """
        losses = {}
        total_loss = 0.0

        if 'edge' in predictions:
            losses['edge'] = self.compute_edge_loss(predictions['edge'], targets['edge'])
            total_loss += self.weights['edge'] * losses['edge']

        if 'color' in predictions:
            losses['color'] = self.compute_color_loss(predictions['color'], targets['color'])
            total_loss += self.weights['color'] * losses['color']

        if 'mae' in predictions:
            losses['mae'] = self.compute_mae_loss(
                predictions['mae'],
                targets['mae_image'],
                targets['mae_mask']
            )
            total_loss += self.weights['mae'] * losses['mae']

        if 'jigsaw' in predictions:
            losses['jigsaw'] = self.compute_jigsaw_loss(predictions['jigsaw'], targets['jigsaw'])
            total_loss += self.weights['jigsaw'] * losses['jigsaw']

        losses['total'] = total_loss
        return losses


if __name__ == "__main__":
    print("Testing MultiTaskPretrainer...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MultiTaskPretrainer(
        in_channels=3,
        features=[64, 128, 256, 512],
        num_permutations=100,
        image_size=512
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")

    # Test each task
    model.eval()
    with torch.no_grad():
        # Edge prediction
        x_rgb = torch.randn(2, 3, 512, 512).to(device)
        edge_out = model(x_rgb, task='edge')
        print(f"Edge output shape: {edge_out.shape}")

        # Colorization
        x_gray = torch.randn(2, 1, 512, 512).to(device)
        color_out = model(x_gray, task='color')
        print(f"Color output shape: {color_out.shape}")

        # MAE
        mae_out = model(x_rgb, task='mae')
        print(f"MAE output shape: {mae_out.shape}")

        # Jigsaw
        jigsaw_out = model(x_rgb, task='jigsaw')
        print(f"Jigsaw output shape: {jigsaw_out.shape}")

    print("\nAll tasks working correctly!")
