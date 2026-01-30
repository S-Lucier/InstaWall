"""
Fine-tuning script for segmentation using pretrained encoder.

Takes the encoder pretrained on self-supervised tasks and fine-tunes
it on labeled battlemap segmentation data.

Usage:
    python train_finetune.py --pretrained ./pretrained_models/encoder_only.pth \
                             --train_images ./data/train_images \
                             --train_masks ./data/train_masks
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'WatabouTestModel' / 'Model Imports'))

from models.encoder import SharedEncoder, ASPP, DoubleConv


class AttentionGate(nn.Module):
    """Attention gate for decoder (same as original model)."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SegmentationModel(nn.Module):
    """Segmentation model using pretrained encoder.

    Architecture matches AttentionASPPUNet but uses a pretrained encoder.
    """

    def __init__(self, encoder, num_classes=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = encoder
        self.features = features

        # Build decoder
        self.ups = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        bottleneck_ch = features[-1] * 2  # After ASPP

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(bottleneck_ch if feature == features[-1]
                                  else feature * 2, feature, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            bottleneck_ch = feature

        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        bottleneck, skip_connections = self.encoder(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder with attention
        x = bottleneck
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip = skip_connections[idx // 2]

            # Apply attention
            attention_gate = self.attention_gates[idx // 2]
            skip = attention_gate(g=x, x=skip)

            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def load_pretrained_encoder(checkpoint_path, device='cuda'):
    """Load pretrained encoder weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint.get('config', {})
    features = config.get('features', [64, 128, 256, 512])

    # Create encoder
    encoder = SharedEncoder(in_channels=3, features=features, use_aspp=True)

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print(f"Loaded pretrained encoder from {checkpoint_path}")

    return encoder, features


def create_segmentation_model(pretrained_path, num_classes=3, device='cuda', freeze_encoder=False):
    """Create segmentation model with pretrained encoder.

    Args:
        pretrained_path: Path to pretrained encoder checkpoint
        num_classes: Number of segmentation classes
        device: Device to use
        freeze_encoder: If True, freeze encoder weights (only train decoder)
    """
    encoder, features = load_pretrained_encoder(pretrained_path, device)

    model = SegmentationModel(encoder, num_classes=num_classes, features=features)
    model = model.to(device)

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune pretrained model for segmentation')

    # Data
    parser.add_argument('--train_images', type=str, required=True)
    parser.add_argument('--train_masks', type=str, required=True)
    parser.add_argument('--val_images', type=str, default=None)
    parser.add_argument('--val_masks', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained encoder')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_encoder', type=float, default=1e-5,
                       help='Learning rate for encoder (lower for fine-tuning)')
    parser.add_argument('--lr_decoder', type=float, default=1e-4,
                       help='Learning rate for decoder')

    # Output
    parser.add_argument('--output_dir', type=str, default='./finetuned_models')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Fine-tuning Segmentation Model")
    print("=" * 60)
    print(f"Pretrained encoder: {args.pretrained}")
    print(f"Training images: {args.train_images}")
    print(f"Number of classes: {args.num_classes}")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_segmentation_model(
        args.pretrained,
        num_classes=args.num_classes,
        device=device,
        freeze_encoder=args.freeze_encoder
    )

    # Setup optimizer with different learning rates
    if not args.freeze_encoder:
        optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': args.lr_encoder},
            {'params': model.ups.parameters(), 'lr': args.lr_decoder},
            {'params': model.attention_gates.parameters(), 'lr': args.lr_decoder},
            {'params': model.final_conv.parameters(), 'lr': args.lr_decoder},
        ])
        print(f"Encoder LR: {args.lr_encoder}, Decoder LR: {args.lr_decoder}")
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_decoder
        )

    print("\nModel ready for fine-tuning!")
    print("To train, integrate with your existing training loop from")
    print("WatabouTestModel/Model Imports/train_3class_attention_multi_extended.py")

    # Save model architecture for reference
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick test
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        out = model(x)
        print(f"\nTest forward pass:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")

    print("\nFine-tuning setup complete!")


if __name__ == '__main__':
    main()
