"""
Task-specific decoders for self-supervised pretraining.

Each decoder takes the shared encoder's output and produces task-specific predictions.
These decoders are discarded after pretraining - only the encoder is kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2D layers with BatchNorm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Standard U-Net decoder with skip connections.

    Used for dense prediction tasks (edge detection, colorization, MAE).

    Args:
        bottleneck_channels: Number of channels from encoder bottleneck
        features: List of feature sizes (should match encoder, reversed)
        out_channels: Number of output channels
    """

    def __init__(self, bottleneck_channels, features=[512, 256, 128, 64], out_channels=1):
        super().__init__()
        self.ups = nn.ModuleList()

        # First upsampling from bottleneck
        in_ch = bottleneck_channels
        for feature in features:
            self.ups.append(
                nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2)
            )
            # After concat with skip connection, channels double
            self.ups.append(DoubleConv(feature * 2, feature))
            in_ch = feature

        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, bottleneck, skip_connections):
        """
        Args:
            bottleneck: Encoder bottleneck features
            skip_connections: List of skip connections (high to low resolution)
                             Already reversed from encoder output
        """
        x = bottleneck

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip = skip_connections[idx // 2]

            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)  # DoubleConv

        return self.final_conv(x)


class LightweightDecoder(nn.Module):
    """Lightweight decoder without skip connections.

    Faster training, used when skip connections aren't critical.
    Good for MAE where we want the encoder to learn structure without
    relying on fine-grained skip connection details.
    """

    def __init__(self, bottleneck_channels, out_channels=3, target_size=512):
        super().__init__()
        self.target_size = target_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, bottleneck, skip_connections=None):
        """Skip connections ignored - lightweight decoder."""
        x = self.decoder(bottleneck)
        # Ensure output matches target size
        if x.shape[2:] != (self.target_size, self.target_size):
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                            mode='bilinear', align_corners=False)
        return x


class JigsawHead(nn.Module):
    """Classification head for jigsaw puzzle task.

    Takes bottleneck features and predicts which permutation was applied.

    Args:
        bottleneck_channels: Number of channels from encoder bottleneck
        num_permutations: Number of possible permutations to classify
    """

    def __init__(self, bottleneck_channels, num_permutations=100):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_permutations)
        )

    def forward(self, bottleneck, skip_connections=None):
        """Skip connections ignored - uses only bottleneck."""
        return self.head(bottleneck)


class EdgeDecoder(UNetDecoder):
    """Decoder for edge prediction task.

    Output: Single channel edge probability map.
    """

    def __init__(self, bottleneck_channels, features=[512, 256, 128, 64]):
        super().__init__(bottleneck_channels, features, out_channels=1)


class ColorizationDecoder(UNetDecoder):
    """Decoder for colorization task.

    Output: 3-channel RGB prediction from grayscale input.
    """

    def __init__(self, bottleneck_channels, features=[512, 256, 128, 64]):
        super().__init__(bottleneck_channels, features, out_channels=3)


class MAEDecoder(LightweightDecoder):
    """Decoder for masked autoencoder reconstruction.

    Uses lightweight decoder without skip connections to force
    the encoder to learn more complete representations.
    """

    def __init__(self, bottleneck_channels, target_size=512):
        super().__init__(bottleneck_channels, out_channels=3, target_size=target_size)


if __name__ == "__main__":
    print("Testing decoders...")

    # Simulate encoder output
    bottleneck = torch.randn(2, 1024, 32, 32)  # After ASPP with 512*2=1024 channels
    skips = [
        torch.randn(2, 512, 32, 32),   # Level 4
        torch.randn(2, 256, 64, 64),   # Level 3
        torch.randn(2, 128, 128, 128), # Level 2
        torch.randn(2, 64, 256, 256),  # Level 1
    ]

    print(f"Bottleneck: {bottleneck.shape}")
    print(f"Skips: {[s.shape for s in skips]}")

    # Test edge decoder
    edge_dec = EdgeDecoder(1024)
    edge_out = edge_dec(bottleneck, skips)
    print(f"Edge decoder output: {edge_out.shape}")

    # Test colorization decoder
    color_dec = ColorizationDecoder(1024)
    color_out = color_dec(bottleneck, skips)
    print(f"Colorization decoder output: {color_out.shape}")

    # Test MAE decoder
    mae_dec = MAEDecoder(1024, target_size=256)
    mae_out = mae_dec(bottleneck)
    print(f"MAE decoder output: {mae_out.shape}")

    # Test jigsaw head
    jigsaw = JigsawHead(1024, num_permutations=100)
    jigsaw_out = jigsaw(bottleneck)
    print(f"Jigsaw head output: {jigsaw_out.shape}")
