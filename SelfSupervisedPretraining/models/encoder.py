"""
Shared encoder for self-supervised pretraining.

This encoder is pretrained on multiple pretext tasks (edge detection, colorization,
MAE, jigsaw) and then transferred to the segmentation model for fine-tuning.

Architecture matches the AttentionASPPUNet encoder for direct weight transfer.
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


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - multi-scale feature extraction.

    Captures context at multiple scales simultaneously.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_rate6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     dilation=6, padding=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_rate12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     dilation=12, padding=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_rate18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     dilation=18, padding=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]

        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_rate6(x)
        feat3 = self.conv3x3_rate12(x)
        feat4 = self.conv3x3_rate18(x)

        feat5 = self.global_pool(x)
        feat5 = self.global_conv(feat5)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)

        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)


class SharedEncoder(nn.Module):
    """Shared encoder for multi-task pretraining.

    Architecture matches AttentionASPPUNet encoder for direct weight transfer.
    Returns multi-scale features for use by task-specific decoders.

    Args:
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        features: List of feature sizes for each level [64, 128, 256, 512]
        use_aspp: Whether to use ASPP in bottleneck (matches AttentionASPPUNet)
    """

    def __init__(self, in_channels=3, features=[64, 128, 256, 512], use_aspp=True):
        super().__init__()
        self.features = features
        self.use_aspp = use_aspp

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling path)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        if use_aspp:
            self.bottleneck = ASPP(features[-1], features[-1] * 2)
        else:
            self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.bottleneck_channels = features[-1] * 2

    def forward(self, x):
        """
        Returns:
            bottleneck: Bottleneck features (lowest resolution, highest channels)
            skip_connections: List of skip connection features (high to low resolution)
        """
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        bottleneck = self.bottleneck(x)

        return bottleneck, skip_connections

    def get_output_channels(self):
        """Returns bottleneck channel count for building decoders."""
        return self.bottleneck_channels


class GrayscaleAdapter(nn.Module):
    """Adapter to allow grayscale input to RGB encoder.

    For colorization task: input is grayscale, but we want to use
    the same encoder pretrained on RGB.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Learnable projection from 1 channel to 3 channels
        self.expand = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        # Initialize to replicate grayscale across RGB
        nn.init.constant_(self.expand.weight, 1.0)

    def forward(self, x):
        # x: (B, 1, H, W) grayscale
        x = self.expand(x)  # (B, 3, H, W)
        return self.encoder(x)


if __name__ == "__main__":
    print("Testing SharedEncoder...")
    encoder = SharedEncoder(in_channels=3, features=[64, 128, 256, 512])
    encoder.eval()

    x = torch.randn(2, 3, 512, 512)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        bottleneck, skips = encoder(x)

    print(f"Bottleneck shape: {bottleneck.shape}")
    print(f"Skip connection shapes: {[s.shape for s in skips]}")

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
