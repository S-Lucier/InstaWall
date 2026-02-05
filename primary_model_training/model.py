"""
UNet model for wall segmentation.

Architecture supports optional ASPP bottleneck and attention gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """Two consecutive Conv2D layers with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
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
    """Atrous Spatial Pyramid Pooling for multi-scale feature extraction."""

    def __init__(self, in_channels: int, out_channels: int):
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


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Upsampled features from decoder (lower resolution context)
            skip: Skip connection features from encoder (higher resolution detail)

        Returns:
            Attention-weighted skip connection features
        """
        # Resize gate to match skip resolution if needed
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=False)

        g = self.W_gate(gate)
        s = self.W_skip(skip)

        attention = self.psi(self.relu(g + s))
        return skip * attention


class WallSegmentationUNet(nn.Module):
    """
    UNet for wall segmentation with optional ASPP and attention gates.

    Args:
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        features: List of feature sizes for each encoder level
        use_aspp: Use ASPP in bottleneck for multi-scale features
        use_attention: Use attention gates in decoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        features: List[int] = None,
        use_aspp: bool = True,
        use_attention: bool = True
    ):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.features = features
        self.use_aspp = use_aspp
        self.use_attention = use_attention

        # Encoder (downsampling path)
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        if use_aspp:
            self.bottleneck = ASPP(features[-1], features[-1] * 2)
        else:
            self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        if use_attention:
            self.attention_gates = nn.ModuleList()

        reversed_features = features[::-1]
        bottleneck_channels = features[-1] * 2

        for i, feature in enumerate(reversed_features):
            # Upsample
            in_ch = bottleneck_channels if i == 0 else reversed_features[i - 1]
            self.ups.append(
                nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2)
            )

            # Attention gate (before concatenation)
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(feature, feature, feature // 2)
                )

            # Double conv after concatenation
            self.up_convs.append(DoubleConv(feature * 2, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Class logits (B, num_classes, H, W)
        """
        # Encoder
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        for i, (up, up_conv) in enumerate(zip(self.ups, self.up_convs)):
            x = up(x)

            skip = skip_connections[i]

            # Handle size mismatch (can happen with odd dimensions)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # Apply attention gate
            if self.use_attention:
                skip = self.attention_gates[i](x, skip)

            # Concatenate and convolve
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        return self.final_conv(x)

    def get_num_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config) -> WallSegmentationUNet:
    """Create model from config."""
    return WallSegmentationUNet(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        features=config.features,
        use_aspp=config.use_aspp,
        use_attention=config.use_attention
    )


if __name__ == "__main__":
    # Test model
    print("Testing WallSegmentationUNet...")

    model = WallSegmentationUNet(
        in_channels=3,
        num_classes=5,
        features=[64, 128, 256, 512],
        use_aspp=True,
        use_attention=True
    )
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        out = model(x)

    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Test with different sizes
    for size in [256, 384, 512]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(x)
        print(f"  {size}x{size} -> {out.shape}")
