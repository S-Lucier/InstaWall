import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate to focus on relevant spatial regions.

    Helps the model focus on small/rare features like doors.
    Applied between encoder skip connections and decoder.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # F_g: channels from decoder (gating signal)
        # F_l: channels from encoder skip connection
        # F_int: intermediate channels for attention computation

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: gating signal from decoder (coarser scale)
        x: skip connection from encoder (finer scale)

        Returns: attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi  # Element-wise multiplication (attention weighting)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - multi-scale feature extraction.

    Captures context at multiple scales simultaneously:
    - Small dilation: fine details (good for doors)
    - Large dilation: broad context (good for walls/room layout)
    - Global pooling: overall scene understanding
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Multiple parallel branches with different receptive fields
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolutions with different dilation rates
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

        # Global average pooling for image-level context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Project concatenated features back to output channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]  # Spatial dimensions

        # Extract features at multiple scales
        feat1 = self.conv1x1(x)              # 1x1 receptive field
        feat2 = self.conv3x3_rate6(x)        # Larger receptive field (doors)
        feat3 = self.conv3x3_rate12(x)       # Even larger (hallways)
        feat4 = self.conv3x3_rate18(x)       # Largest (rooms/walls)

        # Global context
        feat5 = self.global_pool(x)
        feat5 = self.global_conv(feat5)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)

        # Concatenate all scales and project
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)


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


class AttentionASPPUNet(nn.Module):
    """U-Net with Attention Gates and Multi-Scale ASPP for improved small object detection.

    Enhancements:
    1. ASPP bottleneck: Multi-scale context for detecting both large walls and tiny doors
    2. Attention gates: Focus on important regions (helps with sparse door detection)

    Args:
        in_channels: Number of input channels (3 for RGB images)
        out_channels: Number of output channels (3 for 3-class segmentation)
        features: List of feature sizes for each level [64, 128, 256, 512]
    """

    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ASPP Bottleneck (multi-scale feature extraction)
        self.bottleneck = ASPP(features[-1], features[-1] * 2)

        # Decoder (upsampling path) with Attention Gates
        for feature in reversed(features):
            # Upsampling
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )

            # Attention gate (applied to skip connection before concatenation)
            self.attention_gates.append(
                AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2)
            )

            # Double conv after concatenation
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # ASPP Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder with attention
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]

            # Apply attention gate to skip connection
            attention_gate = self.attention_gates[idx // 2]
            skip_connection = attention_gate(g=x, x=skip_connection)

            # Handle potential size mismatch from padding
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate and apply double conv
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# Test function
if __name__ == "__main__":
    print("Testing AttentionASPPUNet...")
    model = AttentionASPPUNet(in_channels=3, out_channels=3)
    model.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
    x = torch.randn(1, 3, 512, 512)

    print(f"Input shape: {x.shape}")
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Compare to standard U-Net
    from unet_model import UNet
    unet = UNet(in_channels=3, out_channels=3)
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"\nStandard U-Net parameters: {unet_params:,}")
    print(f"Parameter increase: {((total_params - unet_params) / unet_params * 100):.1f}%")
