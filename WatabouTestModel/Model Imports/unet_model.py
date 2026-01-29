import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv2D layers with BatchNorm and ReLU activation.
    This is the basic building block of U-Net."""

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


class UNet(nn.Module):
    """U-Net architecture for image segmentation.

    Args:
        in_channels: Number of input channels (3 for RGB images)
        out_channels: Number of output channels (1 for binary segmentation, N for multi-class)
        features: List of feature sizes for each level [64, 128, 256, 512]
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]

            # Handle size mismatches due to pooling
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            # Concatenate skip connection
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx + 1](x)  # DoubleConv

        return self.final_conv(x)


def test_model():
    """Test function to verify model architecture."""
    x = torch.randn((1, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert preds.shape == (1, 1, 256, 256), "Output shape mismatch!"
    print("Model test passed!")


if __name__ == "__main__":
    test_model()
