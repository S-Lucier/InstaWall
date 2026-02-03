import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "model_imports"))

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import argparse

from unet_model import UNet


def load_model(checkpoint_path, device):
    """Load trained U-Net model from checkpoint."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def preprocess_image(image_path, image_height=512, image_width=512):
    """Load and preprocess an image for inference."""
    image = np.array(Image.open(image_path).convert("RGB"))
    original_height, original_width = image.shape[:2]

    # Calculate scaled dimensions (before padding)
    # LongestMaxSize scales so the longest dimension equals max_size
    scale = max(image_height, image_width) / max(original_height, original_width)
    scaled_height = int(original_height * scale)
    scaled_width = int(original_width * scale)

    # Calculate padding (PadIfNeeded centers by default)
    pad_top = (image_height - scaled_height) // 2
    pad_left = (image_width - scaled_width) // 2

    transform = A.Compose([
        A.LongestMaxSize(max_size=max(image_height, image_width)),
        A.PadIfNeeded(min_height=image_height, min_width=image_width, border_mode=0),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0), image, (original_height, original_width, scaled_height, scaled_width, pad_top, pad_left)


def predict(model, image_tensor, device, original_size, threshold=0.5):
    """Run inference on an image and resize to original dimensions."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > threshold).float()

    # Get prediction as numpy array
    prediction_np = prediction.squeeze().cpu().numpy()

    # Unpack dimensions
    original_height, original_width, scaled_height, scaled_width, pad_top, pad_left = original_size

    # Crop to remove padding (extract the actual scaled content from center)
    prediction_cropped = prediction_np[pad_top:pad_top+scaled_height, pad_left:pad_left+scaled_width]

    # Resize back to original dimensions
    prediction_resized = Image.fromarray((prediction_cropped * 255).astype(np.uint8))
    prediction_resized = prediction_resized.resize((original_width, original_height), Image.NEAREST)
    prediction_resized = np.array(prediction_resized) / 255.0

    return prediction_resized


def visualize_results(original_image, prediction, save_path=None):
    """Visualize original image and prediction side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained U-Net model")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="saved_models/checkpoint_epoch_20.pth.tar",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output visualization (if not provided, will display)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary segmentation (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    print(f"Processing image: {args.image}")
    image_tensor, original_image, original_size = preprocess_image(args.image)

    print("Running inference...")
    prediction = predict(model, image_tensor, args.device, original_size, args.threshold)

    print("Visualizing results...")
    visualize_results(original_image, prediction, args.output)

    if args.output:
        # Also save the mask as a separate file
        mask_path = args.output.replace(".png", "_mask.png")
        Image.fromarray((prediction * 255).astype(np.uint8)).save(mask_path)
        print(f"Mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
