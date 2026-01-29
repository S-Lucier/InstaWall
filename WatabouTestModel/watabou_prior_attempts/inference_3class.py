import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Model Imports"))

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import argparse

from unet_model import UNet


def load_model(checkpoint_path, device):
    """Load trained 3-class U-Net model from checkpoint."""
    model = UNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def preprocess_image(image_path, image_height=512, image_width=512):
    """Load and preprocess an image for inference."""
    image = np.array(Image.open(image_path).convert("RGB"))
    original_height, original_width = image.shape[:2]

    # Calculate scaled dimensions
    scale = max(image_height, image_width) / max(original_height, original_width)
    scaled_height = int(original_height * scale)
    scaled_width = int(original_width * scale)

    # Calculate padding
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


def predict_3class(model, image_tensor, device, original_size):
    """Run 3-class inference and resize to original dimensions."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)  # Shape: (1, 3, H, W)
        prediction = torch.argmax(prediction, dim=1)  # Shape: (1, H, W) with class indices

    # Get prediction as numpy array
    prediction_np = prediction.squeeze().cpu().numpy()

    # Unpack dimensions
    original_height, original_width, scaled_height, scaled_width, pad_top, pad_left = original_size

    # Crop to remove padding
    prediction_cropped = prediction_np[pad_top:pad_top+scaled_height, pad_left:pad_left+scaled_width]

    # Resize back to original dimensions
    prediction_resized = Image.fromarray(prediction_cropped.astype(np.uint8))
    prediction_resized = prediction_resized.resize((original_width, original_height), Image.NEAREST)
    prediction_resized = np.array(prediction_resized)

    return prediction_resized


def visualize_3class(original_image, prediction, save_path=None):
    """Visualize original image and 3-class prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Class prediction (color-coded)
    # 0=walls (black), 1=doors (red), 2=floors (white)
    colored_prediction = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    colored_prediction[prediction == 0] = [0, 0, 0]      # walls = black
    colored_prediction[prediction == 1] = [255, 0, 0]    # doors = red
    colored_prediction[prediction == 2] = [255, 255, 255]  # floors = white

    axes[1].imshow(colored_prediction)
    axes[1].set_title("3-Class Prediction\n(Black=Walls, Red=Doors, White=Floors)")
    axes[1].axis("off")

    # Overlay on original
    overlay = original_image.copy()
    door_mask = prediction == 1
    overlay[door_mask] = overlay[door_mask] * 0.5 + np.array([255, 0, 0]) * 0.5

    axes[2].imshow(overlay)
    axes[2].set_title("Doors Highlighted (Red Overlay)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="3-class segmentation inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str,
                       default="saved_models_3class/checkpoint_epoch_5.pth.tar",
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to save output visualization")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")

    args = parser.parse_args()

    print(f"Loading 3-class model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    print(f"Processing image: {args.image}")
    image_tensor, original_image, original_size = preprocess_image(args.image)

    print("Running 3-class inference...")
    prediction = predict_3class(model, image_tensor, args.device, original_size)

    print(f"Class distribution in prediction:")
    print(f"  Walls (0):  {(prediction == 0).sum():,} pixels ({100*(prediction == 0).sum()/prediction.size:.2f}%)")
    print(f"  Doors (1):  {(prediction == 1).sum():,} pixels ({100*(prediction == 1).sum()/prediction.size:.2f}%)")
    print(f"  Floors (2): {(prediction == 2).sum():,} pixels ({100*(prediction == 2).sum()/prediction.size:.2f}%)")

    print("Visualizing results...")
    visualize_3class(original_image, prediction, args.output)

    if args.output:
        # Save the class mask as grayscale (0, 127, 255)
        mask_path = args.output.replace(".png", "_mask.png")
        mask_output = np.zeros_like(prediction, dtype=np.uint8)
        mask_output[prediction == 0] = 0    # walls
        mask_output[prediction == 1] = 127  # doors
        mask_output[prediction == 2] = 255  # floors
        Image.fromarray(mask_output).save(mask_path)
        print(f"Class mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
