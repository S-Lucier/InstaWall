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
from skimage import morphology

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


def post_process_mask(mask, fill_holes=True, remove_small_objects=True,
                      min_size=50, closing_size=5, opening_size=2):
    """Apply post-processing to clean up the predicted mask.

    Args:
        mask: Binary mask (0s and 1s)
        fill_holes: Fill small holes inside playable areas
        remove_small_objects: Remove small isolated regions
        min_size: Minimum size for connected components (pixels)
        closing_size: Kernel size for morphological closing (fills holes)
        opening_size: Kernel size for morphological opening (removes noise)

    Returns:
        Cleaned binary mask
    """
    mask_binary = (mask > 0.5).astype(bool)

    # Morphological closing: fills small holes/gaps in playable areas
    if fill_holes and closing_size > 0:
        kernel = morphology.disk(closing_size)
        mask_binary = morphology.closing(mask_binary, kernel)

    # Morphological opening: removes small noise/specks
    if opening_size > 0:
        kernel = morphology.disk(opening_size)
        mask_binary = morphology.opening(mask_binary, kernel)

    # Remove small disconnected regions
    if remove_small_objects and min_size > 0:
        mask_binary = morphology.remove_small_objects(mask_binary, max_size=min_size)

    return mask_binary.astype(np.float32)


def visualize_results(original_image, prediction, prediction_processed=None, save_path=None):
    """Visualize original image and prediction(s) side by side."""
    num_plots = 3 if prediction_processed is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    if num_plots == 2:
        axes = [axes[0], axes[1]]

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title("Raw Prediction")
    axes[1].axis("off")

    if prediction_processed is not None:
        axes[2].imshow(prediction_processed, cmap="gray")
        axes[2].set_title("Post-Processed")
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Enhanced inference with post-processing")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="saved_models/checkpoint_epoch_800.pth.tar",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output visualization"
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

    # Post-processing options
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable post-processing"
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        default=True,
        help="Fill small holes in playable areas (default: True)"
    )
    parser.add_argument(
        "--closing-size",
        type=int,
        default=5,
        help="Morphological closing kernel size (default: 5)"
    )
    parser.add_argument(
        "--opening-size",
        type=int,
        default=2,
        help="Morphological opening kernel size (default: 2)"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=50,
        help="Minimum size for connected components in pixels (default: 50)"
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    print(f"Processing image: {args.image}")
    image_tensor, original_image, original_size = preprocess_image(args.image)

    print("Running inference...")
    prediction = predict(model, image_tensor, args.device, original_size, args.threshold)

    # Apply post-processing if enabled
    prediction_processed = None
    if not args.no_postprocess:
        print("Applying post-processing...")
        prediction_processed = post_process_mask(
            prediction,
            fill_holes=args.fill_holes,
            remove_small_objects=True,
            min_size=args.min_size,
            closing_size=args.closing_size,
            opening_size=args.opening_size
        )

    print("Visualizing results...")
    visualize_results(original_image, prediction, prediction_processed, args.output)

    if args.output:
        # Save the raw mask
        raw_mask_path = args.output.replace(".png", "_raw_mask.png")
        Image.fromarray((prediction * 255).astype(np.uint8)).save(raw_mask_path)
        print(f"Raw mask saved to: {raw_mask_path}")

        # Save the processed mask if available
        if prediction_processed is not None:
            processed_mask_path = args.output.replace(".png", "_processed_mask.png")
            Image.fromarray((prediction_processed * 255).astype(np.uint8)).save(processed_mask_path)
            print(f"Processed mask saved to: {processed_mask_path}")


if __name__ == "__main__":
    main()
