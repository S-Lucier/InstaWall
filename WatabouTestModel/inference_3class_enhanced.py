import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "Model Imports"))

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import argparse
from skimage import morphology

from unet_attention_aspp import AttentionASPPUNet


def load_model(checkpoint_path, device):
    """Load trained 3-class AttentionASPPUNet model from checkpoint."""
    model = AttentionASPPUNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def post_process_3class(prediction, wall_closing=3, floor_closing=5,
                        remove_small_walls=50, remove_small_floors=50):
    """Apply class-specific post-processing to 3-class predictions.

    Args:
        prediction: (H, W) array with class indices (0=walls, 1=doors, 2=floors)
        wall_closing: Morphological closing size for walls (fills small gaps)
        floor_closing: Morphological closing size for floors (fills small gaps)
        remove_small_walls: Remove isolated wall regions smaller than this (pixels)
        remove_small_floors: Remove isolated floor regions smaller than this (pixels)

    Returns:
        Cleaned prediction with same shape

    Strategy:
        - Walls/floors: Moderate cleanup (large regions, can handle aggressive ops)
        - Doors: NO post-processing (tiny 1-2 grid cells, preserve exactly)
    """
    # Separate classes into binary masks
    walls = (prediction == 0)
    doors = (prediction == 1)  # Keep doors untouched
    floors = (prediction == 2)

    # Post-process walls
    if wall_closing > 0:
        kernel = morphology.disk(wall_closing)
        walls = morphology.closing(walls, kernel)

    if remove_small_walls > 0:
        walls = morphology.remove_small_objects(walls, min_size=remove_small_walls)

    # Post-process floors
    if floor_closing > 0:
        kernel = morphology.disk(floor_closing)
        floors = morphology.closing(floors, kernel)

    if remove_small_floors > 0:
        floors = morphology.remove_small_objects(floors, min_size=remove_small_floors)

    # Recombine classes with priority: doors > floors > walls
    # (Doors are small and rare, preserve them at highest priority)
    cleaned = np.zeros_like(prediction, dtype=np.uint8)
    cleaned[walls] = 0
    cleaned[floors] = 2
    cleaned[doors] = 1  # Doors overwrite anything else

    return cleaned


def visualize_3class_enhanced(original_image, prediction_raw, prediction_processed, save_path=None):
    """Visualize original image, raw prediction, and post-processed result."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Helper function to create colored masks
    def create_colored_mask(pred):
        colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
        colored[pred == 0] = [0, 0, 0]        # walls = black
        colored[pred == 1] = [255, 0, 0]      # doors = red
        colored[pred == 2] = [255, 255, 255]  # floors = white
        return colored

    # Row 1: Original, raw prediction, processed prediction
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(create_colored_mask(prediction_raw))
    axes[0, 1].set_title("Raw Prediction\n(Black=Walls, Red=Doors, White=Floors)", fontsize=14)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(create_colored_mask(prediction_processed))
    axes[0, 2].set_title("Post-Processed\n(Cleaned Walls/Floors, Doors Preserved)", fontsize=14)
    axes[0, 2].axis("off")

    # Row 2: Door overlays
    overlay_raw = original_image.copy()
    door_mask_raw = prediction_raw == 1
    overlay_raw[door_mask_raw] = overlay_raw[door_mask_raw] * 0.5 + np.array([255, 0, 0]) * 0.5

    overlay_processed = original_image.copy()
    door_mask_processed = prediction_processed == 1
    overlay_processed[door_mask_processed] = overlay_processed[door_mask_processed] * 0.5 + np.array([255, 0, 0]) * 0.5

    axes[1, 0].imshow(overlay_raw)
    axes[1, 0].set_title("Raw - Doors Highlighted", fontsize=14)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_processed)
    axes[1, 1].set_title("Processed - Doors Highlighted", fontsize=14)
    axes[1, 1].axis("off")

    # Difference visualization
    diff = np.zeros((*prediction_raw.shape, 3), dtype=np.uint8)
    changed_pixels = prediction_raw != prediction_processed
    diff[changed_pixels] = [255, 255, 0]  # Yellow = changed by post-processing
    diff[~changed_pixels] = [50, 50, 50]  # Dark gray = unchanged

    axes[1, 2].imshow(diff)
    axes[1, 2].set_title(f"Post-Processing Changes\n(Yellow = modified, {changed_pixels.sum():,} pixels)", fontsize=14)
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="3-class segmentation inference with post-processing")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str,
                       default="saved_models_3class/3class_aten_multi_1450.pth.tar",
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to save output visualization")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")

    # Post-processing options
    parser.add_argument("--no-postprocess", action="store_true",
                       help="Disable post-processing")
    parser.add_argument("--wall-closing", type=int, default=3,
                       help="Wall closing kernel size (default: 3)")
    parser.add_argument("--floor-closing", type=int, default=5,
                       help="Floor closing kernel size (default: 5)")
    parser.add_argument("--remove-small-walls", type=int, default=50,
                       help="Remove wall regions smaller than N pixels (default: 50)")
    parser.add_argument("--remove-small-floors", type=int, default=50,
                       help="Remove floor regions smaller than N pixels (default: 50)")

    args = parser.parse_args()

    print(f"Loading 3-class model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    print(f"Processing image: {args.image}")
    image_tensor, original_image, original_size = preprocess_image(args.image)

    print("Running 3-class inference...")
    prediction_raw = predict_3class(model, image_tensor, args.device, original_size)

    print(f"Raw prediction class distribution:")
    print(f"  Walls (0):  {(prediction_raw == 0).sum():,} pixels ({100*(prediction_raw == 0).sum()/prediction_raw.size:.2f}%)")
    print(f"  Doors (1):  {(prediction_raw == 1).sum():,} pixels ({100*(prediction_raw == 1).sum()/prediction_raw.size:.2f}%)")
    print(f"  Floors (2): {(prediction_raw == 2).sum():,} pixels ({100*(prediction_raw == 2).sum()/prediction_raw.size:.2f}%)")

    # Apply post-processing
    prediction_processed = None
    if not args.no_postprocess:
        print("Applying class-specific post-processing...")
        print(f"  Walls: closing={args.wall_closing}, remove_small={args.remove_small_walls}")
        print(f"  Floors: closing={args.floor_closing}, remove_small={args.remove_small_floors}")
        print(f"  Doors: NO post-processing (preserved exactly)")

        prediction_processed = post_process_3class(
            prediction_raw,
            wall_closing=args.wall_closing,
            floor_closing=args.floor_closing,
            remove_small_walls=args.remove_small_walls,
            remove_small_floors=args.remove_small_floors
        )

        print(f"Post-processed class distribution:")
        print(f"  Walls (0):  {(prediction_processed == 0).sum():,} pixels ({100*(prediction_processed == 0).sum()/prediction_processed.size:.2f}%)")
        print(f"  Doors (1):  {(prediction_processed == 1).sum():,} pixels ({100*(prediction_processed == 1).sum()/prediction_processed.size:.2f}%)")
        print(f"  Floors (2): {(prediction_processed == 2).sum():,} pixels ({100*(prediction_processed == 2).sum()/prediction_processed.size:.2f}%)")

    print("Visualizing results...")
    if prediction_processed is not None:
        visualize_3class_enhanced(original_image, prediction_raw, prediction_processed, args.output)
    else:
        # Just show raw if no post-processing
        from WatabouTestModel.inference_3class import visualize_3class
        visualize_3class(original_image, prediction_raw, args.output)

    if args.output:
        # Save the raw mask
        raw_mask_path = args.output.replace(".png", "_raw_mask.png")
        mask_raw = np.zeros_like(prediction_raw, dtype=np.uint8)
        mask_raw[prediction_raw == 0] = 0
        mask_raw[prediction_raw == 1] = 127
        mask_raw[prediction_raw == 2] = 255
        Image.fromarray(mask_raw).save(raw_mask_path)
        print(f"Raw mask saved to: {raw_mask_path}")

        # Save the processed mask if available
        if prediction_processed is not None:
            processed_mask_path = args.output.replace(".png", "_processed_mask.png")
            mask_processed = np.zeros_like(prediction_processed, dtype=np.uint8)
            mask_processed[prediction_processed == 0] = 0
            mask_processed[prediction_processed == 1] = 127
            mask_processed[prediction_processed == 2] = 255
            Image.fromarray(mask_processed).save(processed_mask_path)
            print(f"Processed mask saved to: {processed_mask_path}")


if __name__ == "__main__":
    main()
