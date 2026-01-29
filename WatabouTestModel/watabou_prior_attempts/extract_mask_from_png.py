"""
Extract training masks directly from Watabou PNG files using image processing.
This avoids the JSON coordinate mapping problem.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def extract_mask_from_watabou_png(png_path, debug=False):
    """
    Extract playable space mask from Watabou dungeon PNG using image processing.

    Watabou dungeons have consistent visual features:
    - Rooms: Lighter colored floors (beige/gray)
    - Walls: Darker/different colored (black outlines, stone texture)
    - Water: Blue tint

    Returns:
        mask: Binary mask where 0=playable, 255=walls/exterior
    """
    # Load image
    img = cv2.imread(str(png_path))
    if img is None:
        raise ValueError(f"Could not load image: {png_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Watabou rooms are typically lighter than walls
    # Use Otsu's thresholding to automatically find threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The binary might be inverted depending on background
    # Count white vs black pixels to determine
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    if white_pixels > black_pixels:
        # Most of image is white (probably exterior), rooms are black
        # Invert so rooms become white
        binary = 255 - binary

    # Clean up noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Fill small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Final mask: 0 = playable (black), 255 = walls/exterior (white)
    mask = 255 - binary

    if debug:
        # Save debug images
        debug_dir = Path(png_path).parent / "debug"
        debug_dir.mkdir(exist_ok=True)

        base_name = Path(png_path).stem
        cv2.imwrite(str(debug_dir / f"{base_name}_gray.png"), gray)
        cv2.imwrite(str(debug_dir / f"{base_name}_binary.png"), binary)
        cv2.imwrite(str(debug_dir / f"{base_name}_mask.png"), mask)

        # Create overlay
        overlay = img.copy()
        overlay[mask == 0] = [0, 255, 0]  # Green for playable
        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        cv2.imwrite(str(debug_dir / f"{base_name}_overlay.png"), blended)

    return mask


def process_watabou_pngs(input_dir, output_dir, debug=False):
    """
    Process all Watabou PNGs in a directory and extract masks.

    Args:
        input_dir: Directory containing PNG files
        output_dir: Where to save images and masks
        debug: Save debug visualizations
    """
    input_path = Path(input_dir)

    # Find all PNG files
    png_files = list(input_path.glob("*.png"))

    print(f"Found {len(png_files)} PNG files")

    # Create output directories
    img_dir = Path(output_dir) / "images"
    mask_dir = Path(output_dir) / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for png_file in png_files:
        try:
            # Extract mask
            mask = extract_mask_from_watabou_png(png_file, debug=debug)

            # Copy original image
            img = Image.open(png_file)
            img_output = img_dir / png_file.name
            img.save(img_output)

            # Save mask
            mask_output = mask_dir / png_file.name
            Image.fromarray(mask).save(mask_output)

            print(f"Processed: {png_file.stem}")
            processed += 1

        except Exception as e:
            print(f"Error processing {png_file.name}: {e}")

    print(f"\nProcessed {processed}/{len(png_files)} dungeons successfully")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract masks from Watabou PNGs using CV")
    parser.add_argument("input_dir", help="Directory containing PNG files")
    parser.add_argument("--output", default="data/watabou_cv", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Save debug visualizations")

    args = parser.parse_args()

    process_watabou_pngs(args.input_dir, args.output, args.debug)

    print("\nNext steps:")
    print("1. Check the masks look correct")
    print("2. If debug enabled, check debug/ folder for overlays")
    print("3. Split data and train model")
